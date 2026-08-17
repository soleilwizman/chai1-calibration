"""Stage 1 -- turn annotated transcripts into a labelled, length-matched ORF set.

Reproduces the plm-utils dataset construction closely enough to serve as its
baseline: take the longest ORF per transcript across five start codons, translate
it, label it by the transcript's biotype (coding cDNA vs noncoding ncRNA), and
emit one row per putative peptide.

The one deliberate departure is that the noncoding class is **length-matched** to
the coding class by default (ProtiGeno's control, absent from plm-utils). Without
it, ORF length alone separates the classes well, and any subsequent claim about
what the protein language model contributes is confounded from the start. Pass
``--no-match`` to reproduce the unmatched setup for comparison -- the difference
between the two is itself worth reporting.

Inputs are Ensembl-style FASTA files. Note that stage 0 (fetching them) requires
network access to ftp.ensembl.org; see ``scripts/orf/README.md`` for the exact
URLs and for which hosts are blocked in the sandboxed environment.

Examples
--------
    python scripts/orf/build_orf_dataset.py \
        --coding data/orf/raw/hsap.cdna.fa.gz \
        --noncoding data/orf/raw/hsap.ncrna.fa.gz \
        --species hsap --out data/orf/hsap.csv

    # unmatched, to quantify how much of the signal is just length
    python scripts/orf/build_orf_dataset.py ... --no-match --out data/orf/hsap_unmatched.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orfkit import (  # noqa: E402
    DEFAULT_START_CODONS, ks_statistic, length_matched_indices,
    match_groups_common_support, longest_orf, read_fasta,
)


def parse_ensembl_header(header: str) -> Dict[str, str]:
    """Pull transcript/gene identifiers out of an Ensembl FASTA header.

    Ensembl headers look like::

        ENST00000631435.1 cdna chromosome:GRCh38:CHR_HSCHR7_2_CTG6:... gene:ENSG00000282591.1 ...

    The gene ID is what a gene-age table joins on, so it is extracted separately;
    versions are stripped because age tables are inconsistent about carrying them.
    """
    fields = header.split()
    record = {"transcript_id": fields[0].split(".")[0] if fields else ""}
    for field in fields[1:]:
        if ":" not in field:
            continue
        key, _, value = field.partition(":")
        if key in {"gene", "gene_biotype", "transcript_biotype", "gene_symbol"}:
            record[key] = value.split(".")[0] if key == "gene" else value
    return record


def extract_orfs(
    path: Path,
    label: int,
    start_codons: Sequence[str],
    require_stop: bool,
    both_strands: bool,
    min_aa: int,
    max_aa: Optional[int],
) -> pd.DataFrame:
    """Longest ORF per transcript, as a table. Transcripts with none are dropped."""
    rows: List[Dict[str, object]] = []
    n_transcripts = 0
    for header, sequence in read_fasta(path):
        n_transcripts += 1
        orf = longest_orf(
            sequence, start_codons=start_codons, require_stop=require_stop,
            both_strands=both_strands, min_aa=min_aa,
        )
        if orf is None:
            continue
        if max_aa is not None and orf.aa_length > max_aa:
            continue
        record = parse_ensembl_header(header)
        rows.append({
            "transcript_id": record.get("transcript_id", ""),
            "gene_id": record.get("gene", ""),
            "biotype": record.get("transcript_biotype", ""),
            "label": label,
            "aa_length": orf.aa_length,
            "start_codon": orf.start_codon,
            "strand": orf.strand,
            "transcript_length": len(sequence),
            "protein": orf.protein,
        })
    frame = pd.DataFrame(rows)
    frame.attrs["n_transcripts"] = n_transcripts
    return frame


def build(
    coding_path: Path,
    noncoding_path: Path,
    species: str,
    start_codons: Sequence[str],
    require_stop: bool,
    both_strands: bool,
    min_aa: int,
    max_aa: Optional[int],
    match: str,
    bin_width: int,
    seed: int,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    coding = extract_orfs(coding_path, 1, start_codons, require_stop,
                          both_strands, min_aa, max_aa)
    noncoding = extract_orfs(noncoding_path, 0, start_codons, require_stop,
                             both_strands, min_aa, max_aa)

    if coding.empty or noncoding.empty:
        raise SystemExit(
            f"no ORFs found (coding={len(coding)}, noncoding={len(noncoding)}); "
            "check the input FASTAs and --min-aa"
        )

    stats: Dict[str, object] = {
        "species": species,
        "coding_transcripts": coding.attrs.get("n_transcripts"),
        "noncoding_transcripts": noncoding.attrs.get("n_transcripts"),
        "coding_orfs": int(len(coding)),
        "noncoding_orfs": int(len(noncoding)),
        "median_aa_coding": float(coding["aa_length"].median()),
        "median_aa_noncoding": float(noncoding["aa_length"].median()),
        "raw_length_ks": round(
            ks_statistic(coding["aa_length"], noncoding["aa_length"]), 4),
        "length_matched": match,
        "orf_definition": {
            "start_codons": list(start_codons),
            "require_stop": require_stop,
            "both_strands": both_strands,
            "min_aa": min_aa,
            "max_aa": max_aa,
        },
    }

    if match == "sample":
        idx, report = length_matched_indices(
            coding["aa_length"].to_numpy(), noncoding["aa_length"].to_numpy(),
            bin_width=bin_width, seed=seed,
        )
        noncoding = noncoding.iloc[idx]
        stats["match_report"] = report.as_dict()
        if not report.complete:
            stats["match_warning"] = (
                "the noncoding pool could not supply the full coding length "
                "distribution; residual length imbalance remains and any "
                "classifier gain may partly reflect it. Long ORFs essentially do "
                "not occur by chance in noncoding sequence, so this shortfall is "
                "structural -- use --match common-support to restrict to the "
                "length range where both classes actually exist."
            )
        combined = pd.concat([coding, noncoding], ignore_index=True)
    elif match == "common-support":
        combined, report = match_groups_common_support(
            pd.concat([coding, noncoding], ignore_index=True),
            "label", "aa_length", bin_width=bin_width, seed=seed,
        )
        combined = combined.reset_index(drop=True)
        stats["match_report"] = report
    else:
        combined = pd.concat([coding, noncoding], ignore_index=True)
    combined.insert(0, "species", species)
    combined.insert(1, "sequence_id",
                    combined["species"] + "|" + combined["transcript_id"])
    stats["final_rows"] = int(len(combined))
    stats["final_length_ks"] = round(ks_statistic(
        combined.loc[combined["label"] == 1, "aa_length"],
        combined.loc[combined["label"] == 0, "aa_length"],
    ), 4)
    return combined, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coding", required=True, type=Path,
                        help="cDNA FASTA (coding transcripts)")
    parser.add_argument("--noncoding", required=True, type=Path,
                        help="ncRNA FASTA (noncoding transcripts)")
    parser.add_argument("--species", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start-codons", default=",".join(DEFAULT_START_CODONS),
                        help="comma-separated; plm-utils uses ATG,TTG,CTG,GTG,ACG")
    parser.add_argument("--allow-stopless", action="store_true",
                        help="keep ORFs running off the transcript end")
    parser.add_argument("--both-strands", action="store_true",
                        help="search the reverse strand too (wrong for annotated "
                             "transcripts: it favours the noncoding class)")
    parser.add_argument("--min-aa", type=int, default=20)
    parser.add_argument("--max-aa", type=int, default=None)
    parser.add_argument("--match", choices=("sample", "common-support", "none"),
                        default="sample",
                        help="'sample': draw noncoding ORFs to match the coding "
                             "length distribution (ProtiGeno). 'common-support': "
                             "down-sample both classes per length bin, restricting "
                             "to the range where both exist -- exact, but discards "
                             "the long tail. 'none': reproduces plm-utils.")
    parser.add_argument("--bin-width", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fasta-out", type=Path, default=None,
                        help="also write the peptides as FASTA (for ESM scoring)")
    args = parser.parse_args(argv)

    frame, stats = build(
        args.coding, args.noncoding, args.species,
        tuple(c.strip().upper() for c in args.start_codons.split(",")),
        not args.allow_stopless, args.both_strands,
        args.min_aa, args.max_aa, args.match, args.bin_width, args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    if args.fasta_out:
        from orfkit import write_fasta
        write_fasta(args.fasta_out,
                    ((row.sequence_id, row.protein) for row in frame.itertuples()))
        print(f"wrote {args.fasta_out}")

    print(f"wrote {args.out} ({len(frame)} rows)")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
