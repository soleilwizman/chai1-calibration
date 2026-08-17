"""Attach gene age and pretraining-representation covariates to a scored ORF set.

Joins three things the analysis needs and the ORF table does not carry:

* **gene age** -- origination time, from GenOrigin, GenTree, phylostratr output,
  or any table with a gene identifier and an age. Column names vary between all
  of these, so they are configurable rather than assumed.
* **pretraining representation** -- percent identity to the nearest UniRef50
  neighbour, the covariate that separates "the classifier fails on young genes"
  from "the classifier succeeds on old genes because it memorised them".
  Produced upstream by MMseqs2/DIAMOND against UniRef50; this only joins it.
* **age strata** -- coarse bins for the per-stratum tables.

A warning about what phylostratigraphic age *is*, because it determines what the
final result can claim. Standard phylostratigraphy dates a gene by failure to
detect homologs, and Moyers & Zhang showed that short, fast-evolving genes are
systematically misdated as young for exactly that reason. The bias points the
same way as the hypothesis under test, so a correlation between "young" and
"classifier fails" is partly guaranteed by construction. Synteny-based ages
(fagin) do not share this failure mode; when available, prefer them, and
``--age-method`` records which was used so the write-up cannot forget.

Examples
--------
    python scripts/orf/join_gene_age.py --orfs data/orf/hsap.csv \
        --ages data/orf/genorigin_hsap.tsv --age-col gene_age \
        --out data/orf/hsap.annotated.csv

    # with the pretraining-representation covariate
    python scripts/orf/join_gene_age.py --orfs data/orf/hsap.csv \
        --ages data/orf/ages.tsv --uniref data/orf/hsap.uniref50.tsv \
        --out data/orf/hsap.annotated.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Coarse strata in millions of years. The boundaries are conventional rather than
# principled: ~100 Mya separates lineage-specific genes from mammal-wide ones, and
# ~500 Mya sits near the base of the metazoa. Override with --strata when the
# species demands it -- these are wrong for yeast or for plants.
DEFAULT_STRATA = ((0, 100, "young"), (100, 500, "mid"), (500, 1e9, "old"))


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV, guessing the delimiter from the suffix."""
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def assign_strata(ages: pd.Series,
                  strata: Sequence[tuple] = DEFAULT_STRATA) -> pd.Series:
    """Bin continuous ages into named strata; NaN ages stay NaN."""
    result = pd.Series(np.nan, index=ages.index, dtype=object)
    for low, high, name in strata:
        result = result.mask(ages.notna() & (ages >= low) & (ages < high), name)
    return result


def parse_strata(spec: Optional[str]) -> Sequence[tuple]:
    """Parse ``0:100:young,100:500:mid,500:inf:old``."""
    if not spec:
        return DEFAULT_STRATA
    out: List[tuple] = []
    for chunk in spec.split(","):
        low, high, name = chunk.split(":")
        out.append((float(low), float("inf") if high == "inf" else float(high), name))
    return tuple(out)


def join(
    orfs: pd.DataFrame,
    ages: Optional[pd.DataFrame],
    uniref: Optional[pd.DataFrame],
    gene_key: str,
    age_gene_col: str,
    age_col: str,
    uniref_key: str,
    uniref_col: str,
    strata: Sequence[tuple],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    frame = orfs.copy()
    stats: Dict[str, object] = {"n_orfs": int(len(frame))}

    if ages is not None:
        missing = {age_gene_col, age_col} - set(ages.columns)
        if missing:
            raise SystemExit(
                f"age table is missing {sorted(missing)}; it has {list(ages.columns)}")
        # Strip identifier versions on both sides: age tables are inconsistent
        # about carrying them, and a silent zero-match join is the failure mode.
        lookup = ages[[age_gene_col, age_col]].copy()
        lookup[age_gene_col] = lookup[age_gene_col].astype(str).str.split(".").str[0]
        lookup = lookup.drop_duplicates(subset=age_gene_col)
        frame[gene_key] = frame[gene_key].astype(str).str.split(".").str[0]

        frame = frame.merge(
            lookup.rename(columns={age_gene_col: gene_key, age_col: "age_mya"}),
            on=gene_key, how="left")
        frame["age_stratum"] = assign_strata(frame["age_mya"], strata)

        matched = int(frame["age_mya"].notna().sum())
        stats["age_matched"] = matched
        stats["age_match_rate"] = round(matched / len(frame), 4) if len(frame) else 0.0
        stats["stratum_counts"] = {
            str(k): int(v) for k, v in frame["age_stratum"].value_counts().items()
        }
        # Coding sequences are what the age analysis runs on; a good overall rate
        # can still hide a total miss there.
        if "label" in frame.columns:
            coding = frame[frame["label"].astype(bool)]
            stats["age_match_rate_coding"] = round(
                float(coding["age_mya"].notna().mean()), 4) if len(coding) else 0.0
        if stats["age_match_rate"] < 0.5:
            stats["age_warning"] = (
                f"only {stats['age_match_rate']:.1%} of ORFs matched an age; check "
                f"that {gene_key!r} and {age_gene_col!r} use the same identifier "
                "namespace (Ensembl gene vs transcript vs symbol)"
            )

    if uniref is not None:
        missing = {uniref_key, uniref_col} - set(uniref.columns)
        if missing:
            raise SystemExit(
                f"uniref table is missing {sorted(missing)}; "
                f"it has {list(uniref.columns)}")
        lookup = uniref[[uniref_key, uniref_col]].drop_duplicates(subset=uniref_key)
        frame = frame.merge(
            lookup.rename(columns={uniref_key: "sequence_id",
                                   uniref_col: "uniref50_identity"}),
            on="sequence_id", how="left")
        # No hit means no detectable homolog in the pretraining set, which is a
        # measurement of zero identity rather than a missing measurement. Leaving
        # it NaN would drop exactly the orphans the study is about.
        n_missing = int(frame["uniref50_identity"].isna().sum())
        frame["uniref50_identity"] = frame["uniref50_identity"].fillna(0.0)
        stats["uniref_no_hit"] = n_missing
        stats["uniref_no_hit_filled_as_zero"] = True

    return frame, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--orfs", required=True, type=Path)
    parser.add_argument("--ages", type=Path)
    parser.add_argument("--uniref", type=Path,
                        help="table of nearest-UniRef50 identity per sequence_id")
    parser.add_argument("--gene-key", default="gene_id",
                        help="gene identifier column in the ORF table")
    parser.add_argument("--age-gene-col", default="gene_id",
                        help="gene identifier column in the age table")
    parser.add_argument("--age-col", default="gene_age",
                        help="age column in the age table (millions of years)")
    parser.add_argument("--uniref-key", default="sequence_id")
    parser.add_argument("--uniref-col", default="pident")
    parser.add_argument("--strata", default=None,
                        help="e.g. 0:100:young,100:500:mid,500:inf:old")
    parser.add_argument("--age-method", default="phylostratigraphy",
                        choices=("phylostratigraphy", "synteny", "other"),
                        help="recorded in the stats; synteny-based ages avoid the "
                             "homology-detection bias that inflates this study")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    orfs = read_table(args.orfs)
    ages = read_table(args.ages) if args.ages else None
    uniref = read_table(args.uniref) if args.uniref else None

    frame, stats = join(
        orfs, ages, uniref, args.gene_key, args.age_gene_col, args.age_col,
        args.uniref_key, args.uniref_col, parse_strata(args.strata),
    )
    stats["age_method"] = args.age_method
    if args.age_method == "phylostratigraphy":
        stats["age_method_caveat"] = (
            "BLAST-based ages misdate short, fast-evolving genes as young "
            "(Moyers & Zhang); this bias is aligned with the hypothesis and "
            "bounds what the result can claim"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    stats_path = args.out.with_suffix(".join.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.out} ({len(frame)} rows)")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
