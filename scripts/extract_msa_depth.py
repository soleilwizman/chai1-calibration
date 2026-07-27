"""Extract MSA-depth covariates from the alignments used for prediction.

Supports hypothesis 2 ("calibration degrades with MSA depth, faster than accuracy
does"). For each target's MSA (an ``.a3m`` file, as produced by the Chai-1 MSA
server in stage 3) this computes:

* ``n_sequences`` -- number of aligned sequences (raw depth).
* ``neff``        -- effective number of sequences: the sum of per-sequence
  weights ``1 / (# sequences within `--seqid` identity)``. This down-weights
  redundant homologs and is the depth measure that actually tracks how much
  independent evolutionary signal the model had.
* ``neff_per_col`` -- ``neff`` divided by query length, a length-normalized depth.

Two input formats are supported:

* **``.aligned.pqt``** -- what Chai-1 actually writes (``<output>/msas/*.pqt``): a
  parquet table with a ``sequence`` column of already-gap-aligned rows plus a
  ``source_database`` column (the query row is tagged ``query``). This is the
  preferred input because it is the exact alignment the model consumed.
* **``.a3m``** -- headers starting with ``>``; uppercase/``-`` are match columns
  and lowercase are insertions, which are stripped so every row has the query's
  column count.

Neff is O(n^2) in the number of sequences, and real MSAs run to tens of thousands
of rows, so it is computed on a random subsample of ``--max-seqs`` sequences and
rescaled to the full depth. ``n_sequences`` is always the exact, full count.

Examples
--------
Single MSA (either format)::

    python scripts/extract_msa_depth.py --msa predictions/10AF_1/output/msas/*.pqt

All targets under predictions/::

    python scripts/extract_msa_depth.py --pred-dir predictions \
        --out data/targets/msa_depth.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_INSERTION = re.compile(r"[a-z.]")


def read_a3m(path: Path) -> List[str]:
    """Return match-state sequences from an a3m file (insertions removed)."""
    seqs, cur = [], []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(_INSERTION.sub("", line))
    if cur:
        seqs.append("".join(cur))
    return seqs


def read_aligned_pqt(path: Path) -> List[str]:
    """Return aligned sequences from a Chai-1 ``.aligned.pqt`` parquet MSA.

    Chai-1 writes columns ``sequence, source_database, pairing_key, comment``;
    rows are already gap-aligned to the query, so no insertion stripping is
    needed. The query row (``source_database == 'query'``) is kept -- it is part
    of the depth, and Neff weighting expects it present.
    """
    import pandas as pd

    df = pd.read_parquet(path, columns=None)
    if "sequence" not in df.columns:
        raise ValueError(f"{path} has no 'sequence' column (found {list(df.columns)})")
    return [s for s in df["sequence"].astype(str).tolist() if s]


def read_msa(path: Path) -> List[str]:
    """Read either a ``.aligned.pqt`` or ``.a3m`` MSA into aligned sequences."""
    if path.suffix.lower() == ".pqt" or path.name.endswith(".aligned.pqt"):
        return read_aligned_pqt(path)
    return read_a3m(path)


def compute_neff(seqs: List[str], seqid: float = 0.62, max_seqs: int = 5000,
                 seed: int = 0) -> Dict[str, float]:
    """Depth metrics for a list of equal-length aligned sequences.

    Neff uses HHblits-style weighting: a sequence's weight is the reciprocal of
    the number of sequences (including itself) at least ``seqid`` identical to
    it, summed over all sequences.

    Because that is O(n^2 * length) and real MSAs reach tens of thousands of
    rows, Neff is computed on a random subsample of at most ``max_seqs``
    sequences and rescaled by ``n / n_sampled``. ``n_sequences`` is always the
    exact full count, and ``neff_estimated`` records whether subsampling kicked
    in.
    """
    if not seqs:
        return {"n_sequences": 0, "neff": 0.0, "neff_per_col": 0.0,
                "length": 0, "neff_estimated": False}

    length = len(seqs[0])
    seqs = [s for s in seqs if len(s) == length]  # guard against ragged rows
    n = len(seqs)

    sampled = seqs
    estimated = False
    if n > max_seqs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_seqs, replace=False)
        sampled = [seqs[i] for i in idx]
        estimated = True
    m = len(sampled)

    # Encode as an integer matrix; gaps map to a distinct symbol.
    alphabet = {c: i for i, c in enumerate(sorted(set("".join(sampled))))}
    mat = np.array([[alphabet[c] for c in s] for s in sampled], dtype=np.int16)

    # Chunked pairwise identity: weight = 1 / (# neighbours within `seqid`).
    thresh = seqid * length
    weights = np.zeros(m, dtype=float)
    chunk = max(1, min(256, m))
    for start in range(0, m, chunk):
        block = mat[start:start + chunk]                     # (c, L)
        matches = (block[:, None, :] == mat[None, :, :]).sum(axis=2)  # (c, m)
        clusters = (matches >= thresh).sum(axis=1)           # (c,)
        weights[start:start + chunk] = 1.0 / np.maximum(clusters, 1)

    neff = float(weights.sum())
    if estimated:
        neff *= n / m  # rescale subsample to full depth

    return {
        "n_sequences": n,
        "neff": round(neff, 3),
        "neff_per_col": round(neff / length, 4) if length else 0.0,
        "length": length,
        "neff_estimated": estimated,
    }


def find_msa(target_dir: Path) -> Optional[Path]:
    """Locate a target's MSA, preferring Chai-1's aligned parquet over a3m."""
    for pattern in ("*.aligned.pqt", "*.pqt", "*.a3m"):
        hits = sorted(target_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msa", "--a3m", dest="msa",
                        help="Single MSA file (.aligned.pqt or .a3m) to summarize")
    parser.add_argument("--pred-dir", help="Root dir with one subdir per target holding an MSA")
    parser.add_argument("--out", help="Output JSON (list of {candidate, ...depth})")
    parser.add_argument("--seqid", type=float, default=0.62, help="Identity threshold for Neff")
    parser.add_argument("--max-seqs", type=int, default=5000,
                        help="Subsample cap for the O(n^2) Neff computation")
    args = parser.parse_args()

    if args.msa:
        stats = compute_neff(read_msa(Path(args.msa)), seqid=args.seqid,
                             max_seqs=args.max_seqs)
        print(json.dumps(stats, indent=2))
        return 0

    if not args.pred_dir:
        parser.error("provide either --msa or --pred-dir")

    pred_dir = Path(args.pred_dir)
    rows = []
    for target_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        msa = find_msa(target_dir)
        if msa is None:
            print(f"Warning: no MSA for {target_dir.name}", file=sys.stderr)
            continue
        try:
            stats = compute_neff(read_msa(msa), seqid=args.seqid, max_seqs=args.max_seqs)
        except Exception as exc:  # one bad MSA shouldn't kill a 500-target run
            print(f"Warning: {target_dir.name} MSA unreadable ({exc})", file=sys.stderr)
            continue
        rows.append({"candidate": target_dir.name, **stats})

    payload = json.dumps(rows, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"Wrote MSA depth for {len(rows)} targets to {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
