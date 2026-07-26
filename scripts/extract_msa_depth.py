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

a3m format: lines starting with ``>`` are headers; sequence lines use uppercase /
``-`` for match columns and lowercase for insertions. Insertions (lowercase and
``.``) are stripped so every sequence has the query's column count.

Examples
--------
Single MSA::

    python scripts/extract_msa_depth.py --a3m predictions/10AF_1/msa/query.a3m

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


def compute_neff(seqs: List[str], seqid: float = 0.62) -> Dict[str, float]:
    """Depth metrics for a list of equal-length match-state sequences.

    Neff uses the standard HHblits-style weighting: a sequence's weight is the
    reciprocal of the number of sequences (including itself) that are at least
    ``seqid`` identical to it, summed over all sequences.
    """
    if not seqs:
        return {"n_sequences": 0, "neff": 0.0, "neff_per_col": 0.0, "length": 0}

    length = len(seqs[0])
    seqs = [s for s in seqs if len(s) == length]  # guard against ragged rows
    n = len(seqs)

    # Encode as an integer matrix; gaps map to a distinct symbol.
    alphabet = {c: i for i, c in enumerate(sorted(set("".join(seqs))))}
    mat = np.array([[alphabet[c] for c in s] for s in seqs], dtype=np.int16)

    # Pairwise fractional identity over columns, then weight = 1 / cluster size.
    # O(n^2 * length); fine for typical MSAs, chunked to bound memory.
    weights = np.zeros(n, dtype=float)
    thresh = seqid * length
    for i in range(n):
        matches = (mat == mat[i]).sum(axis=1)      # identity count vs every seq
        cluster = int((matches >= thresh).sum())   # neighbors incl. self
        weights[i] = 1.0 / cluster
    neff = float(weights.sum())
    return {
        "n_sequences": n,
        "neff": round(neff, 3),
        "neff_per_col": round(neff / length, 4) if length else 0.0,
        "length": length,
    }


def find_a3m(target_dir: Path) -> Optional[Path]:
    """Locate the query a3m under a target's prediction directory."""
    candidates = sorted(target_dir.rglob("*.a3m"))
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a3m", help="Single a3m file to summarize")
    parser.add_argument("--pred-dir", help="Root dir with one subdir per target holding an a3m")
    parser.add_argument("--out", help="Output JSON (list of {candidate, ...depth})")
    parser.add_argument("--seqid", type=float, default=0.62, help="Identity threshold for Neff")
    args = parser.parse_args()

    if args.a3m:
        stats = compute_neff(read_a3m(Path(args.a3m)), seqid=args.seqid)
        print(json.dumps(stats, indent=2))
        return 0

    if not args.pred_dir:
        parser.error("provide either --a3m or --pred-dir")

    pred_dir = Path(args.pred_dir)
    rows = []
    for target_dir in sorted(p for p in pred_dir.iterdir() if p.is_dir()):
        a3m = find_a3m(target_dir)
        if a3m is None:
            print(f"Warning: no a3m for {target_dir.name}", file=sys.stderr)
            continue
        stats = compute_neff(read_a3m(a3m), seqid=args.seqid)
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
