"""Assemble the analysis table: per-residue lDDT/pLDDT joined with covariates.

Final assembly step before ``calibration.py``. It:

1. concatenates every per-target CSV from ``compute_lddt.py`` (one file per
   candidate under ``--lddt-dir``) into one long per-residue table, tagging each
   row with its ``candidate``;
2. left-joins per-target covariates from the metadata already collected
   (``candidates_covariates.json``) and, when present, the MSA-depth and
   training-identity covariate files;
3. derives a few binned columns convenient for ``calibration.py --by``
   (apo/holo, MSA-depth tertiles, novelty tertiles).

The output (``data/analysis/all_residues.csv`` by default) is exactly what
``calibration.py --scores`` expects, and every covariate column can be passed to
``--by`` to test the hypotheses.

Example
-------
    python scripts/build_dataset.py \
        --lddt-dir data/analysis/per_target \
        --out data/analysis/all_residues.csv
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_lddt_tables(lddt_dir: Path) -> pd.DataFrame:
    """Concatenate per-target lDDT CSVs, adding a ``candidate`` column from the filename."""
    frames = []
    for csv in sorted(lddt_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        df["candidate"] = csv.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No per-target CSVs found in {lddt_dir}")
    return pd.concat(frames, ignore_index=True)


def _covariate_frame(path: Optional[Path], keep: List[str]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    data = json.load(path.open("r", encoding="utf-8"))
    df = pd.DataFrame(data)
    if "candidate" not in df.columns:
        return None
    cols = ["candidate"] + [c for c in keep if c in df.columns]
    return df[cols]


def _tertile_labels(series: pd.Series, names=("low", "mid", "high")) -> pd.Series:
    """Bin a numeric per-target covariate into tertiles by rank (NaN-safe)."""
    if series.notna().sum() < 3:
        return pd.Series([None] * len(series), index=series.index)
    try:
        return pd.qcut(series.rank(method="first"), 3, labels=list(names))
    except ValueError:
        return pd.Series([None] * len(series), index=series.index)


def build(lddt_dir: Path, covariates: Optional[Path], msa_depth: Optional[Path],
          training_identity: Optional[Path], min_coverage: float = 0.8) -> pd.DataFrame:
    df = load_lddt_tables(lddt_dir)

    # Drop targets whose prediction/reference alignment was poor -- their lDDT is
    # not trustworthy, and including them would quietly bias the calibration.
    if "coverage" in df.columns and min_coverage > 0:
        per_target = df.groupby("candidate")["coverage"].first()
        bad = sorted(per_target[per_target < min_coverage].index)
        if bad:
            print(f"Excluding {len(bad)} target(s) with coverage < {min_coverage:.0%}: "
                  f"{', '.join(bad[:10])}{' ...' if len(bad) > 10 else ''}")
            df = df[~df["candidate"].isin(bad)].reset_index(drop=True)

    meta = _covariate_frame(
        covariates,
        ["nonpolymer_entity_count", "rcsb_mutation_count", "entity_length",
         "rcsb_entity_polymer_type"],
    )
    depth = _covariate_frame(msa_depth, ["n_sequences", "neff", "neff_per_col"])
    novelty = _covariate_frame(training_identity, ["max_identity"])

    for cov in (meta, depth, novelty):
        if cov is not None:
            df = df.merge(cov, on="candidate", how="left")

    # Derived per-residue covariates (constant within a target) for --by.
    if "nonpolymer_entity_count" in df.columns:
        df["ligand_state"] = np.where(
            df["nonpolymer_entity_count"].fillna(0) > 0, "holo", "apo")

    # Tertiles are computed per-target (not per-residue) so bin edges aren't
    # dominated by big proteins contributing many rows.
    per_target = df.drop_duplicates("candidate").set_index("candidate")
    if "neff" in df.columns:
        bins = _tertile_labels(per_target["neff"])
        df["msa_depth_bin"] = df["candidate"].map(dict(zip(per_target.index, bins)))
    if "max_identity" in df.columns:
        bins = _tertile_labels(per_target["max_identity"])
        df["novelty_bin"] = df["candidate"].map(dict(zip(per_target.index, bins)))

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lddt-dir", default="data/analysis/per_target",
                        help="Directory of per-target lDDT CSVs from compute_lddt.py")
    parser.add_argument("--covariates", default="data/targets/candidates_covariates.json")
    parser.add_argument("--msa-depth", default="data/targets/msa_depth.json")
    parser.add_argument("--training-identity", default="data/targets/training_identity.json")
    parser.add_argument("--out", default="data/analysis/all_residues.csv")
    parser.add_argument("--min-coverage", type=float, default=0.8,
                        help="Drop targets whose alignment coverage is below this (0 = keep all)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def _opt(p: str) -> Optional[Path]:
        path = root / p
        return path if path.exists() else None

    df = build(
        lddt_dir=root / args.lddt_dir,
        covariates=_opt(args.covariates),
        msa_depth=_opt(args.msa_depth),
        training_identity=_opt(args.training_identity),
        min_coverage=args.min_coverage,
    )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} residues from {df['candidate'].nunique()} targets "
          f"to {out_path}")
    print(f"Columns available for --by: "
          f"{[c for c in df.columns if c not in ('lddt','plddt','res_id','n_atoms')]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
