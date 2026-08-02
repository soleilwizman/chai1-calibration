"""Per-residue ensemble agreement, and whether it predicts error better than pLDDT.

Chai-1 emits five diffusion samples per target; the pipeline scores only rank 0
and discards the rest. Those four extra models are a free second confidence
signal: where the samples agree, the model has converged on one answer; where
they disagree, it has not. Nothing about that signal is used by pLDDT.

This script computes, per residue, the mean pairwise Cα lDDT among all ten model
pairs (an *agreement* score in 0..1), then asks the question that matters: does
agreement predict realized accuracy better than the model's own stated
confidence does?

Residues are keyed by the *reference* chain and residue id — the same keys
``compute_lddt.py`` emits — by aligning the reference to model 0 and reusing that
alignment for all five. Predictions number residues 1..N from the input FASTA, so
joining on prediction numbering would silently mismatch wherever the experimental
structure numbers differently.

Examples
--------
Compute agreement for every target that has five models::

    python scripts/ensemble_agreement.py --pred-dir predictions \
        --ref-dir data/raw/cif --out data/analysis/ensemble_agreement.csv

Then compare its predictive power against pLDDT::

    python scripts/ensemble_agreement.py --compare \
        --agreement data/analysis/ensemble_agreement.csv \
        --scores data/analysis/all_residues.csv
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import biotite.structure as struc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_lddt import load_structure, _aligned_residue_pairs  # noqa: E402


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no SciPy dependency)."""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return float(s["x"].corr(s["y"], method="spearman"))


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank formula.

    ``positive`` marks the residues we want a *low* score to flag, so the score
    is negated: a good detector of bad residues gives them low values.
    """
    s = pd.DataFrame({"s": -np.asarray(score, float),
                      "p": np.asarray(positive, bool)}).dropna()
    npos, nneg = int(s["p"].sum()), int((~s["p"]).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = s["s"].rank()
    return float((ranks[s["p"]].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def target_agreement(ref_path: Path, model_paths: List[Path]) -> Optional[pd.DataFrame]:
    """Mean pairwise Cα lDDT among a target's models, keyed by reference residue."""
    if len(model_paths) < 2:
        return None
    ref = load_structure(ref_path)
    models = [load_structure(p) for p in model_paths]

    # All samples decode the same sequence with the same numbering, so one
    # reference alignment (against model 0) indexes every model identically.
    pairs, _ = _aligned_residue_pairs(ref, models[0])
    if not pairs:
        return None

    cas = []
    for m in models:
        ca = m[m.atom_name == "CA"]
        if len(cas) and ca.array_length() != cas[0].array_length():
            return None  # ragged sample set; skip rather than mis-pair
        cas.append(ca)

    # Pairwise residue-wise lDDT over all C(n,2) model pairs.
    per_pair = []
    for i, j in combinations(range(len(cas)), 2):
        per_pair.append(np.asarray(struc.lddt(cas[i], cas[j], aggregation="residue"),
                                   dtype=float))
    agreement_by_ordinal = np.nanmean(np.vstack(per_pair), axis=0)

    ref_starts = struc.get_residue_starts(ref)
    rows = []
    for (r_starts, r, s_starts, s) in pairs:
        if s >= len(agreement_by_ordinal):
            continue
        rows.append({"chain": ref.chain_id[ref_starts[r]],
                     "res_id": int(ref.res_id[ref_starts[r]]),
                     "ensemble_agreement": float(agreement_by_ordinal[s])})
    return pd.DataFrame(rows) if rows else None


def build(pred_dir: Path, ref_dir: Path, model_glob: str) -> pd.DataFrame:
    frames = []
    targets = sorted(p for p in pred_dir.iterdir() if p.is_dir())
    for tdir in targets:
        models = sorted((tdir / "output").glob(model_glob))
        if len(models) < 2:
            continue
        entry = tdir.name.split("_")[0]
        ref = next((ref_dir / f"{entry}{ext}" for ext in (".cif.gz", ".bcif.gz")
                    if (ref_dir / f"{entry}{ext}").exists()), None)
        if ref is None:
            print(f"Warning: no reference for {tdir.name}", file=sys.stderr)
            continue
        try:
            df = target_agreement(ref, models)
        except Exception as exc:
            print(f"Warning: {tdir.name} failed ({type(exc).__name__}: {exc})",
                  file=sys.stderr)
            continue
        if df is None or df.empty:
            continue
        df.insert(0, "candidate", tdir.name)
        df["n_models"] = len(models)
        frames.append(df)
        if len(frames) % 50 == 0:
            print(f"  {len(frames)} targets done", file=sys.stderr)
    if not frames:
        raise SystemExit("No targets produced ensemble agreement.")
    return pd.concat(frames, ignore_index=True)


def compare(agreement_path: Path, scores_path: Path, bad_quantile: float = 0.10) -> None:
    agree = pd.read_csv(agreement_path)
    scores = pd.read_csv(scores_path)
    df = scores.merge(agree, on=["candidate", "chain", "res_id"], how="inner")
    if df.empty:
        raise SystemExit("Join produced no rows — check candidate/chain/res_id keys.")

    print(f"joined residues: {len(df):,} across {df.candidate.nunique()} targets\n")

    thr = df["lddt"].quantile(bad_quantile)
    bad = df["lddt"] <= thr
    print(f"'bad' residue = lDDT <= {thr:.4f} (worst {bad_quantile:.0%}), "
          f"n={int(bad.sum()):,}\n")

    rows = [
        {"signal": "pLDDT (model's own)", "spearman_vs_lddt": spearman(df.plddt, df.lddt),
         "auroc_flag_bad": auc(df.plddt, bad)},
        {"signal": "ensemble agreement", "spearman_vs_lddt": spearman(df.ensemble_agreement, df.lddt),
         "auroc_flag_bad": auc(df.ensemble_agreement, bad)},
    ]
    # Rank-average of the two, a cheap combination that needs no fitting.
    combo = df["plddt"].rank(pct=True) + df["ensemble_agreement"].rank(pct=True)
    rows.append({"signal": "combined (rank avg)", "spearman_vs_lddt": spearman(combo, df.lddt),
                 "auroc_flag_bad": auc(combo, bad)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\ncorrelation between the two signals: "
          f"{spearman(df.plddt, df.ensemble_agreement):.4f}")
    print("(a low value means agreement carries information pLDDT does not)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", default="predictions")
    parser.add_argument("--ref-dir", default="data/raw/cif")
    parser.add_argument("--model-glob", default="pred.model_idx_*.cif")
    parser.add_argument("--out", default="data/analysis/ensemble_agreement.csv")
    parser.add_argument("--compare", action="store_true",
                        help="Skip building; compare an existing agreement table to pLDDT")
    parser.add_argument("--agreement", default="data/analysis/ensemble_agreement.csv")
    parser.add_argument("--scores", default="data/analysis/all_residues.csv")
    parser.add_argument("--bad-quantile", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.compare:
        compare(root / args.agreement, root / args.scores, args.bad_quantile)
        return 0

    df = build(root / args.pred_dir, root / args.ref_dir, args.model_glob)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} residues from {df.candidate.nunique()} targets to {out}")
    print(f"mean agreement={df.ensemble_agreement.mean():.4f}  "
          f"min={df.ensemble_agreement.min():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
