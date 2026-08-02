"""Target-clustered calibration statistics with bootstrap confidence intervals.

Every figure in the per-residue analysis treats 115,275 residues as independent
observations. They are not: residues within one protein share a structure, a
sequence, an MSA, and a single prediction run, so the effective sample size is
closer to the number of *targets* (511) than the number of residues. Pooled
point estimates are still unbiased, but any interval or test built on the
residue count is far too narrow.

This module re-does the analysis with the target as the unit of resampling. The
bootstrap draws targets with replacement, re-pools their residues, and recomputes
the statistic — so the resulting intervals carry the uncertainty that actually
exists across proteins.

Exact and fast: ECE is a weighted sum over confidence bins, so each target is
reduced once to a ``(n_bins, 3)`` array of per-bin ``[count, sum_conf, sum_acc]``.
A bootstrap replicate is then a sum of those arrays rather than a re-binning of
raw residues, which makes thousands of replicates cheap and keeps the result
identical to recomputing from scratch.

Examples
--------
    python scripts/cluster_analysis.py --scores data/analysis/all_residues.csv

    python scripts/cluster_analysis.py --scores data/analysis/all_residues.csv \
        --contrast flexibility:flexible-rigid --contrast ligand_state:apo-holo
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _confidence(plddt: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(plddt, dtype=float) / 100.0, 0.0, 1.0)


def bin_stats(plddt, lddt, n_bins: int = 20) -> np.ndarray:
    """Reduce residues to a ``(n_bins, 3)`` array of [count, sum_conf, sum_acc].

    This is the sufficient statistic for ECE and overconfidence: both are
    recoverable from bin-wise sums, so replicates can be built by adding these
    arrays instead of re-binning residues.
    """
    conf = _confidence(plddt)
    acc = np.asarray(lddt, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges, right=False), 1, n_bins) - 1

    out = np.zeros((n_bins, 3), dtype=float)
    np.add.at(out[:, 0], idx, 1.0)
    np.add.at(out[:, 1], idx, conf)
    np.add.at(out[:, 2], idx, acc)
    return out


def metrics_from_bins(agg: np.ndarray) -> Dict[str, float]:
    """Compute ECE, MCE and overconfidence from aggregated bin sums."""
    count = agg[:, 0]
    total = count.sum()
    if total == 0:
        return {"n": 0, "ece": np.nan, "mce": np.nan, "overconfidence": np.nan,
                "mean_plddt": np.nan, "mean_lddt": np.nan}
    nz = count > 0
    mean_conf = np.divide(agg[:, 1], count, out=np.zeros_like(count), where=nz)
    mean_acc = np.divide(agg[:, 2], count, out=np.zeros_like(count), where=nz)
    gap = np.abs(mean_conf - mean_acc)[nz]
    weights = count[nz] / total
    return {
        "n": int(total),
        "ece": float((weights * gap).sum()),
        "mce": float(gap.max()),
        "overconfidence": float((agg[:, 1].sum() - agg[:, 2].sum()) / total),
        "mean_plddt": float(agg[:, 1].sum() / total * 100.0),
        "mean_lddt": float(agg[:, 2].sum() / total),
    }


def per_target_bins(df: pd.DataFrame, n_bins: int = 20,
                    subset: Optional[pd.Series] = None) -> Tuple[List[str], np.ndarray]:
    """Return (target_ids, stacked bin arrays) — one ``(n_bins, 3)`` slab per target.

    ``subset`` optionally restricts to a residue-level stratum while keeping one
    slab per target, so a target contributing no residues to the stratum simply
    contributes zeros rather than dropping out of the resample.
    """
    work = df if subset is None else df[subset]
    ids = sorted(df["candidate"].unique())
    index = {c: i for i, c in enumerate(ids)}
    slabs = np.zeros((len(ids), n_bins, 3), dtype=float)
    for cand, g in work.groupby("candidate"):
        slabs[index[cand]] = bin_stats(g["plddt"].to_numpy(), g["lddt"].to_numpy(), n_bins)
    return ids, slabs


def bootstrap_metric(slabs: np.ndarray, metric: str = "overconfidence",
                     n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap over targets (the first axis of ``slabs``)."""
    rng = np.random.default_rng(seed)
    n = slabs.shape[0]
    point = metrics_from_bins(slabs.sum(axis=0))[metric]
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        reps[b] = metrics_from_bins(slabs[pick].sum(axis=0))[metric]
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "se": float(np.nanstd(reps)), "n_targets": n}


def bootstrap_contrast(df: pd.DataFrame, column: str, level_a: str, level_b: str,
                       metric: str = "overconfidence", n_bins: int = 20,
                       n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap the ``level_a - level_b`` difference, resampling whole targets.

    Both strata are resampled with the *same* target draw, which preserves the
    pairing when a target contributes residues to both levels (as it does for
    residue-level covariates such as flexibility) and is what makes the interval
    a valid test of the difference rather than of two independent means.
    """
    _, slabs_a = per_target_bins(df, n_bins, subset=df[column] == level_a)
    _, slabs_b = per_target_bins(df, n_bins, subset=df[column] == level_b)

    def diff(sa, sb):
        ma = metrics_from_bins(sa.sum(axis=0))[metric]
        mb = metrics_from_bins(sb.sum(axis=0))[metric]
        return ma - mb

    rng = np.random.default_rng(seed)
    n = slabs_a.shape[0]
    point = diff(slabs_a, slabs_b)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        reps[b] = diff(slabs_a[pick], slabs_b[pick])
    lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"contrast": f"{column}: {level_a} - {level_b}", "point": point,
            "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(reps)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def per_target_table(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """One row per target: its own ECE, overconfidence and means."""
    rows = []
    for cand, g in df.groupby("candidate"):
        m = metrics_from_bins(bin_stats(g["plddt"].to_numpy(), g["lddt"].to_numpy(), n_bins))
        rows.append({"candidate": cand, **m})
    return pd.DataFrame(rows).sort_values("overconfidence", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default="data/analysis/all_residues.csv")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="overconfidence",
                        help="Metric to bootstrap (overconfidence | ece | mce)")
    parser.add_argument("--contrast", action="append", default=[],
                        help="Repeatable, as COLUMN:LEVEL_A-LEVEL_B "
                             "(e.g. flexibility:flexible-rigid)")
    parser.add_argument("--per-target-out", default=None,
                        help="Optional CSV path for the per-target table")
    args = parser.parse_args()

    path = Path(args.scores)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in ("plddt", "lddt", "candidate"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {path}")

    ids, slabs = per_target_bins(df, args.bins)
    print(f"targets={len(ids)}  residues={len(df):,}  bootstrap={args.boot}\n")

    print("=== Pooled, with target-clustered 95% CI ===")
    for metric in ("overconfidence", "ece"):
        r = bootstrap_metric(slabs, metric=metric, n_boot=args.boot, seed=args.seed)
        print(f"  {metric:15s} {r['point']:+.5f}  95% CI [{r['lo']:+.5f}, {r['hi']:+.5f}]"
              f"  (SE {r['se']:.5f})")

    per_target = per_target_table(df, args.bins)
    oc = per_target["overconfidence"]
    print(f"\n=== Spread across targets (n={len(per_target)}) ===")
    print(f"  overconfidence  median {oc.median():+.5f}   IQR [{oc.quantile(.25):+.5f}, "
          f"{oc.quantile(.75):+.5f}]   range [{oc.min():+.5f}, {oc.max():+.5f}]")
    print(f"  targets overconfident: {(oc > 0).sum()}/{len(oc)} "
          f"({100 * (oc > 0).mean():.1f}%)")

    if args.contrast:
        print("\n=== Contrasts (target-clustered bootstrap) ===")
        for spec in args.contrast:
            try:
                column, levels = spec.split(":", 1)
                a, b = levels.split("-", 1)
            except ValueError:
                print(f"  skipping malformed --contrast '{spec}' "
                      f"(expected COLUMN:LEVEL_A-LEVEL_B)")
                continue
            if column not in df.columns:
                print(f"  skipping '{spec}': no column '{column}'")
                continue
            r = bootstrap_contrast(df, column, a, b, metric=args.metric,
                                   n_bins=args.bins, n_boot=args.boot, seed=args.seed)
            flag = "significant" if r["excludes_zero"] else "n.s."
            print(f"  {r['contrast']:38s} {r['point']:+.5f}  "
                  f"95% CI [{r['lo']:+.5f}, {r['hi']:+.5f}]  {flag}")

    if args.per_target_out:
        out = Path(args.per_target_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        per_target.to_csv(out, index=False)
        print(f"\nWrote per-target table to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
