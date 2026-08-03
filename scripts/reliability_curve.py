"""The reliability curve itself: where on the confidence axis does pLDDT break?

Every other script here reports scalars summarising a curve -- ECE averages the
confidence-accuracy gap over bins, MCE takes its maximum, ``overconfidence``
collapses it to one signed number. None of them says *where* the model is
honest and where it over-claims, which is the question anyone setting a pLDDT
threshold actually has. A model honest at 95 and over-claiming at 70 calls for
the opposite advice from one that behaves the other way round.

Three things are produced.

**The per-bin curve.** Residues are binned by confidence (equal-width, matching
the ECE definition) and each bin reports mean confidence, realized accuracy, and
the signed gap with a **target-clustered** 95% interval -- resampling proteins,
not residues, for the reasons in ``cluster_analysis``. Bins holding few residues
are flagged rather than trusted.

**An MCE diagnosis.** MCE is the worst single bin, so one sparse bin can own it.
The script reports which bin supplies the maximum and how many residues it holds,
and recomputes MCE over adequately-populated bins so the two can be compared.

**A threshold table.** For each candidate pLDDT cutoff: how many residues
survive, what accuracy they actually deliver, and how far their mean confidence
overshoots it. This is the operational form of the whole study -- it answers
"if I keep everything above 90, what am I really getting?"

Examples
--------
    python scripts/reliability_curve.py --scores data/analysis/all_residues.csv \
        --out data/analysis/reliability_curve.csv \
        --plot data/analysis/reliability_curve.png

Restrict to a stratum to see whether the curve's shape (not just its level)
changes with a covariate::

    python scripts/reliability_curve.py --where flexibility=flexible
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_analysis import per_target_bins  # noqa: E402

DEFAULT_THRESHOLDS = [50, 60, 70, 80, 85, 90, 95]


def _gap_from_agg(agg: np.ndarray) -> np.ndarray:
    """Signed per-bin gap (mean confidence - mean accuracy); NaN where empty."""
    count = agg[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(count > 0, agg[:, 1] / count, np.nan)
        acc = np.where(count > 0, agg[:, 2] / count, np.nan)
    return conf - acc


def curve(df: pd.DataFrame, n_bins: int = 20, n_boot: int = 2000,
          seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Per-bin reliability table with target-clustered intervals on the gap."""
    _, slabs = per_target_bins(df, n_bins)
    agg = slabs.sum(axis=0)
    count = agg[:, 0]
    total = count.sum()

    rng = np.random.default_rng(seed)
    n_targets = slabs.shape[0]
    reps = np.empty((n_boot, n_bins), dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_targets, size=n_targets)
        reps[b] = _gap_from_agg(slabs[pick].sum(axis=0))
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)],
                                  axis=0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(count > 0, agg[:, 1] / count, np.nan)
        mean_acc = np.where(count > 0, agg[:, 2] / count, np.nan)
    return pd.DataFrame({
        "bin_lo": edges[:-1] * 100,
        "bin_hi": edges[1:] * 100,
        "n": count.astype(int),
        "share": count / total if total else np.nan,
        "mean_plddt": mean_conf * 100,
        "mean_lddt": mean_acc,
        "gap_pp": (mean_conf - mean_acc) * 100,
        "gap_lo": lo * 100,
        "gap_hi": hi * 100,
    })


def threshold_table(df: pd.DataFrame, thresholds: List[float], n_bins: int = 20,
                    n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """What a user actually gets by keeping every residue above each cutoff."""
    rng_seed = seed
    rows = []
    for t in thresholds:
        keep = df["plddt"] >= t
        if not keep.any():
            continue
        _, slabs = per_target_bins(df, n_bins, subset=keep)
        agg = slabs.sum(axis=0)
        total = agg[:, 0].sum()
        point = (agg[:, 1].sum() - agg[:, 2].sum()) / total

        rng = np.random.default_rng(rng_seed)
        n_targets = slabs.shape[0]
        reps = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            pick = rng.integers(0, n_targets, size=n_targets)
            a = slabs[pick].sum(axis=0)
            tot = a[:, 0].sum()
            reps[b] = (a[:, 1].sum() - a[:, 2].sum()) / tot if tot else np.nan
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanpercentile(reps, [2.5, 97.5])

        rows.append({
            "threshold": t,
            "residues_kept": int(total),
            "share_kept": total / len(df),
            "mean_plddt": agg[:, 1].sum() / total * 100,
            "mean_lddt": agg[:, 2].sum() / total,
            "gap_pp": point * 100,
            "gap_lo": lo * 100,
            "gap_hi": hi * 100,
        })
    return pd.DataFrame(rows)


def diagnose_mce(tab: pd.DataFrame, min_count: int) -> Dict[str, float]:
    """Which bin owns MCE, and what MCE becomes once sparse bins are excluded."""
    populated = tab[tab["n"] > 0]
    worst = populated.loc[populated["gap_pp"].abs().idxmax()]
    dense = tab[tab["n"] >= min_count]
    dense_worst = (dense.loc[dense["gap_pp"].abs().idxmax()]
                   if not dense.empty else None)
    return {
        "mce_pp": float(abs(worst["gap_pp"])),
        "mce_bin": f"{worst['bin_lo']:.0f}-{worst['bin_hi']:.0f}",
        "mce_bin_n": int(worst["n"]),
        "mce_bin_share": float(worst["share"]),
        "mce_dense_pp": (float(abs(dense_worst["gap_pp"]))
                         if dense_worst is not None else float("nan")),
        "mce_dense_bin": (f"{dense_worst['bin_lo']:.0f}-{dense_worst['bin_hi']:.0f}"
                          if dense_worst is not None else "-"),
    }


def make_plot(tab: pd.DataFrame, path: Path, min_count: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = tab[tab["n"] >= min_count]
    thin = tab[(tab["n"] > 0) & (tab["n"] < min_count)]

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6.4, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.plot([0, 100], [0, 1], ls="--", lw=1, color="0.6", label="perfect calibration")
    ax.errorbar(ok["mean_plddt"], ok["mean_lddt"],
                yerr=[(ok["gap_hi"] - ok["gap_pp"]) / 100,
                      (ok["gap_pp"] - ok["gap_lo"]) / 100],
                fmt="o-", ms=4, lw=1.5, capsize=2, color="#1a4d8f",
                label=f"n >= {min_count}")
    if not thin.empty:
        ax.plot(thin["mean_plddt"], thin["mean_lddt"], "x", ms=6, color="#c04040",
                label=f"n < {min_count} (unreliable)")
    ax.set_ylabel("realized lDDT")
    ax.set_title("Chai-1 reliability curve (target-clustered 95% CI)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    axh.bar(tab["mean_plddt"].fillna((tab.bin_lo + tab.bin_hi) / 2),
            tab["n"], width=4.0, color="0.7")
    axh.set_yscale("log")
    axh.set_xlabel("pLDDT")
    axh.set_ylabel("residues")
    axh.grid(alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default="data/analysis/all_residues.csv")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=200,
                        help="Bins below this are flagged as unreliable")
    parser.add_argument("--where", default=None,
                        help="Restrict to a stratum, as COLUMN=VALUE")
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    parser.add_argument("--out", default=None, help="CSV path for the per-bin table")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    path = Path(args.scores)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in ("plddt", "lddt", "candidate"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {path}")

    if args.where:
        col, _, val = args.where.partition("=")
        if col not in df.columns:
            raise SystemExit(f"No column '{col}'; have {sorted(df.columns)[:20]}")
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            val = val.strip().lower() in ("true", "t", "1", "yes")
        elif pd.api.types.is_numeric_dtype(series):
            val = pd.to_numeric(val)
        df = df[series == val]
        if df.empty:
            raise SystemExit(f"--where {args.where} selected no residues")
        print(f"restricted to {args.where}: {len(df):,} residues, "
              f"{df.candidate.nunique()} targets\n")

    tab = curve(df, n_bins=args.bins, n_boot=args.boot, seed=args.seed)
    print(f"targets={df.candidate.nunique()}  residues={len(df):,}  "
          f"bins={args.bins}  bootstrap={args.boot}\n")

    print("=== Reliability curve (gap = confidence - accuracy, pp) ===")
    show = tab[tab["n"] > 0].copy()
    show["flag"] = np.where(show["n"] < args.min_count, "sparse", "")
    show["ci"] = show.apply(lambda r: f"[{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]", axis=1)
    print(show[["bin_lo", "bin_hi", "n", "share", "mean_plddt", "mean_lddt",
                "gap_pp", "ci", "flag"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    dense = tab[tab["n"] >= args.min_count]
    over = dense[dense["gap_lo"] > 0]
    under = dense[dense["gap_hi"] < 0]
    print("\n=== Where the model breaks (well-populated bins only) ===")
    print(f"  overconfident (CI above 0): "
          f"{', '.join(f'{r.bin_lo:.0f}-{r.bin_hi:.0f}' for r in over.itertuples()) or 'none'}")
    print(f"  underconfident (CI below 0): "
          f"{', '.join(f'{r.bin_lo:.0f}-{r.bin_hi:.0f}' for r in under.itertuples()) or 'none'}")

    d = diagnose_mce(tab, args.min_count)
    print("\n=== MCE diagnosis ===")
    print(f"  MCE {d['mce_pp']:.2f} pp comes from bin {d['mce_bin']}, "
          f"which holds {d['mce_bin_n']:,} residues ({d['mce_bin_share']:.4%})")
    print(f"  MCE over bins with n >= {args.min_count}: "
          f"{d['mce_dense_pp']:.2f} pp (bin {d['mce_dense_bin']})")
    if d["mce_bin_n"] < args.min_count:
        print("  -> MCE is set by a sparse bin and should not be quoted as a "
              "summary statistic.")

    thr = threshold_table(df, [float(t) for t in args.thresholds.split(",")],
                          n_bins=args.bins, seed=args.seed)
    print("\n=== If you filter at each pLDDT threshold ===")
    thr_show = thr.copy()
    thr_show["ci"] = thr_show.apply(
        lambda r: f"[{r.gap_lo:+.2f}, {r.gap_hi:+.2f}]", axis=1)
    print(thr_show[["threshold", "residues_kept", "share_kept", "mean_plddt",
                    "mean_lddt", "gap_pp", "ci"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(out, index=False)
        thr.to_csv(out.with_name(out.stem + "_thresholds.csv"), index=False)
        print(f"\nWrote {out} and {out.with_name(out.stem + '_thresholds.csv')}")
    if args.plot:
        make_plot(tab, Path(args.plot), args.min_count)
        print(f"Wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
