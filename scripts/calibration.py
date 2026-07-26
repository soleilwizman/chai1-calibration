"""Calibration analysis: does Chai-1's predicted pLDDT match realized lDDT?

Stage 5 of the pipeline -- the scientific payload. Given a tidy per-residue
table with ``plddt`` (0..100, predicted confidence) and ``lddt`` (0..1, realized
accuracy) columns -- produced by ``compute_lddt.py`` and concatenated across
targets -- this module answers "is the model calibrated, and where does it break?"

A model is *calibrated* when, among residues it labels with confidence c, the
realized accuracy is also c. We treat ``plddt/100`` as the confidence and compare
it to ``lddt`` in reliability bins.

Metrics
-------
* **ECE** (Expected Calibration Error) -- count-weighted mean gap between
  confidence and accuracy across bins. 0 = perfect.
* **MCE** (Maximum Calibration Error) -- worst single-bin gap.
* **Overconfidence** -- signed mean(confidence - accuracy). Positive means the
  model claims more than it delivers (the failure mode in the hypotheses).

Both a callable API and a CLI are provided. The CLI can stratify every metric by
a covariate column (e.g. ``nonpolymer_entity_count``, an MSA-depth bin, or a
disorder flag) to test the project's hypotheses directly.

CLI example
-----------
    python scripts/calibration.py \
        --scores data/analysis/all_residues.csv \
        --plot data/analysis/reliability.png \
        --by ligand_bin
"""
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalibrationSummary:
    n: int
    ece: float
    mce: float
    overconfidence: float   # signed: conf - acc, positive = overconfident
    mean_plddt: float       # 0..100
    mean_lddt: float        # 0..1


def _as_confidence(plddt: np.ndarray) -> np.ndarray:
    """Map pLDDT (0..100) to a confidence in [0, 1]."""
    plddt = np.asarray(plddt, dtype=float)
    return np.clip(plddt / 100.0, 0.0, 1.0)


def reliability_curve(plddt, lddt, n_bins: int = 20) -> pd.DataFrame:
    """Bin residues by confidence and report accuracy vs. confidence per bin.

    Fixed-width bins over [0, 1]. Empty bins are dropped. Columns:
    ``bin_lo, bin_hi, count, mean_conf, mean_acc, gap`` where ``gap`` is
    ``mean_conf - mean_acc`` (positive = overconfident in that bin).
    """
    conf = _as_confidence(plddt)
    acc = np.asarray(lddt, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index 1..n_bins; clip the exact-1.0 edge into the last bin.
    idx = np.clip(np.digitize(conf, edges, right=False), 1, n_bins) - 1

    rows = []
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            continue
        mc = float(conf[sel].mean())
        ma = float(acc[sel].mean())
        rows.append({
            "bin_lo": edges[b], "bin_hi": edges[b + 1], "count": count,
            "mean_conf": mc, "mean_acc": ma, "gap": mc - ma,
        })
    return pd.DataFrame(rows)


def summarize(plddt, lddt, n_bins: int = 20) -> CalibrationSummary:
    """Compute ECE, MCE, overconfidence and means for a set of residues."""
    conf = _as_confidence(plddt)
    acc = np.asarray(lddt, dtype=float)
    n = conf.size
    curve = reliability_curve(plddt, lddt, n_bins=n_bins)
    if curve.empty:
        return CalibrationSummary(0, float("nan"), float("nan"),
                                  float("nan"), float("nan"), float("nan"))
    weights = curve["count"].to_numpy() / n
    abs_gap = curve["gap"].abs().to_numpy()
    return CalibrationSummary(
        n=n,
        ece=float((weights * abs_gap).sum()),
        mce=float(abs_gap.max()),
        overconfidence=float(conf.mean() - acc.mean()),
        mean_plddt=float(np.asarray(plddt, dtype=float).mean()),
        mean_lddt=float(acc.mean()),
    )


def stratified_summary(df: pd.DataFrame, by: str, n_bins: int = 20,
                       min_count: int = 50) -> pd.DataFrame:
    """Compute the calibration summary within each level of a covariate column."""
    if by not in df.columns:
        raise KeyError(f"Stratification column '{by}' not in table columns {list(df.columns)}")
    rows = []
    for level, g in df.groupby(by, dropna=False):
        if len(g) < min_count:
            continue
        s = summarize(g["plddt"].to_numpy(), g["lddt"].to_numpy(), n_bins=n_bins)
        rows.append({by: level, **asdict(s)})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def plot_reliability(plddt, lddt, out_path: Path, n_bins: int = 20,
                     title: str = "Chai-1 pLDDT calibration") -> None:
    """Save a reliability diagram (accuracy vs. confidence + count histogram)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = reliability_curve(plddt, lddt, n_bins=n_bins)
    s = summarize(plddt, lddt, n_bins=n_bins)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax.plot(curve["mean_conf"], curve["mean_acc"], "o-", color="#1f77b4",
            label="observed")
    for _, r in curve.iterrows():
        ax.vlines(r["mean_conf"], r["mean_acc"], r["mean_conf"],
                  color="#d62728", alpha=0.3, lw=1)
    ax.set_ylabel("realized accuracy (mean lDDT)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE={s.ece:.3f}  MCE={s.mce:.3f}  "
                 f"overconf={s.overconfidence:+.3f}  n={s.n:,}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    axh.bar(curve["mean_conf"], curve["count"], width=1.0 / n_bins * 0.9,
            color="#7f7f7f")
    axh.set_ylabel("residues")
    axh.set_xlabel("predicted confidence (pLDDT / 100)")
    axh.set_xlim(0, 1)
    axh.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True,
                        help="CSV/Parquet with 'plddt' and 'lddt' columns")
    parser.add_argument("--plot", default=None, help="Path to save reliability diagram")
    parser.add_argument("--by", default=None, help="Covariate column to stratify by")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.scores)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in ("plddt", "lddt"):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {path}")

    overall = summarize(df["plddt"].to_numpy(), df["lddt"].to_numpy(), n_bins=args.bins)
    print("=== Overall calibration ===")
    for k, v in asdict(overall).items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

    if args.by:
        print(f"\n=== Stratified by '{args.by}' ===")
        strat = stratified_summary(df, args.by, n_bins=args.bins, min_count=args.min_count)
        with pd.option_context("display.width", 120, "display.max_columns", None):
            print(strat.to_string(index=False))

    if args.plot:
        plot_reliability(df["plddt"].to_numpy(), df["lddt"].to_numpy(),
                         Path(args.plot), n_bins=args.bins)
        print(f"\nWrote reliability diagram to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
