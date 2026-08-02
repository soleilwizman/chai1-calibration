"""What distinguishes the badly-calibrated tail of targets?

The pooled overconfidence (+0.57 pp) is not a property of the typical protein:
the median target is very slightly *under*confident and only ~49% are
overconfident at all. The aggregate comes from a minority with a long right tail
(up to +17 pp). The useful question is therefore not "how overconfident is the
model" but **which targets land in that tail**.

Eyeballing a sorted list is how spurious patterns get found — with a dozen
covariates and twenty targets, something always looks shared. This instead scores
every available covariate the same way and ranks them, so the comparison is
systematic and the multiplicity is visible.

For each covariate it reports:

* **AUC** — probability that a randomly chosen tail target scores higher than a
  randomly chosen non-tail target. 0.5 is no signal; distance from 0.5 in either
  direction is the effect size, and it is non-parametric, so a single extreme
  target cannot manufacture it.
* **permutation p** — how often a random relabelling of tail membership produces
  an AUC at least this extreme. No distributional assumptions.

Per-target covariates come from the metadata files; residue-level properties
(flexibility, exposure, secondary structure) are aggregated to per-target
fractions, since a target's *composition* is what could plausibly explain its
calibration.

Examples
--------
    python scripts/tail_analysis.py --per-target data/analysis/per_target_calibration.csv \
        --scores data/analysis/all_residues.csv --top 20
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def auc(values: np.ndarray, is_tail: np.ndarray) -> float:
    """P(tail target > non-tail target), via the Mann-Whitney rank statistic."""
    s = pd.DataFrame({"v": values, "t": is_tail}).dropna()
    npos, nneg = int(s["t"].sum()), int((~s["t"].astype(bool)).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = s["v"].rank()
    return float((ranks[s["t"].astype(bool)].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def permutation_p(values: np.ndarray, is_tail: np.ndarray,
                  n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the observed AUC.

    Ranks are invariant under relabelling, so they are computed once and each
    replicate only re-sums the ranks of a fresh random tail. That turns the
    inner loop into a sum over ``npos`` values instead of a re-rank of the whole
    column, which is the difference between seconds and minutes across a dozen
    covariates.
    """
    s = pd.DataFrame({"v": values, "t": is_tail}).dropna()
    if len(s) < 5:
        return float("nan")
    ranks = s["v"].rank().to_numpy()
    labels = s["t"].to_numpy().astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")

    denom = npos * nneg
    const = npos * (npos + 1) / 2
    obs = abs((ranks[labels].sum() - const) / denom - 0.5)

    rng = np.random.default_rng(seed)
    n = len(ranks)
    # One (n_perm, npos) draw of tail memberships, vectorised.
    picks = np.argpartition(rng.random((n_perm, n)), npos, axis=1)[:, :npos]
    stats = np.abs((ranks[picks].sum(axis=1) - const) / denom - 0.5)
    return float((int((stats >= obs).sum()) + 1) / (n_perm + 1))


def load_target_covariates(root: Path) -> pd.DataFrame:
    """Per-target metadata from the covariate JSON files (whichever exist)."""
    frames = []
    specs = [
        ("data/targets/candidates_covariates.json",
         ["entity_length", "nonpolymer_entity_count", "rcsb_mutation_count",
          "rcsb_artifact_monomer_count", "max_train_identity"]),
        ("data/targets/msa_depth.json", ["n_sequences", "neff", "neff_per_col"]),
        ("data/targets/training_identity.json", ["max_identity", "n_hits"]),
    ]
    for rel, keep in specs:
        p = root / rel
        if not p.exists():
            continue
        d = json.load(p.open("r", encoding="utf-8"))
        if not isinstance(d, list) or not d:
            continue
        df = pd.DataFrame(d)
        if "candidate" not in df.columns:
            continue
        cols = ["candidate"] + [c for c in keep if c in df.columns]
        frames.append(df[cols].drop_duplicates("candidate"))
    if not frames:
        return pd.DataFrame(columns=["candidate"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="candidate", how="outer")
    return out


def residue_composition(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate residue-level properties to per-target fractions and means."""
    rows = []
    for cand, g in scores.groupby("candidate"):
        r: Dict[str, float] = {"candidate": cand, "n_residues": len(g)}
        if "flexibility" in g:
            r["frac_flexible"] = float((g["flexibility"] == "flexible").mean())
            r["frac_rigid"] = float((g["flexibility"] == "rigid").mean())
        if "structured" in g:
            r["frac_coil"] = float((g["structured"] == "coil").mean())
        if "exposure" in g:
            r["frac_exposed"] = float((g["exposure"] == "exposed").mean())
        # Deliberately excludes plddt/lddt: per_target_calibration.csv already has
        # them, and they are outcomes, not covariates -- overconfidence is defined
        # as mean_conf - mean_acc, so ranking mean_lddt against the tail would
        # rediscover that definition and crowd out real explanations.
        for col in ("ref_bfactor_z", "ref_bfactor", "rsa", "coverage"):
            if col in g:
                r[f"mean_{col}"] = float(g[col].mean())
        if "near_chain_gap" in g:
            r["frac_near_gap"] = float(g["near_chain_gap"].astype(bool).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-target", default="data/analysis/per_target_calibration.csv")
    parser.add_argument("--scores", default="data/analysis/all_residues.csv")
    parser.add_argument("--metric", default="overconfidence")
    parser.add_argument("--top", type=int, default=20,
                        help="Tail size by --metric (0 to use --quantile instead)")
    parser.add_argument("--quantile", type=float, default=0.10,
                        help="Tail fraction when --top is 0")
    parser.add_argument("--perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional CSV for the ranked table")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    per_target = pd.read_csv(root / args.per_target)
    scores = pd.read_csv(root / args.scores)

    feats = residue_composition(scores).merge(
        load_target_covariates(root), on="candidate", how="left")
    df = per_target.merge(feats, on="candidate", how="left")

    df = df.sort_values(args.metric, ascending=False).reset_index(drop=True)
    k = args.top if args.top > 0 else max(1, int(round(len(df) * args.quantile)))
    df["is_tail"] = False
    df.loc[df.index[:k], "is_tail"] = True

    print(f"targets={len(df)}   tail = worst {k} by {args.metric}")
    print(f"tail {args.metric}: {df.loc[df.is_tail, args.metric].min():+.4f} .. "
          f"{df.loc[df.is_tail, args.metric].max():+.4f}")
    print(f"rest {args.metric}: {df.loc[~df.is_tail, args.metric].min():+.4f} .. "
          f"{df.loc[~df.is_tail, args.metric].max():+.4f}\n")

    print("=== worst targets ===")
    show = [c for c in ["candidate", args.metric, "mean_plddt", "mean_plddt_x",
                        "mean_lddt", "mean_lddt_x", "n_residues", "mean_coverage"]
            if c in df.columns]
    print(df.head(min(k, 20))[show].to_string(index=False))

    # Outcomes and any merge-collision duplicates are not candidate explanations.
    skip = {"candidate", "is_tail", "n", "ece", "mce", "overconfidence",
            "mean_plddt", "mean_lddt"}
    numeric = [c for c in df.columns
               if c not in skip and not c.endswith(("_x", "_y"))
               and pd.api.types.is_numeric_dtype(df[c])
               and df[c].notna().sum() >= 20 and df[c].nunique() > 2]

    rows = []
    tail = df["is_tail"].to_numpy()
    for col in numeric:
        v = df[col].to_numpy(dtype=float)
        a = auc(v, tail)
        rows.append({
            "covariate": col,
            "auc_tail_higher": a,
            "effect": abs(a - 0.5) if a == a else np.nan,
            "tail_median": float(np.nanmedian(v[tail])),
            "rest_median": float(np.nanmedian(v[~tail])),
            "p_perm": permutation_p(v, tail, n_perm=args.perm, seed=args.seed),
        })
    table = (pd.DataFrame(rows).sort_values("effect", ascending=False)
             .reset_index(drop=True))

    print(f"\n=== what distinguishes the tail? ({len(numeric)} covariates tested) ===")
    print("AUC 0.5 = no signal; >0.5 = higher in tail; <0.5 = lower in tail\n")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(table.drop(columns=["effect"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    alpha = 0.05 / max(1, len(numeric))
    sig = table[table["p_perm"] < alpha]
    print(f"\nBonferroni threshold for {len(numeric)} tests: p < {alpha:.5f}")
    if sig.empty:
        print("No covariate separates the tail after correcting for multiplicity.")
        print("The tail is not explained by anything measured here.")
    else:
        for _, r in sig.iterrows():
            direction = "higher" if r["auc_tail_higher"] > 0.5 else "lower"
            print(f"  {r['covariate']}: {direction} in tail "
                  f"(AUC {r['auc_tail_higher']:.3f}, p={r['p_perm']:.5f})")

    if args.out:
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nWrote ranked table to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
