"""Does a PLM coding/noncoding classifier fail specifically on young genes?

plm-utils (Borges et al. 2024) and ProtiGeno (Tu et al. 2023) both report average
performance: ESM embeddings into a supervised head, scored as one number over a
whole transcriptome. Neither asks where the errors land. This asks whether they
land on the evolutionarily novel ORFs -- the ones a coding-potential tool exists
to find in the first place, since anything with detectable homology can be
annotated by homology.

The obvious version of this analysis is wrong, in two ways that both inflate the
effect. Guarding against them is most of what this module does.

**Length.** Young genes are short, and ORF length is the dominant classical
coding signal. An unmatched comparison recovers "short ORFs are hard", which is
the *stated motivation* of both papers rather than a critique of them. Every
comparison here is therefore run on length-matched sets (``--match-on
aa_length``), so length cannot be read off a sequence to predict either its class
or its stratum.

**Age inference itself.** Phylostratigraphy dates a gene by failure to detect
homologs, and Moyers & Zhang showed short, fast-evolving genes are systematically
misdated as young for exactly that reason. That bias points the same direction as
the hypothesis: a BLAST-based age assignment and an ESM-embedding classifier can
fail on the same sequences for the same underlying cause -- no detectable
homologs -- which would manufacture the correlation. This is not a confound that
statistics can remove; it is a bound on interpretation, and
:func:`representation_decomposition` is the attempt to distinguish the two
mechanisms rather than assume one.

Endpoints
---------
**Primary: recall by age stratum.** "The tool fails on young genes" is precisely
"recall drops on young coding genes". Recall needs no negatives, so it sidesteps
the awkward question of what a noncoding transcript's evolutionary age even
means. Strata are matched to a common length distribution before comparison.

**Secondary: AUC by age stratum.** Each stratum's coding ORFs are scored against
length-matched negatives drawn from the shared noncoding pool. Threshold-free, so
it does not depend on where the classifier's operating point happens to sit.

**Decomposition: age, or pretraining representation?** The sharper hypothesis is
not that the classifier fails on young genes but that it *succeeds on old ones
for the wrong reason*. ESM-2 was pretrained on UniRef50, where conserved genes
are dense and orphans are absent or singletons by construction. If accuracy
tracks a sequence's representation in the pretraining set, and age adds nothing
once that is conditioned on, then the classifier is doing homology lookup rather
than learning what makes a sequence protein-like -- a mechanism, not a benchmark
number. :func:`representation_decomposition` tests exactly this by comparing the
pooled association of age with correctness against the same association computed
within strata of pretraining representation.

Examples
--------
    # primary endpoint, length-matched across age strata
    python scripts/orf/age_stratified_eval.py --table data/orf/scored.csv \
        --out data/orf/age_stratified.json

    # is the age effect just pretraining representation?
    python scripts/orf/age_stratified_eval.py --table data/orf/scored.csv \
        --control uniref50_identity
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
from orfkit import match_groups_common_support  # noqa: E402


# --------------------------------------------------------------------- metrics

def auc(values: Sequence[float], positive: Sequence[bool]) -> float:
    """P(positive scores higher than negative), via the Mann-Whitney statistic.

    0.5 is no signal. Non-parametric, so one extreme sequence cannot manufacture
    an effect. Matches the convention used in ``scripts/tail_analysis.py``.
    """
    frame = pd.DataFrame({"v": values, "p": positive}).dropna()
    npos = int(frame["p"].sum())
    nneg = int((~frame["p"].astype(bool)).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = frame["v"].rank()
    return float(
        (ranks[frame["p"].astype(bool)].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    )


def permutation_p(values: Sequence[float], positive: Sequence[bool],
                  n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation p-value for an observed AUC.

    Ranks are invariant under relabelling, so they are computed once and each
    replicate only re-sums the ranks of a fresh random positive set.
    """
    frame = pd.DataFrame({"v": values, "p": positive}).dropna()
    npos = int(frame["p"].sum())
    n = len(frame)
    if npos == 0 or npos == n:
        return float("nan")
    ranks = frame["v"].rank().to_numpy()
    observed = auc(frame["v"], frame["p"].astype(bool))
    denom = npos * (n - npos)
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, n)), axis=1)[:, :npos]
    sums = ranks[idx].sum(axis=1)
    null = (sums - npos * (npos + 1) / 2) / denom
    return float((np.abs(null - 0.5) >= abs(observed - 0.5)).mean())


def classification_metrics(label: np.ndarray, score: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, float]:
    """Recall, TNR, precision, accuracy, MCC and AUC for one group.

    MCC is reported because it is the metric plm-utils uses, and because it
    degrades honestly under the class imbalance that appears once a set is split
    by age -- old strata are large, young strata are small.
    """
    label = np.asarray(label).astype(bool)
    predicted = np.asarray(score) >= threshold

    tp = int((predicted & label).sum())
    tn = int((~predicted & ~label).sum())
    fp = int((predicted & ~label).sum())
    fn = int((~predicted & label).sum())

    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": int(label.size),
        "n_coding": int(label.sum()),
        "n_noncoding": int((~label).sum()),
        "recall": _safe(tp, tp + fn),
        "tnr": _safe(tn, tn + fp),
        "precision": _safe(tp, tp + fp),
        "accuracy": _safe(tp + tn, label.size),
        "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else float("nan"),
        "auc": auc(score, label) if (label.any() and (~label).any()) else float("nan"),
    }


# ----------------------------------------------------- cross-stratum matching

def match_strata_common_support(
    frame: pd.DataFrame,
    group_col: str,
    match_col: str = "aa_length",
    bin_width: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Equalise ``match_col`` across every stratum by per-bin down-sampling.

    Without this, a difference in recall between age strata is uninterpretable:
    the young stratum is shorter, and short ORFs are harder for every method ever
    published. After it, the strata differ in age and not in length, so a
    surviving difference is about age.

    The obvious implementation -- pick one stratum and match the others to its
    length distribution -- cannot work here, and failing quietly is its worst
    property. Young genes are short *by construction*, so matching old genes to
    the young length profile demands old genes shorter than old genes get; the
    draw comes up short at the low end and the residual length imbalance is
    exactly the confound the matching was supposed to remove.

    Instead, each length bin is down-sampled to the **minimum count across
    strata**. Every stratum then carries an identical length histogram by
    construction, bins where any stratum is empty drop out on their own (a
    common-support restriction, rather than an extrapolation), and there is no
    shortfall to report because none is possible.

    The cost is sample size, so the report carries ``retained_fraction`` and the
    surviving length range: matching that keeps 4% of the data has bought
    comparability with a set that may no longer represent the transcriptome, and
    that trade should be visible rather than inferred.
    """
    matched, report = match_groups_common_support(
        frame, group_col, match_col, bin_width=bin_width, seed=seed)
    # The shared primitive names the key ``per_group_n``; the age-stratified
    # reports read better as strata, and downstream code keys off this name.
    if "per_group_n" in report:
        report["per_stratum_n"] = report.pop("per_group_n")
    return matched, report


def stratified_metrics(
    frame: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    score_col: str = "score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Classification metrics per stratum, ordered by stratum label."""
    rows = []
    for group, block in frame.groupby(group_col, dropna=True):
        row = {group_col: group}
        row.update(classification_metrics(
            block[label_col].to_numpy(), block[score_col].to_numpy(), threshold))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


# ------------------------------------------------------------ decomposition

def representation_decomposition(
    frame: pd.DataFrame,
    age_col: str,
    control_col: str,
    label_col: str = "label",
    score_col: str = "score",
    threshold: float = 0.5,
    n_bins: int = 4,
    n_perm: int = 5000,
    seed: int = 0,
) -> Dict[str, object]:
    """Does age still predict classifier error once representation is conditioned?

    Computes the association between ``age_col`` and per-sequence correctness two
    ways:

    * **pooled** -- AUC of age for predicting an error, over everything.
    * **conditional** -- the same AUC computed within bins of ``control_col``
      (typically identity to the pretraining set), then pooled as a
      size-weighted mean. This is a Mantel-Haenszel-style stratified estimate.

    Reading the result:

    * pooled ≫ 0.5 and conditional ≈ 0.5 -- age carried no information of its
      own. The classifier's failures track how well a sequence is represented in
      pretraining, and "young" was a proxy for "absent from UniRef50". This is
      the homology-lookup result, and it is the interesting one.
    * both ≫ 0.5 -- age predicts failure beyond representation. Something about
      novel sequences defeats the model even when pretraining coverage is held
      fixed.
    * pooled ≈ 0.5 -- no age effect to explain; stop here rather than hunting
      subgroups.

    Restricted to coding sequences: "error" means a real gene the classifier
    missed, and a noncoding transcript has no gene age to condition on.
    """
    coding = frame[frame[label_col].astype(bool)].dropna(subset=[age_col, control_col])
    if coding.empty:
        return {"error": "no coding sequences with both age and control present"}

    missed = (coding[score_col].to_numpy() < threshold)
    age = coding[age_col].to_numpy()

    pooled_auc = auc(age, missed)
    pooled_p = permutation_p(age, missed, n_perm=n_perm, seed=seed)

    # Quantile bins keep each control stratum populated; duplicate edges collapse.
    try:
        bins = pd.qcut(coding[control_col], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.Series(["all"] * len(coding), index=coding.index)

    per_bin: List[Dict[str, object]] = []
    weights: List[int] = []
    aucs: List[float] = []
    for level, block in coding.groupby(bins, observed=True):
        block_missed = block[score_col].to_numpy() < threshold
        if block_missed.all() or not block_missed.any():
            per_bin.append({
                "bin": str(level), "n": len(block),
                "auc": None, "note": "no contrast (all correct or all missed)",
            })
            continue
        block_auc = auc(block[age_col].to_numpy(), block_missed)
        per_bin.append({
            "bin": str(level), "n": len(block),
            "error_rate": round(float(block_missed.mean()), 4),
            "auc": round(block_auc, 4),
        })
        aucs.append(block_auc)
        weights.append(len(block))

    conditional = (
        float(np.average(aucs, weights=weights)) if aucs else float("nan")
    )

    return {
        "n_coding": int(len(coding)),
        "error_rate": round(float(missed.mean()), 4),
        "pooled_auc": round(pooled_auc, 4),
        "pooled_permutation_p": round(pooled_p, 4),
        "conditional_auc": round(conditional, 4) if aucs else None,
        "attenuation": (
            round(abs(pooled_auc - 0.5) - abs(conditional - 0.5), 4) if aucs else None
        ),
        "control_bins": per_bin,
        "interpretation": _interpret(pooled_auc, conditional if aucs else float("nan")),
    }


def _interpret(pooled: float, conditional: float) -> str:
    if not np.isfinite(pooled) or abs(pooled - 0.5) < 0.05:
        return ("No pooled association between age and classifier error; there is "
                "no age effect to decompose.")
    if not np.isfinite(conditional):
        return ("Pooled association present, but no control stratum retained a "
                "contrast; the decomposition is uninformative here.")
    if abs(conditional - 0.5) < 0.05:
        return ("Age predicts error pooled, but not within strata of pretraining "
                "representation: consistent with the classifier succeeding by "
                "homology lookup, with age acting as a proxy.")
    if abs(conditional - 0.5) < abs(pooled - 0.5) * 0.6:
        return ("Age effect is substantially attenuated by conditioning on "
                "pretraining representation, but not eliminated.")
    return ("Age predicts error even within strata of pretraining representation; "
            "not explained by pretraining coverage alone.")


# --------------------------------------------------------------------- driver

def run(
    frame: pd.DataFrame,
    age_col: str,
    label_col: str,
    score_col: str,
    match_on: Optional[str],
    control_col: Optional[str],
    threshold: float,
    bin_width: int,
    seed: int,
    n_perm: int,
    age_numeric_col: Optional[str] = None,
) -> Dict[str, object]:
    required = {age_col, label_col, score_col}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"table is missing required columns: {sorted(missing)}")

    result: Dict[str, object] = {
        "n_rows": int(len(frame)),
        "age_column": age_col,
        "threshold": threshold,
        "unmatched": stratified_metrics(
            frame, age_col, label_col, score_col, threshold
        ).to_dict(orient="records"),
    }

    if match_on and match_on in frame.columns:
        coding = frame[frame[label_col].astype(bool)]
        matched, report = match_strata_common_support(
            coding, age_col, match_on, bin_width=bin_width, seed=seed)
        result["match_on"] = match_on
        result["match_report"] = report
        result["matched_recall"] = stratified_metrics(
            matched, age_col, label_col, score_col, threshold
        ).to_dict(orient="records")
        warnings: List[str] = []
        retained = report.get("retained_fraction", 0.0)
        if isinstance(retained, float) and retained < 0.25:
            warnings.append(
                f"length matching retained only {retained:.1%} of coding ORFs; "
                "the matched comparison is comparable but may no longer be "
                "representative of the transcriptome"
            )
        # Binning equalises the histogram, not the within-bin spread. On small
        # matched sets the residual can be large enough to leave the very confound
        # this step exists to remove -- so it has to be said out loud.
        residual = report.get("max_pairwise_ks")
        if isinstance(residual, float) and residual > 0.05:
            warnings.append(
                f"strata still differ in length after matching (max pairwise "
                f"KS={residual:.3f}); with {report.get('n_after')} matched ORFs the "
                "bins are too thin to equalise within-bin spread -- reduce "
                "--bin-width or pool strata before interpreting the comparison"
            )
        if warnings:
            result["match_warning"] = warnings
    elif match_on:
        result["match_warning"] = f"column {match_on!r} not present; NOT length-matched"

    if control_col:
        # The decomposition ranks sequences by age, so it needs a numeric age --
        # ranking the categorical stratum labels would order them alphabetically
        # and quietly return a meaningless AUC.
        numeric_age = age_numeric_col or age_col
        if control_col not in frame.columns:
            result["decomposition"] = {"error": f"column {control_col!r} not present"}
        elif numeric_age not in frame.columns:
            result["decomposition"] = {"error": f"column {numeric_age!r} not present"}
        elif not pd.api.types.is_numeric_dtype(frame[numeric_age]):
            result["decomposition"] = {
                "error": f"column {numeric_age!r} is not numeric; pass --age-numeric "
                         "with a continuous gene age (e.g. origination time in Mya)"
            }
        else:
            result["decomposition"] = representation_decomposition(
                frame, numeric_age, control_col, label_col, score_col,
                threshold=threshold, n_perm=n_perm, seed=seed)
            result["decomposition"]["age_column"] = numeric_age

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", required=True, type=Path,
                        help="CSV of scored ORFs (one row per putative ORF)")
    parser.add_argument("--age-col", default="age_stratum",
                        help="categorical stratum used for the per-stratum tables")
    parser.add_argument("--age-numeric", default=None,
                        help="continuous gene age (e.g. age_mya) used by the "
                             "decomposition; defaults to --age-col")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--match-on", default="aa_length",
                        help="covariate equalised across strata ('' to disable)")
    parser.add_argument("--control", default=None,
                        help="pretraining-representation column for the decomposition")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bin-width", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.table)
    result = run(
        frame, args.age_col, args.label_col, args.score_col,
        args.match_on or None, args.control, args.threshold,
        args.bin_width, args.seed, args.n_perm, args.age_numeric,
    )

    text = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
