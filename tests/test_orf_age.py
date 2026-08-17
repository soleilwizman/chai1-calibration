"""Validate the age-stratification instrument against known ground truth.

The claim this study wants to make -- "the classifier fails specifically on young
genes" -- is exactly the kind of claim a confound produces for free. Young genes
are short; short ORFs are hard for every coding-potential method ever published.
So before trusting the analysis on real data, it has to be shown to give the
right answer in worlds where the truth is known by construction.

Four such worlds are simulated here. In each, classifier correctness is generated
from one specified mechanism, and the test asserts that the analysis recovers
that mechanism rather than a plausible-looking substitute:

* **A -- length only.** Correctness depends on ORF length alone. Young genes are
  shorter, so a naive stratification *must* show recall collapsing on young
  genes. Length matching must make that collapse disappear. If it does not, every
  positive result this pipeline produces is suspect.
* **B -- age beyond length.** Correctness depends on age over and above length.
  The effect must survive length matching.
* **C -- pretraining representation only.** Correctness depends on how well a
  sequence is represented in the pretraining set; age merely correlates with it.
  The decomposition must attribute the effect to representation, not age.
* **D -- age beyond representation.** Correctness depends on age even at fixed
  representation. The decomposition must not explain the effect away.

A and C are the false-positive guards; B and D are the power checks. An
instrument that passes only A and C is one that can never detect anything.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "orf" / "age_stratified_eval.py"
spec = importlib.util.spec_from_file_location("age_stratified_eval", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(mechanism: str, n: int = 24000, seed: int = 0) -> pd.DataFrame:
    """Build a coding-ORF table whose error process is known by construction.

    Gene age drives both ORF length and pretraining representation, mimicking the
    real correlations: young genes are shorter and are absent from UniRef50. What
    differs between mechanisms is only which of those actually causes error.
    """
    rng = np.random.default_rng(seed)

    age_mya = rng.uniform(0, 1000, n)
    # Young genes are shorter, with enough spread that the strata's length
    # distributions genuinely overlap -- otherwise matching cannot be tested.
    aa_length = np.clip(rng.normal(100 + 0.15 * age_mya, 80), 20, 600).astype(int)
    # Old genes are densely represented in UniRef50; young ones are not.
    uniref = np.clip(rng.normal(20 + 0.07 * age_mya, 12), 0, 100)

    length_term = (aa_length - 120) / 60.0
    repr_term = (uniref - 45) / 15.0
    age_term = (age_mya - 500) / 250.0

    if mechanism == "length_only":
        logit = length_term
    elif mechanism == "age_beyond_length":
        logit = length_term + 1.4 * age_term
    elif mechanism == "representation_only":
        logit = repr_term
    elif mechanism == "age_beyond_representation":
        logit = repr_term + 1.4 * age_term
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    correct = rng.random(n) < _sigmoid(logit)
    # A continuous score consistent with correctness, so AUC is meaningful and
    # the 0.5 threshold reproduces `correct` exactly.
    score = np.where(correct, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))

    stratum = pd.cut(
        age_mya, bins=[-1, 100, 500, 1001], labels=["young", "mid", "old"]
    ).astype(str)

    return pd.DataFrame({
        "sequence_id": [f"s{i}" for i in range(n)],
        "label": 1,
        "score": score,
        "aa_length": aa_length,
        "age_mya": age_mya,
        "age_stratum": stratum,
        "uniref50_identity": uniref,
    })


def _recall_by(records, stratum: str) -> float:
    for row in records:
        if row["age_stratum"] == stratum:
            return row["recall"]
    raise AssertionError(f"stratum {stratum!r} missing from {records}")


class LengthConfoundTests(unittest.TestCase):
    """Scenario A: a pure length confound must not survive matching."""

    def setUp(self):
        self.frame = simulate("length_only", seed=1)
        self.result = evaluator.run(
            self.frame, age_col="age_stratum", label_col="label", score_col="score",
            match_on="aa_length", control_col=None, threshold=0.5,
            bin_width=10, seed=1, n_perm=200,
        )

    def test_naive_stratification_shows_a_spurious_age_effect(self):
        gap = (_recall_by(self.result["unmatched"], "old")
               - _recall_by(self.result["unmatched"], "young"))
        self.assertGreater(
            gap, 0.10,
            "simulation should produce an apparent age effect via length alone")

    def test_length_matching_removes_the_spurious_effect(self):
        gap = (_recall_by(self.result["matched_recall"], "old")
               - _recall_by(self.result["matched_recall"], "young"))
        self.assertLess(
            abs(gap), 0.05,
            f"length matching failed to remove a pure length confound (gap={gap:.3f})")

    def test_matching_equalises_lengths_across_strata(self):
        report = self.result["match_report"]
        self.assertTrue(report["complete"], report)
        # Per-bin common-support matching makes identical length histograms a
        # structural guarantee, not a best effort -- so this should be ~0, not
        # merely small.
        self.assertLess(
            report["max_pairwise_ks"], 0.02,
            f"strata still differ in length after matching: {report}")
        self.assertEqual(
            len(set(report["per_stratum_n"].values())), 1,
            f"strata should end up the same size: {report['per_stratum_n']}")

    def test_matching_reports_its_cost(self):
        report = self.result["match_report"]
        self.assertIn("retained_fraction", report)
        self.assertIn("common_support", report)
        self.assertGreater(report["retained_fraction"], 0.0)


class GenuineAgeEffectTests(unittest.TestCase):
    """Scenario B: a real age effect must survive matching."""

    def setUp(self):
        self.frame = simulate("age_beyond_length", seed=2)
        self.result = evaluator.run(
            self.frame, age_col="age_stratum", label_col="label", score_col="score",
            match_on="aa_length", control_col=None, threshold=0.5,
            bin_width=10, seed=2, n_perm=200,
        )

    def test_effect_survives_length_matching(self):
        gap = (_recall_by(self.result["matched_recall"], "old")
               - _recall_by(self.result["matched_recall"], "young"))
        self.assertGreater(
            gap, 0.15,
            f"length matching destroyed a genuine age effect (gap={gap:.3f}) -- "
            "the instrument has no power")


class DecompositionTests(unittest.TestCase):
    """Scenarios C and D: age versus pretraining representation."""

    def test_representation_only_is_attributed_to_representation(self):
        frame = simulate("representation_only", seed=3)
        result = evaluator.run(
            frame, age_col="age_stratum", label_col="label", score_col="score",
            match_on="aa_length", control_col="uniref50_identity", threshold=0.5,
            bin_width=10, seed=3, n_perm=200, age_numeric_col="age_mya",
        )
        decomp = result["decomposition"]

        self.assertGreater(
            abs(decomp["pooled_auc"] - 0.5), 0.10,
            "age should look predictive before conditioning")
        self.assertLess(
            abs(decomp["conditional_auc"] - 0.5), 0.06,
            "conditioning on representation should absorb the age effect, "
            f"got {decomp['conditional_auc']}")
        self.assertGreater(decomp["attenuation"], 0.05)
        self.assertIn("homology lookup", decomp["interpretation"])

    def test_age_beyond_representation_is_not_explained_away(self):
        frame = simulate("age_beyond_representation", seed=4)
        result = evaluator.run(
            frame, age_col="age_stratum", label_col="label", score_col="score",
            match_on="aa_length", control_col="uniref50_identity", threshold=0.5,
            bin_width=10, seed=4, n_perm=200, age_numeric_col="age_mya",
        )
        decomp = result["decomposition"]

        self.assertGreater(
            abs(decomp["conditional_auc"] - 0.5), 0.10,
            "a genuine age effect was wrongly explained away by conditioning "
            f"(conditional_auc={decomp['conditional_auc']})")

    def test_categorical_age_is_refused_rather_than_ranked(self):
        """Ranking stratum *labels* would silently return a meaningless AUC."""
        frame = simulate("representation_only", seed=5)
        result = evaluator.run(
            frame, age_col="age_stratum", label_col="label", score_col="score",
            match_on=None, control_col="uniref50_identity", threshold=0.5,
            bin_width=10, seed=5, n_perm=200,  # no age_numeric_col
        )
        self.assertIn("error", result["decomposition"])
        self.assertIn("not numeric", result["decomposition"]["error"])


class MetricTests(unittest.TestCase):
    def test_metrics_on_a_known_confusion_matrix(self):
        # 3 TP, 1 FN, 2 TN, 1 FP
        label = np.array([1, 1, 1, 1, 0, 0, 0])
        score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.2, 0.6])
        m = evaluator.classification_metrics(label, score)

        self.assertEqual(m["n"], 7)
        self.assertAlmostEqual(m["recall"], 3 / 4)
        self.assertAlmostEqual(m["tnr"], 2 / 3)
        self.assertAlmostEqual(m["precision"], 3 / 4)
        self.assertAlmostEqual(m["accuracy"], 5 / 7)
        self.assertAlmostEqual(m["mcc"], (3 * 2 - 1 * 1) / np.sqrt(4 * 4 * 3 * 3))

    def test_auc_is_half_for_pure_noise(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=4000)
        positive = rng.random(4000) < 0.5
        self.assertLess(abs(evaluator.auc(values, positive) - 0.5), 0.03)

    def test_permutation_p_is_uniformish_under_the_null(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=300)
        positive = rng.random(300) < 0.5
        p = evaluator.permutation_p(values, positive, n_perm=2000, seed=0)
        self.assertGreater(p, 0.05)

    def test_missing_columns_raise(self):
        with self.assertRaises(SystemExit):
            evaluator.run(
                pd.DataFrame({"label": [1], "score": [0.9]}),
                age_col="age_stratum", label_col="label", score_col="score",
                match_on=None, control_col=None, threshold=0.5,
                bin_width=10, seed=0, n_perm=10,
            )


if __name__ == "__main__":
    unittest.main()
