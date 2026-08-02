# Results

Calibration of Chai-1's per-residue confidence (pLDDT) against realized accuracy
(Cα lDDT) on 511 experimental protein structures released 2024–2026, after the
model's training cutoff.

> **Status:** preliminary. Stratified tables are per-residue and descriptive;
> all confidence intervals and significance claims come from a bootstrap that
> resamples whole targets. See [Limitations](#limitations).

## Dataset

| | |
|---|---|
| Targets predicted | 511 of 512 (99.8%) |
| Residues scored | 115,275 |
| Mean residues/target | 226 |
| Source | RCSB, released 2024-01-01 to 2026-12-31 |

Selection: X-ray only, resolution ≤ 2.0 Å, R-free ≤ 0.25, ≤ 10 unmodeled
residues, single model, single protein entity, 100–400 residues, deduplicated at
30% sequence identity (559 clusters), minus 47 targets containing non-standard
monomers (512). One target (`10DT_1`) failed prediction for an unrelated
tooling reason.

Predictions are Chai-1 rank-0 models (3 trunk recycles, 200 diffusion steps,
MSA server). Accuracy is Cα lDDT against the experimental structure, with
residues paired by global sequence alignment rather than residue numbering.

## 1. Chai-1 is well calibrated overall

| metric | value |
|---|---|
| ECE | 0.0057 |
| MCE | 0.1949 |
| Overconfidence (conf − acc) | **+0.57 pp** |
| Mean pLDDT | 94.70 |
| Mean lDDT | 0.9413 |

Averaged over all residues, Chai-1 claims 94.70% confidence and delivers 94.13%
accuracy — an over-claim of 0.57 percentage points (95% CI [+0.31, +0.84] pp,
bootstrapped over targets). On unseen structures, its confidence head is close to
honest.

### The mean is not the typical protein

Resolved per target, that pooled average turns out to describe almost no
individual protein:

| statistic over 511 targets | overconfidence |
|---|---|
| median | **−0.07 pp** |
| IQR | [−1.13, +2.09] pp |
| range | [−7.18, +17.27] pp |
| targets overconfident | 249 / 511 (**48.7%**) |

The median target is very slightly *under*confident, and barely half of targets
are overconfident at all — yet the pooled mean is +0.57 pp. The aggregate is
produced by a minority of badly-calibrated targets with a long right tail, not by
a population-wide bias.

The operative framing is therefore not "Chai-1 is mildly overconfident" but
**"Chai-1 is well calibrated on the typical protein, with a subpopulation where
it fails badly."** For anyone using pLDDT as a filter, the question that matters
is which targets fall in that tail — not a small global correction.

The sections below characterize what distinguishes them.

## 2. Miscalibration concentrates in mobile regions

Three proxies for local mobility, each derived from the experimental structure
and each measuring something different — an experimental refinement parameter,
a geometric assignment, and a surface property:

| proxy | low-mobility | high-mobility | ratio |
|---|---|---|---|
| B-factor z-score | rigid **+0.18 pp** | flexible **+2.05 pp** | 11.6× |
| Secondary structure | helix/sheet +0.20 | coil +0.93 | 4.7× |
| Solvent accessibility | buried +0.31 | exposed +1.23 | 4.0× |

All three are monotonic and agree in direction. The flexible − rigid contrast is
**+1.87 pp, 95% CI [+1.37, +2.37] pp** (bootstrapped over targets) — the largest
and most precisely estimated effect in this analysis.

The mechanism is visible in the component columns. Between rigid and flexible
residues, realized accuracy falls by **7.98 pp** (0.962 → 0.882) while confidence
falls by only **6.11 pp** (96.4 → 90.3). Chai-1 recognizes that mobile regions
are harder — it does lower pLDDT — but it does not lower it enough. **Confidence
degrades more slowly than accuracy.**

## 3. Novel targets are worse calibrated

Stratified by maximum sequence identity to any PDB entry released before
2024-01-01:

| novelty | mean pLDDT | mean lDDT | overconfidence |
|---|---|---|---|
| high (familiar) | 95.34 | 0.956 | **−0.30 pp** |
| mid | 95.55 | 0.948 | +0.76 |
| low (novel) | 94.57 | 0.937 | +0.91 |

Monotonic across the range, and significant under target-clustered resampling:
low − high novelty = **+1.22 pp, 95% CI [+0.65, +1.79] pp**. Accuracy declines by
1.9 pp from familiar to novel targets while confidence declines by only 0.77 pp —
the same under-adjustment seen for flexibility.

On familiar targets Chai-1 is slightly **under**confident. Overconfidence
emerges only as targets become novel.

## 4. Ligand-free structures: suggestive, not established

Marginally, apo structures are both less accurate (0.924 vs 0.944) and worse
calibrated (+1.02 vs +0.50 pp) than holo. Because ligand binding rigidifies
proteins, this could be flexibility observed through a correlated variable.
Stratifying by flexibility shows the gap does not vanish:

| flexibility | apo | holo | gap |
|---|---|---|---|
| flexible | +3.54 pp | +1.82 pp | **+1.72** |
| intermediate | +0.72 | +0.40 | +0.32 |
| rigid | +0.46 | +0.14 | +0.32 |
| *marginal* | | | *+0.52* |

**However, the effect does not reach significance once targets are treated as
the unit of analysis:** apo − holo = +0.52 pp, 95% CI [−0.18, +1.25] pp. Apo/holo
is a target-level property, so all of a target's residues carry the same label
and the residue count contributes no independent information — precisely the
situation where per-residue stratification is most misleading. With 15,069 apo
residues drawn from a modest number of apo targets, the design is
underpowered for this comparison.

The direction is consistent with a plausible mechanism — Chai-1 predicts from
sequence alone and cannot know whether a ligand is present, so where a bound
ligand would rigidify a mobile region the apo structure is more variable than the
model assumes — but this dataset does not establish the effect. Testing it
properly would need a target set enriched for apo structures (the baseline query
yields only 122 apo clusters; see README).

## 5. MSA depth shows no consistent effect

Marginally, low-depth MSAs appear worst calibrated (+0.72 pp vs +0.40 pp at mid
depth), but the relationship is non-monotonic — high depth (+0.58 pp) is worse
than mid. Stratifying by flexibility resolves this as noise rather than signal:

| flexibility | low − high depth |
|---|---|
| flexible | **−0.22 pp** |
| intermediate | +0.13 |
| rigid | +0.31 |

The sign is inconsistent across strata and every magnitude is below 0.35 pp. The
target-clustered contrast agrees: low − high depth = **+0.15 pp, 95% CI
[−0.49, +0.78] pp**.

**We do not find support for the hypothesis that calibration degrades with MSA
depth.** One plausible explanation is restriction of range: the target set is
drawn from well-characterized, crystallizable proteins, and even the "low" depth
tertile contains hundreds to thousands of sequences. A set including genuinely
shallow alignments (tens of sequences) might show the effect.

## 6. Flexibility acts as a risk multiplier

Reading down any stratified table, the ordering flexible ≫ intermediate > rigid
holds throughout. Effect sizes within each flexibility bin:

| effect | rigid | flexible | amplification |
|---|---|---|---|
| Novelty (low − high) | +0.88 pp | +2.11 pp | 2.4× |
| Ligand-free (apo − holo)* | +0.32 pp | +1.72 pp | 5.4× |

\* the ligand-free effect is not significant on its own (§4); the amplification
pattern is reported for completeness, and rests on the novelty row.

Local mobility is not one factor among several — it is the axis along which the
others act. Chai-1 is well calibrated, and in places underconfident, on rigid,
familiar, ligand-bound structure. Its miscalibration concentrates in mobile
regions, and compounds there with novelty and ligand absence.

## 7. Ensemble disagreement adds nothing over pLDDT

Chai-1 emits five diffusion samples per target and the pipeline scores only rank
0. The four discarded models are a free second opinion: mean pairwise Cα lDDT
across the ten model pairs gives a per-residue *agreement* score that costs no
additional GPU time. If disagreement flagged errors better than the model's own
confidence, it would be an easy win for anyone filtering on pLDDT.

It does not.

| signal | Spearman vs lDDT | AUROC (flag worst decile) |
|---|---|---|
| **pLDDT** (model's own) | **0.691** | **0.914** |
| ensemble agreement | 0.591 | 0.854 |
| combined (rank average) | 0.687 | 0.899 |

pLDDT wins on both measures, and the naive combination is *worse* than pLDDT
alone — expected when an unweighted average dilutes a stronger signal with a
weaker one correlated at 0.73.

An unweighted average is a weak test of complementarity, so we also tested
incremental value directly: regress lDDT on pLDDT and correlate agreement with
the residual. The result is **−0.036** — indistinguishable from zero. Ensemble
agreement carries no information about accuracy that pLDDT does not already
contain.

**Why.** The five samples share weights, MSA, and trunk representation, differing
only in diffusion noise. Their spread measures *sampling variance* within one
converged prediction, not the model's uncertainty about whether that prediction
is right. pLDDT, by contrast, is produced by a head trained against true
structures, so it can express error modes on which all five samples happily
agree. Agreement is blind exactly where the model is confidently wrong.

**Practical consequence:** scoring one model per target is sufficient. There is
no reason to compute ensemble disagreement as a confidence signal for this model,
and the result is a positive characterization of pLDDT — a trained confidence
head that outperforms the obvious model-free alternative and subsumes it.

## Statistical approach

Point estimates are pooled over all residues. Intervals and significance come
from a percentile bootstrap over **targets** (2,000 replicates,
`scripts/cluster_analysis.py`): each replicate draws 511 targets with
replacement, re-pools their residues, and recomputes the statistic. Contrasts
draw both strata under the same target sample, which preserves the pairing when a
target contributes residues to both levels.

Target clustering widens intervals roughly fourfold relative to a residue-level
bootstrap. That difference is not cosmetic — it is what moves the apo/holo and
MSA-depth results from apparently-established to not-established.

## Limitations

1. **Residues are not independent**, and this is now accounted for. Stratified
   tables are per-residue and should be read descriptively; every interval and
   significance claim comes from a bootstrap that resamples whole *targets*
   (`scripts/cluster_analysis.py`), which is ~4x wider than a residue-level
   interval. Two effects that look convincing per-residue — apo/holo and MSA
   depth — do not survive this correction.

2. **The disorder effect is a lower bound.** Genuinely disordered residues are
   usually absent from crystal structures and therefore have no lDDT to score.
   The mobility proxies measure flexibility among residues that *were* modeled.
   The selection filter (≤ 10 unmodeled residues) further excludes proteins with
   substantial disorder — the most flexible target examined has a maximum
   B-factor of ~44 Å², well below what an intrinsically disordered region would
   show.

3. **Restricted range on MSA depth**, as noted in §5.

4. **MCE ≈ 0.19–0.21 recurs across strata**, suggesting a single sparse
   confidence bin dominates the maximum-error statistic. The reliability diagram
   should be inspected before quoting MCE.

5. **One stratum is unexplained.** 11,017 residues have no novelty value
   (`novelty_bin = NaN`) and are consistently the worst calibrated (+1.55 pp) and
   least accurate (lDDT 0.889) group in every table. These targets have not been
   characterized.

6. **Single model per target**, which §7 shows is sufficient: ensemble
   disagreement across all five samples is redundant with pLDDT. This does not
   rule out other uses of the discarded samples (e.g. as an accuracy estimate
   rather than a confidence signal).

## Reproducing

```bash
./scripts/run_all.sh                       # full pipeline
python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility
python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility,ligand_state
```
