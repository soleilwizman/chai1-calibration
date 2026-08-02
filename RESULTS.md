# Results

Calibration of Chai-1's per-residue confidence (pLDDT) against realized accuracy
(Cα lDDT) on 511 experimental protein structures released 2024–2026, after the
model's training cutoff.

> **Status:** preliminary. All figures below are per-residue and unclustered;
> see [Limitations](#limitations) before drawing inferential conclusions.

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
accuracy — an over-claim of 0.57 percentage points. On unseen structures, its
confidence head is close to honest.

This average conceals systematic structure, which the remainder of this section
characterizes.

## 2. Miscalibration concentrates in mobile regions

Three proxies for local mobility, each derived from the experimental structure
and each measuring something different — an experimental refinement parameter,
a geometric assignment, and a surface property:

| proxy | low-mobility | high-mobility | ratio |
|---|---|---|---|
| B-factor z-score | rigid **+0.18 pp** | flexible **+2.05 pp** | 11.6× |
| Secondary structure | helix/sheet +0.20 | coil +0.93 | 4.7× |
| Solvent accessibility | buried +0.31 | exposed +1.23 | 4.0× |

All three are monotonic and agree in direction.

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

Monotonic across the range. Accuracy declines by 1.9 pp from familiar to novel
targets while confidence declines by only 0.77 pp — the same under-adjustment
seen for flexibility.

On familiar targets Chai-1 is slightly **under**confident. Overconfidence
emerges only as targets become novel.

## 4. Ligand-free structures are harder, beyond their mobility

Marginally, apo structures are both less accurate (0.924 vs 0.944) and worse
calibrated (+1.02 vs +0.50 pp) than holo. Because ligand binding rigidifies
proteins, this could be flexibility observed through a correlated variable. It
is not:

| flexibility | apo | holo | gap |
|---|---|---|---|
| flexible | +3.54 pp | +1.82 pp | **+1.72** |
| intermediate | +0.72 | +0.40 | +0.32 |
| rigid | +0.46 | +0.14 | +0.32 |
| *marginal* | | | *+0.52* |

The gap survives at every level of flexibility. Notably, the gap among flexible
residues (+1.72 pp) is more than three times the marginal estimate (+0.52 pp),
which is diluted by the rigid and holo residues that dominate the pooled
average.

This is mechanistically expected: Chai-1 predicts from sequence alone and has no
way to know whether a ligand is present. Where a bound ligand would rigidify a
mobile region, the ligand-free structure is more variable than the model assumes.

## 5. MSA depth shows no consistent effect

Marginally, low-depth MSAs appear worst calibrated (+0.72 pp vs +0.40 pp at mid
depth), but the relationship is non-monotonic — high depth (+0.58 pp) is worse
than mid. Stratifying by flexibility resolves this as noise rather than signal:

| flexibility | low − high depth |
|---|---|
| flexible | **−0.22 pp** |
| intermediate | +0.13 |
| rigid | +0.31 |

The sign is inconsistent across strata and every magnitude is below 0.35 pp.

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
| Ligand-free (apo − holo) | +0.32 pp | +1.72 pp | 5.4× |

Local mobility is not one factor among several — it is the axis along which the
others act. Chai-1 is well calibrated, and in places underconfident, on rigid,
familiar, ligand-bound structure. Its miscalibration concentrates in mobile
regions, and compounds there with novelty and ligand absence.

## Limitations

1. **Residues are not independent.** All figures are per-residue across 511
   proteins. Within-protein correlation means the effective sample size is far
   below 115,275, and no significance testing is reported here. Any inferential
   claim requires clustering by target.

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

6. **Single model per target.** Only the rank-0 prediction was scored; Chai-1
   emits five. Ensemble disagreement is an unused confidence signal.

## Reproducing

```bash
./scripts/run_all.sh                       # full pipeline
python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility
python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility,ligand_state
```
