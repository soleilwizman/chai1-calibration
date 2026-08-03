# Does Chai-1 know when it is wrong?

**A calibration study of Chai-1's per-residue confidence on 511 post-cutoff crystal structures.**

This is the full write-up: what the model is, what we measured, how we measured
it, what came out, and what it means. [`RESULTS.md`](RESULTS.md) has the same
findings in condensed form with every number; this document explains them.

---

## 1. The short version

Chai-1 reports a confidence score for every residue it predicts (pLDDT, 0–100).
We asked whether that number is *honest* — when the model says 95, is it right
95% of the time? — on 511 protein structures released after its training data
ends, so it had never seen the answers.

Four findings:

1. **Chai-1's confidence is close to honest overall.** Expected calibration
   error 0.0057; it over-claims by 0.57 percentage points on average
   (95% CI [+0.31, +0.84]). That is a small number and a real achievement.

2. **But that average describes almost no individual protein.** The *median*
   target is very slightly under-confident, and only 48.7% of targets are
   overconfident at all. The pooled over-claim is produced by a minority of
   targets that fail badly (up to +17 pp). The right mental model is not
   "slightly overconfident everywhere" but **"well calibrated, with a bad tail."**

3. **Inside a protein, miscalibration lives in mobile regions** — and peaks at
   the edge of disorder. Flexible residues are over-claimed by +1.87 pp more than
   rigid ones (95% CI [+1.37, +2.37]), confirmed by three independent proxies.
   Residues flanking an actual break in the crystal — where density ran out —
   are over-claimed by **+8.10 pp** (95% CI [+4.34, +12.22]), 15.7× the global
   rate and the largest effect in the study. The mechanism is visible throughout:
   from rigid to flexible residues accuracy drops 7.98 pp while confidence drops
   6.11 pp, and at a chain break accuracy drops 18.4 pp against 10.3 pp of
   confidence. The model knows these regions are harder. It does not know *how
   much* harder.

4. **Across proteins, what predicts failure is novelty, not flexibility.** A
   target's overall flexibility says nothing about whether that target is badly
   calibrated (AUC 0.469, p = 0.64). Low sequence identity to anything in the
   pre-cutoff PDB does (+1.22 pp, 95% CI [+0.65, +1.79]). These are different
   questions with different answers, and conflating them is easy.

Two hypotheses we set out to test did **not** survive: MSA depth has no
consistent effect, and the apo/holo gap is suggestive but underpowered. One
free idea we tried also failed: ensemble disagreement across Chai-1's five
diffusion samples carries **no** information about accuracy that pLDDT does not
already have (residual correlation −0.036).

---

## 2. Background

### 2.1 What Chai-1 is

Chai-1 is an open-weights molecular structure prediction model released by Chai
Discovery in October 2024. It is in the AlphaFold3 family: given sequences, it
predicts all-atom 3D coordinates, and it handles proteins together with ligands
and nucleic acids rather than proteins alone.

The parts that matter for this study, all of them directly observed in the runs:

- **It runs from sequence.** You hand it a FASTA; it does the rest.
- **It retrieves an MSA** (multiple sequence alignment — a stack of evolutionarily
  related sequences) from a server at inference time. We ran with
  `use_msa_server=True`, and the alignments it fetched are on disk as
  `.aligned.pqt` parquet files, which is how we measured MSA depth.
- **It uses a protein language model.** The first run downloads 5.7 GB of ESM-2
  weights.
- **It generates coordinates by diffusion.** We ran 3 trunk recycles and 200
  diffusion timesteps. It emits **five samples** per target, ranked; the pipeline
  scores rank 0.
- **It writes pLDDT into the B-factor column** of the output mmCIF, one value per
  atom, 0–100. That column is the entire subject of this study.

### 2.2 What pLDDT is, and why calibration is the question

**lDDT** (local Distance Difference Test) scores a predicted structure against a
true one, per residue, from 0 to 1. Instead of superimposing the two structures
globally, it asks a local question: for each residue, take all its interatomic
distances to nearby atoms, and check what fraction of them are reproduced within
tolerance. It is superposition-free, which makes it robust to a correct domain
being placed at a wrong angle — the domain still scores well internally.

**pLDDT** is the model's *prediction of its own lDDT*, made without seeing the
answer. A separate head, trained against true structures, outputs it.

So the two numbers are directly comparable on the same scale, and the question
writes itself:

> Over all residues where the model said 95, was the realized lDDT 0.95?

That is **calibration**. It is not the same question as accuracy, and it is the
one that governs how the model gets used in practice. Essentially everyone
downstream — deciding whether to trust a loop, whether to dock into a pocket,
whether a prediction is worth an experiment — thresholds on pLDDT. A model that
is 90% accurate and *knows which 10% is wrong* is far more useful than one that
is 93% accurate and cannot tell you.

### 2.3 Why the accuracy benchmarks do not answer this

Published benchmarks report mean accuracy on a held-out set. That tells you the
model is good. It does not tell you whether the confidence score is trustworthy,
whether it fails uniformly or in a concentrated tail, or *which* structural or
evolutionary properties make the confidence unreliable. Those are the questions
this repo was built to answer, and each one requires per-residue paired
(confidence, realized accuracy) data on structures the model has not seen.

---

## 3. What we did

### 3.1 The measurement

For each target: predict the structure from sequence alone, then score every
residue against the experimental crystal structure. That yields one row per
residue holding `plddt` (what the model claimed) and `lddt` (what it delivered),
plus covariates. 115,275 rows across 511 targets.

From those pairs we compute:

| metric | meaning |
|---|---|
| **Overconfidence** | mean(pLDDT/100) − mean(lDDT). Signed. Positive = over-claiming. |
| **ECE** | Expected Calibration Error: bin residues by confidence, average \|confidence − accuracy\| per bin, weight by bin size. |
| **MCE** | The worst single bin's gap. |
| Reliability diagram | Confidence on x, realized accuracy on y. Perfect calibration is the diagonal. |

### 3.2 Choosing targets, and why each filter is there

The whole study depends on the ground truth being trustworthy and the targets
being genuinely unseen. The RCSB query (`data/targets/q.json`) enforces both:

| filter | why |
|---|---|
| Released 2024-01-01 to 2026 | after the training cutoff — the model cannot have memorized these |
| X-ray only, resolution ≤ 2.0 Å | the "true" structure must actually be well determined |
| R-free ≤ 0.25 | a second, independent quality guard; empirically the stricter of the two |
| ≤ 10 unmodeled residues | if a third of the protein is missing from the crystal, we cannot score it |
| Single model, single protein entity | isolates monomer folding from complex assembly, a different problem |
| 100–400 residues | excludes peptides and enormous multi-domain proteins |
| Dedup at 30% sequence identity | otherwise 40 lysozyme variants would dominate the statistics |

That yields **559 clusters** from 2,814 entities. We then removed **47** targets
containing non-standard monomers (selenomethionine, phosphoserine and similar) —
Chai-1 predicts standard amino acids, so scoring it against a chemically modified
residue penalizes it for something it was never asked to produce. **512 targets**
went to prediction; 511 completed (one failed for an unrelated tooling reason).

The README records a sensitivity table showing how the pool responds to each
filter, so nobody has to guess whether a threshold was load-bearing.

### 3.3 The pipeline

| # | Stage | Script |
|---|---|---|
| 1 | Target selection and covariates | `extract_candidate_covariates.py` |
| 2 | Download experimental ground truth | `download_structures.py` |
| 3 | Run Chai-1 | `run_predictions.py` |
| 3b | MSA depth / Neff | `extract_msa_depth.py` |
| 3c | Identity to pre-cutoff PDB | `extract_training_identity.py` |
| 4 | Per-residue lDDT vs pLDDT | `compute_lddt.py` |
| 4b | Merge into one table | `build_dataset.py` |
| 5 | Calibration metrics, stratified | `calibration.py` |
| 6 | Target-clustered bootstrap | `cluster_analysis.py` |
| 7 | Ensemble agreement | `ensemble_agreement.py` |
| 8 | What distinguishes the tail | `tail_analysis.py` |

`run_all.sh` chains all of it and is resumable — every stage skips work already
on disk, which matters when the batch takes 22 hours and one target can fail.

### 3.4 Three decisions that would otherwise have silently corrupted the answer

These are the parts of the build worth knowing about, because each one produces
a plausible-looking number when done wrong.

**Residues are paired by sequence alignment, not by residue number.** The
prediction numbers residues 1..N from the input FASTA. The crystal structure uses
whatever numbering the depositors chose — offsets, gaps, an unmodeled His-tag,
chain IDs that do not match. Joining on residue number would have compared
residue 40 of the prediction to residue 40 of the crystal, which is a different
amino acid whenever there is any offset at all, and the resulting lDDT would look
like a real number. `compute_lddt.py` globally aligns the two sequences and pairs
residues through the alignment, then reports `coverage` (fraction of reference
residues successfully paired) so low-coverage targets can be flagged rather than
trusted.

**lDDT is Cα-only by default.** All-atom lDDT is the more standard structural
metric, but AlphaFold-style pLDDT is trained to predict the *Cα* lDDT. Scoring
all-atom against a Cα-trained confidence measures side-chain packing that pLDDT
was never claiming to predict, and would make the model look overconfident for
the wrong reason. `--metric all-atom` exists; it is not the right comparison here.

**Residues within a protein are not independent.** 115,275 residues come from
511 proteins. Residues in one protein share a fold, a sequence, an MSA and a
single prediction run — the effective sample size is much closer to 511 than to
115,275. Every interval and every significance claim in this study therefore
comes from a **bootstrap that resamples whole targets**
(`cluster_analysis.py`, 2,000 replicates), not residues. This widens intervals
roughly fourfold, and that is not a cosmetic difference: **it is what moved the
apo/holo and MSA-depth results from apparently-established to not-established.**
Both would have been reported as findings under a residue-level interval.

### 3.5 The covariates, and which hypothesis each one tests

| covariate | source | tests |
|---|---|---|
| `ref_bfactor_z` | crystal B-factors, z-scored within each structure | disorder / mobility |
| `sse` (helix/sheet/coil) | geometric assignment from the crystal | disorder / mobility |
| `rsa` (relative solvent accessibility) | computed surface area | disorder / mobility |
| `neff`, `neff_per_col` | Chai-1's own retrieved MSA | MSA depth |
| `max_train_identity`, `n_pre_cutoff_hits` | RCSB search restricted to pre-2024 releases | novelty / leakage |
| `ligand_state` (apo/holo) | `nonpolymer_entity_count` | ligand confound |
| `ensemble_agreement` | pairwise lDDT across the 5 diffusion samples | free second confidence signal |

B-factors are z-scored *within* each structure because raw B-factors are not
comparable across refinements — different resolutions and refinement protocols
put them on different scales.

---

## 4. What it showed

Full tables in [`RESULTS.md`](RESULTS.md). The narrative:

### The headline number is good, and the headline number is misleading

Mean pLDDT 94.70, mean lDDT 0.9413, ECE 0.0057. Chai-1's confidence head is
close to honest on structures it has never seen. That is a genuine result and
worth stating plainly.

Then resolve it per target and the picture changes: median overconfidence
**−0.07 pp**, IQR [−1.13, +2.09], range [−7.18, +17.27], and **only 48.7% of
targets are overconfident at all**. The pooled +0.57 pp is not a population-wide
bias that you could correct with a global offset. It is a long right tail
dragging the mean. For anyone thresholding on pLDDT, the useful question is not
"what is the average correction" but "am I in the tail" — which is why the rest
of the study is about the tail.

### Within a protein: mobility, and the model under-adjusts for it

Three proxies, each measuring something structurally different — an experimental
refinement parameter, a geometric assignment, a surface property — all agree and
all are monotonic:

| proxy | rigid end | mobile end | ratio |
|---|---|---|---|
| B-factor z-score | +0.18 pp | +2.05 pp | 11.6× |
| Secondary structure | +0.20 (helix/sheet) | +0.93 (coil) | 4.7× |
| Solvent accessibility | +0.31 (buried) | +1.23 (exposed) | 4.0× |

Flexible − rigid = **+1.87 pp, 95% CI [+1.37, +2.37]** under target clustering.

The component columns explain the mechanism. Rigid → flexible: accuracy falls
7.98 pp, confidence falls 6.11 pp. Chai-1 *does* lower pLDDT in mobile regions —
it is not blind to them — but it lowers it by too little. **Confidence degrades
more slowly than accuracy.** That specific shape is what a well-trained but
insufficiently pessimistic confidence head looks like.

**And the closer you get to real disorder, the worse it gets.** All three proxies
above measure mobility among residues the crystallographer could model. A fourth
covariate reaches past that: the residues flanking an actual break in the
experimental chain, where density ran out entirely. Those 204 residues (across 90
targets) are over-claimed by **+8.65 pp against a global +0.57 pp** — 15.7×, and
**+8.10 pp, 95% CI [+4.34, +12.22]** under target clustering. It is the largest
effect in the study.

It is not mobility relabeled: holding flexibility fixed, gap adjacency still
multiplies overconfidence by ~5× (flexible 1.97 → 10.07 pp; intermediate
0.44 → 2.34 pp). And the under-adjustment is at its most extreme — accuracy
falls 18.4 pp at a gap boundary while confidence falls 10.3 pp. Chai-1 clearly
detects these residues; it still captures only half of what goes wrong.

This quantifies the caveat rather than removing it. Genuinely disordered residues
are *absent* from crystal structures, so they have no lDDT to score at all, and
the ≤10-unmodeled-residue filter admits only the mildest gaps. The +1.87 pp
mobility effect is a lower bound, and +8.65 pp at a chain break says how loose a
bound it is. A jump in residue numbering is not always missing density —
numbering conventions and engineered deletions produce one too — but those false
positives are ordinary residues that dilute the contrast toward zero, so the
figure is a floor.

### Across proteins: novelty, not mobility

This is the distinction that a single pooled analysis hides, and it took the tail
analysis to make it visible.

| | what varies | what predicts miscalibration |
|---|---|---|
| **within a protein** | which residue | mobility (+1.87 pp) |
| **across proteins** | which target | novelty (+1.22 pp) — *not* mobility |

Stratified by maximum sequence identity to any pre-2024 PDB entry, overconfidence
runs −0.30 pp (familiar) → +0.76 (mid) → +0.91 (novel), monotonic, and the
low − high contrast is **+1.22 pp, 95% CI [+0.65, +1.79]**. On *familiar* targets
Chai-1 is slightly **under**confident; over-claiming emerges only as targets get
novel. The same under-adjustment appears: accuracy falls 1.9 pp across the
novelty range, confidence falls only 0.77 pp.

Meanwhile a target's `frac_flexible` has **AUC 0.469, p = 0.64** for tail
membership — no relationship at all. The reason is mechanical rather than
surprising: every protein contains flexible loops, so mobility explains *where
inside a structure* to distrust the confidence, but cannot distinguish one target
from another. Novelty is a whole-target property, so it can.

Practically: a target's flexibility profile is not a usable warning sign. Low
identity to anything in the pre-cutoff PDB is.

### Mobility is the axis the other effects act along

Within each flexibility bin, the novelty effect amplifies: +0.88 pp on rigid
residues, +2.11 pp on flexible ones (2.4×). Chai-1 is well calibrated — and in
places under-confident — on rigid, familiar, ligand-bound structure. Its
miscalibration concentrates in mobile regions and compounds there.

### Two hypotheses that did not survive

**MSA depth: no support.** The marginal pattern is non-monotonic (high depth is
worse than mid), stratifying by flexibility flips the sign across strata, and
the target-clustered contrast is +0.15 pp, 95% CI [−0.49, +0.78]. The most
likely explanation is restriction of range: this target set is drawn from
well-characterized crystallizable proteins, and even the "low" depth tertile
holds hundreds to thousands of sequences. A set with genuinely shallow
alignments — tens of sequences — might well show the effect. We did not test it.

**Apo/holo: suggestive, underpowered.** Apo structures are marginally worse
(+1.02 vs +0.50 pp) and the gap survives stratification by flexibility, so it is
not purely mobility in disguise. But apo/holo is a *target-level* label — every
residue in a target carries the same value, so 15,069 apo residues contribute
nothing like 15,069 independent observations. Under target clustering:
+0.52 pp, 95% CI **[−0.18, +1.25]**. Not established. The mechanism is plausible
(the model predicts from sequence and cannot know a ligand is there to rigidify a
loop), and testing it properly needs a target set enriched for apo structures —
the baseline query yields only 122 apo clusters.

Both of these are cases where the residue-level analysis looked convincing and
the correct analysis did not. That is the main methodological lesson of the
project.

### Ensemble disagreement is redundant with pLDDT

Chai-1 gives five diffusion samples and we were scoring one. The four discards
are a free second opinion: where the samples agree the model converged, where
they disagree it did not. If disagreement flagged errors better than pLDDT, that
would be an easy win for anyone filtering on confidence.

| signal | Spearman vs lDDT | AUROC (flag worst decile) |
|---|---|---|
| **pLDDT** | **0.691** | **0.914** |
| ensemble agreement | 0.591 | 0.854 |
| combined (rank average) | 0.687 | 0.899 |

pLDDT wins on both, and the naive combination is *worse* than pLDDT alone. A
stronger test — regress lDDT on pLDDT, correlate agreement with the residual —
gives **−0.036**, indistinguishable from zero. Agreement carries no information
about accuracy that pLDDT does not already contain.

The reason is structural. The five samples share weights, MSA and trunk
representation, differing only in diffusion noise. Their spread measures
*sampling variance within one converged prediction*, not uncertainty about
whether that prediction is right. pLDDT comes from a head trained against true
structures, so it can express error modes on which all five samples happily
agree. **Agreement is blind exactly where the model is confidently wrong** —
which is the case you care about.

This is a negative result that doubles as a positive characterization of pLDDT:
a trained confidence head that beats the obvious model-free alternative and
subsumes it. Practically, scoring one model per target is sufficient.

### The tail, and a warning about how you define it

Sorting targets by overconfidence and scanning 20 covariates over the worst 20,
novelty ranks first (tail median 0.42 identity vs 0.91 for the rest) and
flexibility ranks nowhere. Nothing clears Bonferroni correction (p < 0.0025 for
20 tests), but novelty was a pre-registered hypothesis independently significant
in the main analysis, so this reads as a consistent second look rather than a
new claim.

But "worst by overconfidence" mixes two failure modes that matter differently:

| target | pLDDT | lDDT | |
|---|---|---|---|
| 8VO2_1 | 94.6 | 0.773 | confident and wrong |
| 9FWA_1 | 62.8 | 0.504 | says it does not know, still over-claims |

The second is largely benign — pLDDT in the 60s already tells you not to trust
it. The first is the one that costs something. Redefining the tail as *high
confidence, low accuracy* (pLDDT > 90, lDDT < 0.85 → 9 targets; loosened to
> 85 / < 0.88 → 41 targets) selects a different group, and **novelty stops being
the top separator.** What is stable across both cuts is a size/exposure cluster:
these targets are shorter and more solvent-exposed. Nothing survives Bonferroni
at either threshold, and the three covariates involved (mean RSA, residue count,
entity length) are not independent measurements — shorter proteins have higher
surface-to-volume ratio. So this is a lead, not a result.

The dissociation itself is the finding:

| tail definition | selects | separated by |
|---|---|---|
| worst by overconfidence | over-claiming, often at low pLDDT | novelty |
| high pLDDT, low lDDT | confident and wrong | size / exposure |

Novelty explains why the model over-claims. It does not explain the failures
that matter operationally.

---

## 5. What this says about Chai-1

**The confidence head works, and it is the good kind of imperfect.** ECE 0.0057
on genuinely unseen structures, and a confidence signal that beats the obvious
model-free alternative and fully subsumes it. Where it errs, it errs by
under-adjusting rather than by being uninformative — pLDDT moves in the right
direction on every covariate we tested, just not far enough. That is a much more
tractable failure than a confidence score that is simply uncorrelated with error.

**Its failure mode is concentrated, not diffuse.** Half the targets are
under-confident. The problem is a minority tail, which means the useful
intervention is *identifying* tail targets, not applying a global correction.
Subtracting 0.57 pp from every pLDDT would make the typical protein worse.

**It has not fully learned "I have not seen anything like this."** The clearest
statement of the gap: on familiar targets Chai-1 is slightly under-confident; on
novel ones it over-claims. Accuracy responds to novelty about 2.5× more strongly
than confidence does. The confidence head has learned that mobile, novel regions
are harder — it just systematically underestimates by how much. That is
consistent with a head trained on a distribution where near-homologs are usually
available, being run where they are not.

**Its remaining confidently-wrong cases are not explained by anything we
measured.** Nine targets have pLDDT > 90 and lDDT < 0.85. That is the population
a user would actually be burned by, and neither novelty, flexibility, MSA depth
nor ligand state separates it. The suggestive signal is size and surface
exposure, at n = 9–41 and uncorrected. Whatever is going on there is the most
interesting open question in this dataset.

**For a practitioner, the operational summary is:** trust pLDDT; it is close to
honest and you cannot do better with a cheap model-free trick. Distrust it more
than you otherwise would in flexible, exposed regions, and on targets with low
sequence identity to the pre-cutoff PDB. Do not bother computing ensemble
disagreement. And note that a pLDDT filter set anywhere below ~94 admits every
confidently-wrong target we found.

---

## 6. What we did not establish

1. **Apo/holo** — direction consistent, mechanism plausible, CI includes zero.
   Needs an apo-enriched target set.
2. **MSA depth** — no support found, but restriction of range means we tested a
   narrow slice. A shallow-MSA set is the missing experiment.
3. **The disorder effect is a lower bound.** Genuinely disordered residues are
   absent from crystals and excluded by our filters. The most flexible target
   here has a maximum B-factor around 44 Å², well below what an intrinsically
   disordered region would show. The +1.87 pp is what mobility does *among
   well-ordered residues*.
4. **The confidently-wrong tail is characterized, not explained** (§4).
5. **MCE ≈ 0.19–0.21 recurs across strata**, which suggests one sparse
   confidence bin dominates the maximum-error statistic. Read the reliability
   diagram before quoting MCE.
6. **11,017 residues have no novelty value** and are the worst-calibrated group
   in every table (+1.55 pp, lDDT 0.889). These targets have not been
   characterized. This is the largest loose end in the dataset.
7. **The cutoff assumption is load-bearing.** The design assumes Chai-1's
   training data ends at or before 2024-01-01. If it extends later, some targets
   are contaminated, and contamination biases toward *better* apparent
   calibration — so the reported overconfidence would be an underestimate. The
   novelty covariate partly guards against this: it measures identity to
   pre-2024 entries directly, and the fact that overconfidence *rises* with
   novelty is what you would expect if the pre-2024 material is what the model
   learned from.
8. **Monomers only, X-ray only, 100–400 residues, ≤2.0 Å.** Nothing here
   generalizes to complexes, to nucleic acids, to cryo-EM structures, or to the
   ligand and multi-chain modes that are Chai-1's actual differentiator from
   AlphaFold2. This is a study of one mode of one model.

---

## 7. What to run next

Ordered by value per GPU-hour:

1. **Test the size hypothesis properly.** It came out of a scan, so it needs a
   pre-specified confirmation: predict a matched set of short (100–150 aa) and
   long (300–400 aa) targets from the existing candidate pool and compare the
   rate of pLDDT > 90 / lDDT < 0.85. One test, no multiplicity penalty — it
   either promotes the lead or kills it.
2. **Characterize the 11,017-residue no-novelty stratum.** No GPU needed. It is
   the worst-calibrated group in the dataset and we do not know what it is.
3. **An apo-enriched target set** to give the ligand hypothesis a fair test.
4. **A shallow-MSA target set** to give the MSA-depth hypothesis a fair test.
5. **Spot-check the 204 gap-flanking residues** against their mmCIF. The effect
   is large, significant and spread over 90 targets, but it rests on residue
   numbering as a proxy for missing density. At n = 204 that is checkable by
   hand, and worth checking before the number is quoted anywhere.

---

## 8. Reproducing

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # add chai_lab on a GPU host

./scripts/run_all.sh                     # full pipeline, resumable

python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility
python scripts/calibration.py --scores data/analysis/all_residues.csv --by flexibility,ligand_state
python scripts/cluster_analysis.py --scores data/analysis/all_residues.csv \
    --contrast flexibility:flexible-rigid --contrast novelty_bin:low-high
python scripts/ensemble_agreement.py --compare
python scripts/tail_analysis.py --top 20
python scripts/tail_analysis.py --min-plddt 90 --max-lddt 0.85
```

Stages 2–3 need network egress (RCSB / wwPDB, the MSA server) and a GPU. A 24 GB
card is ample — a 179-residue target uses about 2 GB. The full 512-target batch
took about a day on an L4.

---

## Glossary

| term | meaning |
|---|---|
| **lDDT** | Local Distance Difference Test — realized per-residue accuracy, 0–1, superposition-free |
| **pLDDT** | The model's *predicted* lDDT for a residue, 0–100. Written into the mmCIF B-factor column |
| **Calibration** | Whether predicted confidence matches realized accuracy |
| **ECE / MCE** | Expected / Maximum Calibration Error — average and worst-bin confidence-accuracy gap |
| **Overconfidence** | Signed gap: mean confidence − mean accuracy. Positive = over-claiming |
| **MSA** | Multiple Sequence Alignment — evolutionarily related sequences, the model's main evidence |
| **Neff** | Effective MSA depth after down-weighting near-duplicate sequences |
| **B-factor** | Crystallographic measure of atomic positional uncertainty; a proxy for local mobility |
| **RSA** | Relative Solvent Accessibility — how exposed a residue is |
| **Apo / holo** | Without / with a bound ligand |
| **Target clustering** | Treating each protein, not each residue, as the unit of resampling |
| **Bonferroni correction** | Dividing the significance threshold by the number of tests run |
