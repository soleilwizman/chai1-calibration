# chai1-calibration

## Hypotheses
- pLDDT is systematically overconfident on residues in intrinsically disordered regions.
- Confidence degrades with MSA depth, and the calibration degrades faster than the accuracy does (i.e. the model does not know that it does not know).
- Targets with low maximum sequence identity to any pre-cutoff PDB entry are both less accurate and worse-calibrated.
- I'm measuring per-residue predicted pLDDT vs actual lDDT for PDB structures 2024-2026 (so not in pretraining).

## Candidate filtering sensitivity
The baseline candidate pool is defined as:
- resolution ≤ 2.0
- R-free ≤ 0.25
- unmodeled residues ≤ 10
- single model, single polymer entity, X-ray only
- sequence length 100–400

This query returns 2,814 entities collapsing to 559 clusters when deduplicated at 30% sequence identity.

We do not filter on `nonpolymer_entity_count` in the baseline. Apo/holo status should be recorded as a covariate and analyzed separately because it represents a real confound for sequence-only prediction.

### Sensitivity variants
| Variant | Entities | Clusters |
|---|---|---|
| Baseline | 2,814 | 559 |
| resolution ≤ 1.8 | 2,403 | 462 |
| resolution ≤ 2.5 | 3,075 | 631 |
| no R-free filter | 3,268 | 596 |
| unmodeled ≤ 5 | 2,371 | 418 |
| unmodeled ≤ 25 | 3,567 | 775 |
| dedup at 50% identity | 2,814 | 582 |
| dedup at 90% identity | 2,814 | 652 |
| apo only (`nonpolymer_entity_count = 0`) | 431 | 122 |
| no engineered mutations (`entity_poly.rcsb_mutation_count = 0`) | 2,262 | 482 |

### Notes
- The apo-only filter is intentionally avoided in the baseline because it reduces the pool from 559 clusters to only 122 clusters.
- Record `nonpolymer_entity_count` as a covariate rather than filtering apo-only. Chai-1 predicts apo from sequence alone, so holo structures are a genuine confound and should be analyzed as a covariate.
- Relaxing resolution to 2.5 Å adds only 72 clusters, while removing the R-free filter adds 37 clusters, so the R-free cutoff is the stronger quality constraint.
- Increasing unmodeled residue tolerance is the best lever for pool size, but comes with more structural ambiguity.
- Deduplication at 30% sequence identity is performed server-side by the RCSB query with `group_by`.

## Pipeline
The project runs as a sequence of stages. Stage 1 is complete (559 curated
targets + covariates are committed); the remaining stages are implemented as
scripts under `scripts/`.

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | Target selection & covariates | `extract_candidate_covariates.py` | `data/targets/candidates_covariates.json` |
| 2 | Download ground-truth structures | `download_structures.py` | `data/raw/cif/*.cif.gz` |
| 3 | Run Chai-1 predictions | `run_predictions.py` | `predictions/<candidate>/output/pred.model_idx_0.cif` |
| 3b | MSA-depth covariate (hyp. 2) | `extract_msa_depth.py` | `data/targets/msa_depth.json` |
| 3c | Training-identity covariate (hyp. 3) | `extract_training_identity.py` | `data/targets/training_identity.json` |
| 4 | Compute per-residue lDDT vs pLDDT | `compute_lddt.py` | `data/analysis/per_target/<candidate>.csv` |
| 4b | Merge lDDT + covariates | `build_dataset.py` | `data/analysis/all_residues.csv` |
| 5 | Calibration analysis | `calibration.py` | metrics + reliability diagram |

The core comparison is per residue: **lDDT** (realized accuracy, from stage 4)
against **pLDDT** (Chai-1's predicted confidence, written into the mmCIF B-factor
column). A calibrated model has `pLDDT ≈ 100 × lDDT` within any confidence bin;
`calibration.py` reports ECE, MCE, and signed overconfidence, and can stratify
every metric by a covariate (`--by`) to test the hypotheses above.

### Methodology notes (why the numbers are trustworthy)
- **Residues are paired by sequence alignment, not residue number.** The
  prediction is numbered 1..N from the FASTA; the experimental structure may use
  a different numbering (offsets, gaps, an unmodeled His-tag). `compute_lddt.py`
  globally aligns the two sequences, so tags/offsets/chain-ID differences don't
  silently corrupt the score. Each target reports `coverage` (fraction of
  reference residues aligned); low coverage is flagged, not trusted.
- **lDDT is Cα by default** (`--metric ca`), matching what AlphaFold-style pLDDT
  is trained to predict. `--metric all-atom` is available but is *not* the right
  comparison for pLDDT calibration.
- **Disorder proxies come from the experimental structure** (hypothesis 1):
  `ref_bfactor_z` (B-factor z-scored within each structure -- raw B-factors
  aren't comparable across refinements), `rsa`, `sse`, and `near_chain_gap`.
  Caveat: genuinely disordered residues are usually *absent* from a crystal
  structure and so have no lDDT to score; these measure flexibility among the
  residues that *were* modelled, which is a lower bound on the disorder effect.
- **Stratification bins are computed per target, not per residue**, so a few
  large proteins don't dominate the bin edges. Note residues within a protein are
  correlated: treat per-residue ECE as descriptive and cluster by `candidate` for
  any significance testing.

### Setup

Analysis-only host (download, scoring, covariates, calibration -- no GPU needed):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Prediction host (stage 3) additionally needs the model, which is GPU-only and so
is kept out of `requirements.txt`. On a bare CUDA box:

```bash
apt update && apt install -y git python3-venv python3-pip tmux   # prepend sudo if not root
git clone https://github.com/soleilwizman/chai1-calibration.git
cd chai1-calibration
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt chai_lab
nvidia-smi                                    # confirm the GPU is visible
```

Chai-1 downloads its model weights on first run (several GB), so the first
prediction takes noticeably longer than the rest. A 24 GB card is ample for this
target set: a 179-residue target uses ~2 GB.

### Running the pipeline

The whole batch is one resumable command (see `scripts/run_all.sh`); every stage
skips work already on disk, so it is safe to interrupt and re-run:

```bash
mkdir -p logs
tmux new -s chai                      # so it survives a dropped connection
./scripts/run_all.sh 2>&1 | tee logs/run_all.log
# detach: Ctrl-B then D   |   reattach: tmux attach -t chai
```

Knobs: `MAX_TARGETS=N` (smoke test), `SKIP_DOWNLOAD=1`, `SKIP_PREDICT=1`
(score on a CPU box), `SKIP_IDENTITY=1`.

Or run the stages individually:

```bash
# Stage 2: fetch experimental ground truth (needs RCSB egress access)
python scripts/download_structures.py

# Stage 3: stage inputs (any host), then predict on a GPU host with chai_lab
python scripts/run_predictions.py --dry-run      # writes input FASTAs only
python scripts/run_predictions.py                # GPU + chai_lab required

# Stage 3b/3c: covariates for the hypotheses (MSA depth, novelty vs pre-cutoff PDB)
python scripts/extract_msa_depth.py --pred-dir predictions --out data/targets/msa_depth.json
python scripts/extract_training_identity.py --out data/targets/training_identity.json

# Stage 4: score each prediction against its reference (Cα lDDT by default)
# Chai-1 writes predictions/<id>/output/pred.model_idx_0.cif (rank-0 model)
python scripts/compute_lddt.py --ref data/raw/cif/10AF.cif.gz \
    --pred predictions/10AF_1/output/pred.model_idx_0.cif \
    --out data/analysis/per_target/10AF_1.csv

# Stage 4b: merge all per-target lDDT tables with covariates into one table
python scripts/build_dataset.py --lddt-dir data/analysis/per_target \
    --out data/analysis/all_residues.csv

# Stage 5: analyze, stratifying by any covariate column
python scripts/calibration.py --scores data/analysis/all_residues.csv \
    --plot data/analysis/reliability.png --by msa_depth_bin   # or novelty_bin, ligand_state
```

> **Note:** stages 2–3 require outbound network / GPU access. Stage 2 pulls
> structures from `files.wwpdb.org` (the canonical wwPDB HTTPS egress host),
> falling back to the RCSB mirrors (`models.rcsb.org`, `files.rcsb.org`). In
> sandboxes where all of these are blocked by egress policy, stage 2 cannot run;
> the code is otherwise environment-agnostic.

## Files
- `data/targets/q.json`: baseline RCSB search query
- `data/targets/candidates_raw.json`: representative target entities
- `data/targets/candidates_covariates.json`: per-target metadata covariates
- `scripts/extract_searchable_attrs.py`: extract searchable schema attributes from `schema.json`
- `scripts/extract_candidate_covariates.py`: fetch candidate covariate data for analysis
- `scripts/download_structures.py`: stage 2 — download experimental mmCIF ground truth
- `scripts/run_predictions.py`: stage 3 — write Chai-1 FASTAs and run predictions
- `scripts/extract_msa_depth.py`: stage 3b — MSA depth / Neff from Chai-1 `.aligned.pqt` (or a3m)
- `scripts/extract_training_identity.py`: stage 3c — max identity to pre-cutoff PDB
- `scripts/compute_lddt.py`: stage 4 — sequence-aligned per-residue lDDT vs pLDDT
- `scripts/build_dataset.py`: stage 4b — merge per-target lDDT with covariates
- `scripts/calibration.py`: stage 5 — reliability curve, ECE/MCE, covariate stratification
- `scripts/run_all.sh`: driver — runs every stage end to end, resumable
