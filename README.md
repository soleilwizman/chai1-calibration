# chai1-calibration

Is Chai-1's per-residue confidence (pLDDT) honest on structures it has never
seen? We predicted 511 crystal structures released after the training cutoff and
compared claimed confidence against realized Cα lDDT.

- **[RESULTS.md](RESULTS.md)** 

Chai-1 is close to honest overall (ECE 0.0057, +0.57 pp over-claim. When Chai-1 generates five structural samples per inference run, ensemble disagreement across those samples typically remains minimal on rigid core regions, concentrating instead on flexible loops, intrinsically disordered regions (IDRs), or ambiguous ligand/interface poses. Miscalibration
concentrates in mobile regions within a protein (+1.87 pp) and on novel targets
across proteins (+1.22 pp). Residues around
an unmodeled stretch are over-claimed by +8.10 pp, 15.7x global. MSA
depth surpringly has no effect here. 

## Candidate filtering sensitivity
The baseline candidate pool is defined as:
- resolution ≤ 2.0
- R-free ≤ 0.25
- unmodeled residues ≤ 10
- single model, single polymer entity, X-ray only
- sequence length 100–400

This query returns 2,814 entities collapsing to 559 clusters when deduplicated at 30% sequence identity.

### Non-standard monomer exclusion (559 → 512)
Targets containing non-standard monomers (selenomethionine, phosphoserine, etc) are excluded
during covariate extraction. Chai-1 predicts only standard amino acids.

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
- Relaxing resolution to 2.5 Å adds only 72 clusters, while removing the R-free filter adds 37 clusters, so the R-free cutoff is the stronger quality constraint. The free R-factor prevents overfitting 3D models to X-ray diffraction data, setting aside a subset of diffraction data that is never used during model refinement.
- Deduplication at 30% sequence identity is performed server-side by the RCSB query with `group_by`.

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | Target selection & covariates | `extract_candidate_covariates.py` | `data/targets/candidates_covariates.json` |
| 2 | Download ground-truth structures | `download_structures.py` | `data/raw/cif/*.cif.gz` |
| 3 | Run Chai-1 predictions | `run_predictions.py` | `predictions/<candidate>/output/pred.model_idx_0.cif` |
| 3b | MSA-depth covariate trial | `extract_msa_depth.py` | `data/targets/msa_depth.json` |
| 3c | Training-identity covariate trial | `extract_training_identity.py` | `data/targets/training_identity.json` |
| 4 | Compute per-residue lDDT vs pLDDT | `compute_lddt.py` | `data/analysis/per_target/<candidate>.csv` |
| 4b | Merging lDDT + covariates | `build_dataset.py` | `data/analysis/all_residues.csv` |
| 5 | Calibration analysis | `calibration.py` | metrics + reliability diagram |


### Methodology notes  
- **Residues are paired by sequence alignment, not residue number.** The
  prediction is numbered 1..N from the FASTA; the experimental structure may use
  a different numbering (offsets, gaps, etc). `compute_lddt.py`
  globally aligns the two sequences, so tags/offsets/chain-ID differences don't
  silently corrupt the score. Each target reports `coverage` (fraction of
  reference residues aligned); low coverage is flagged, not trusted.
- **lDDT is Cα by default** (`--metric ca`), matching what AlphaFold-style pLDDT
  is trained to predict. `--metric all-atom` is available but is *not* the right
  comparison for pLDDT calibration.
- **Disorder proxies**
  `ref_bfactor_z` (B-factor z-scored within each structure - raw B-factors
  aren't comparable across refinements), `rsa`, `sse`, and `near_chain_gap`.
Note that genuinely disordered residues are usually *absent* from a crystal
  structure and so have no lDDT to score; the first three measure flexibility
  among the residues that *were* modelled, which is a lower bound on the disorder
  effect. `near_chain_gap` -- residues flanking an actual break in the chain --
  reaches closest to real disorder and carries by far the largest effect
  (+8.10 pp, 95% CI [+4.34, +12.22]); see RESULTS.md section 2.

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

Chai-1 downloads its model weights on first run (several GB) -- then the results will start folwing in once the model takes time to download.

### Running the pipeline

The whole batch was made into one resumable command such that it'd be safe to interrupt an rerun if using rented gpu. 

```bash
mkdir -p logs
tmux new -s chai                      # so it survives a dropped connection
./scripts/run_all.sh 2>&1 | tee logs/run_all.log
# detach: Ctrl-B then D   |   reattach: tmux attach -t chai
```

Knobs: `MAX_TARGETS=N` (smoke test), `SKIP_DOWNLOAD=1`, `SKIP_PREDICT=1`
(score on a CPU box), `SKIP_IDENTITY=1`.

### Preserving a run before the GPU host is destroyed

`predictions/`, `data/analysis/` and `data/raw/cif/` are all gitignored, so a
finished batch is on one machine. 

```bash
pip install -U huggingface_hub && hf auth login
hf upload <user>/chai1-calibration-run predictions predictions --repo-type dataset
hf upload <user>/chai1-calibration-run data/raw/cif data/raw/cif --repo-type dataset'''

> **Note:** stages 2–3 require GPU access. Stage 2 pulls
> structures from the appropriate databases, and so in
> sandboxes where all of these are blocked, it can't be run.

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
- `scripts/reliability_curve.py`: the reliability curve itself -- per-bin gap with
  clustered CIs, an MCE diagnosis, and a pLDDT-threshold table
- `scripts/cluster_analysis.py`: target-clustered bootstrap CIs (residues are not independent)
- `scripts/ensemble_agreement.py`: per-residue agreement across Chai-1's 5 samples, vs pLDDT
- `scripts/tail_analysis.py`: which covariates distinguish the badly-calibrated targets
- `scripts/run_all.sh`: driver — runs every stage end to end, resumable
- `scripts/archive_run.sh`: preserve a finished run before tearing down the GPU host
