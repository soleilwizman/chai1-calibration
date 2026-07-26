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
| 2 | Download ground-truth structures | `download_structures.py` | `data/raw/cif/*.bcif.gz` |
| 3 | Run Chai-1 predictions | `run_predictions.py` | `predictions/<candidate>/*.cif` |
| 4 | Compute per-residue lDDT vs pLDDT | `compute_lddt.py` | `data/analysis/<candidate>.csv` |
| 5 | Calibration analysis | `calibration.py` | metrics + reliability diagram |

The core comparison is per residue: **lDDT** (realized accuracy, from stage 4)
against **pLDDT** (Chai-1's predicted confidence, written into the mmCIF B-factor
column). A calibrated model has `pLDDT ≈ 100 × lDDT` within any confidence bin;
`calibration.py` reports ECE, MCE, and signed overconfidence, and can stratify
every metric by a covariate (`--by`) to test the hypotheses above.

### Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### Running the pipeline
```bash
# Stage 2: fetch experimental ground truth (needs RCSB egress access)
python scripts/download_structures.py

# Stage 3: stage inputs (any host), then predict on a GPU host with chai_lab
python scripts/run_predictions.py --dry-run      # writes input FASTAs only
python scripts/run_predictions.py                # GPU + chai_lab required

# Stage 4: score each prediction against its reference
python scripts/compute_lddt.py --ref data/raw/cif/10AF.bcif.gz \
    --pred predictions/10AF_1/pred.cif --out data/analysis/10AF_1.csv

# Stage 5: concatenate the per-target CSVs, optionally merge covariates, analyze
python scripts/calibration.py --scores data/analysis/all_residues.csv \
    --plot data/analysis/reliability.png --by nonpolymer_entity_count
```

> **Note:** stages 2–3 require outbound network / GPU access. In sandboxes where
> RCSB hosts (`files.rcsb.org`, `data.rcsb.org`, `models.rcsb.org`) are blocked by
> egress policy, stage 2 cannot run; the code is otherwise environment-agnostic.

## Files
- `data/targets/q.json`: baseline RCSB search query
- `data/targets/candidates_raw.json`: representative target entities
- `data/targets/candidates_covariates.json`: per-target metadata covariates
- `scripts/extract_searchable_attrs.py`: extract searchable schema attributes from `schema.json`
- `scripts/extract_candidate_covariates.py`: fetch candidate covariate data for analysis
- `scripts/download_structures.py`: stage 2 — download experimental mmCIF ground truth
- `scripts/run_predictions.py`: stage 3 — write Chai-1 FASTAs and run predictions
- `scripts/compute_lddt.py`: stage 4 — per-residue lDDT vs pLDDT from two structures
- `scripts/calibration.py`: stage 5 — reliability curve, ECE/MCE, covariate stratification
