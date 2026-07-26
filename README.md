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

## Files
- `data/targets/q.json`: baseline RCSB search query
- `data/targets/candidates_raw.json`: representative target entities
- `scripts/extract_searchable_attrs.py`: extract searchable schema attributes from `schema.json`
- `scripts/extract_candidate_covariates.py`: scaffold candidate covariate data for analysis
