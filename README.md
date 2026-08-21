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

