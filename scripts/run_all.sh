#!/usr/bin/env bash
# Run the full chai1-calibration pipeline end to end.
#
# Every stage is idempotent and resumable: re-running skips work that is already
# on disk, so it is safe to interrupt this script (Ctrl-C, dropped SSH, crash)
# and start it again -- it picks up where it left off.
#
# Because a full batch takes many hours, run it under tmux/nohup:
#
#     tmux new -s chai
#     ./scripts/run_all.sh 2>&1 | tee logs/run_all.log
#     # detach with Ctrl-B then D; reattach later with: tmux attach -t chai
#
# Environment knobs:
#   MAX_TARGETS=N   only process the first N targets (smoke test; default: all)
#   SKIP_DOWNLOAD=1 skip stage 2 (references already on disk / no RCSB egress)
#   SKIP_PREDICT=1  skip the GPU stage (e.g. scoring on a CPU box)
#   SKIP_IDENTITY=1 skip the RCSB sequence-identity covariate (needs egress)
set -uo pipefail   # deliberately NOT -e: one bad target must not kill the batch

cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p logs data/analysis/per_target

PY=python3
MAX_TARGETS="${MAX_TARGETS:-0}"
max_flag=""
[ "$MAX_TARGETS" -gt 0 ] 2>/dev/null && max_flag="--max $MAX_TARGETS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- stage 2
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  log "Stage 2: downloading experimental reference structures"
  $PY scripts/download_structures.py $max_flag
else
  log "Stage 2: skipped (SKIP_DOWNLOAD=1)"
fi
log "  references on disk: $(ls data/raw/cif 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- stage 3
if [ "${SKIP_PREDICT:-0}" != "1" ]; then
  log "Stage 3: running Chai-1 predictions (GPU; --skip-existing makes this resumable)"
  $PY scripts/run_predictions.py --skip-existing $max_flag
else
  log "Stage 3: skipped (SKIP_PREDICT=1)"
fi
log "  predictions complete: $(ls -d predictions/*/output/pred.model_idx_0.cif 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- stage 4
log "Stage 4: scoring predictions against references (Ca lDDT)"
scored=0; skipped=0; failed=0
for pred in predictions/*/output/pred.model_idx_0.cif; do
  [ -e "$pred" ] || continue
  id=$(basename "$(dirname "$(dirname "$pred")")")   # predictions/<id>/output/..
  entry="${id%_*}"                                    # 10AF_1 -> 10AF
  out="data/analysis/per_target/${id}.csv"

  if [ -s "$out" ]; then skipped=$((skipped+1)); continue; fi

  ref=""
  for cand in "data/raw/cif/${entry}.cif.gz" "data/raw/cif/${entry}.bcif.gz"; do
    [ -e "$cand" ] && ref="$cand" && break
  done
  if [ -z "$ref" ]; then
    echo "  WARN no reference structure for ${entry}, skipping ${id}" >&2
    failed=$((failed+1)); continue
  fi

  if $PY scripts/compute_lddt.py --ref "$ref" --pred "$pred" --out "$out" \
        >> logs/compute_lddt.log 2>&1; then
    scored=$((scored+1))
  else
    echo "  WARN scoring failed for ${id} (see logs/compute_lddt.log)" >&2
    failed=$((failed+1))
  fi
done
log "  scored=$scored already-done=$skipped failed=$failed"

# ------------------------------------------------------------ stages 3b/3c
log "Stage 3b: MSA depth covariate"
$PY scripts/extract_msa_depth.py --pred-dir predictions \
    --out data/targets/msa_depth.json

if [ "${SKIP_IDENTITY:-0}" != "1" ]; then
  log "Stage 3c: max sequence identity to pre-cutoff PDB (needs RCSB egress)"
  $PY scripts/extract_training_identity.py $max_flag \
      --out data/targets/training_identity.json \
      || log "  WARN training-identity step failed (network?); continuing without it"
else
  log "Stage 3c: skipped (SKIP_IDENTITY=1)"
fi

# ------------------------------------------------------------- stages 4b/5
log "Stage 4b: merging per-residue scores with covariates"
$PY scripts/build_dataset.py --lddt-dir data/analysis/per_target \
    --out data/analysis/all_residues.csv || exit 1

log "Stage 5: calibration analysis"
$PY scripts/calibration.py --scores data/analysis/all_residues.csv \
    --plot data/analysis/reliability.png | tee logs/calibration_overall.txt

for covariate in flexibility structured exposure msa_depth_bin novelty_bin ligand_state; do
  echo
  log "Stratified by ${covariate}"
  $PY scripts/calibration.py --scores data/analysis/all_residues.csv \
      --by "$covariate" 2>/dev/null | tee "logs/calibration_${covariate}.txt"
done

log "DONE. Reliability diagram: data/analysis/reliability.png"
log "     Merged table:        data/analysis/all_residues.csv"
