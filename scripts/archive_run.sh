#!/usr/bin/env bash
# Preserve a completed run before the GPU host is destroyed.
#
# Everything the batch produced -- predictions/, data/analysis/, data/raw/cif/ --
# is gitignored, so it exists on exactly one machine. This script does the parts
# that need no external account and cost nothing:
#
#   1. writes an inventory + checksums (so a later restore can be verified)
#   2. force-adds the small derived outputs to git, commits them
#   3. optionally builds tarballs of the large directories
#
# It does NOT push and does NOT delete anything.
#
#     ./scripts/archive_run.sh                 # inventory + git commit
#     ./scripts/archive_run.sh --tar           # also build tarballs
#     ./scripts/archive_run.sh --tar --no-git  # tarballs only
#
# Then get the large data off the box (see REPORT/README "Preserving a run"),
# verify, and only then terminate the instance.
set -uo pipefail
cd "$(dirname "$0")/.."

DO_TAR=0
DO_GIT=1
for a in "$@"; do
  case "$a" in
    --tar) DO_TAR=1 ;;
    --no-git) DO_GIT=0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

OUT="archive"
mkdir -p "$OUT"
MANIFEST="$OUT/MANIFEST.txt"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ------------------------------------------------------------- 1. inventory
log "Building inventory -> $MANIFEST"
{
  echo "chai1-calibration run inventory"
  echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:      $(hostname)"
  echo "git:       $(git rev-parse HEAD 2>/dev/null || echo '(not a repo)')"
  echo
  echo "--- counts ---"
  printf "%-34s %s\n" "reference structures"   "$(ls data/raw/cif 2>/dev/null | wc -l)"
  printf "%-34s %s\n" "prediction dirs"        "$(ls -d predictions/*/ 2>/dev/null | wc -l)"
  printf "%-34s %s\n" "predicted models (.cif)" "$(find predictions -name 'pred.model_idx_*.cif' 2>/dev/null | wc -l)"
  printf "%-34s %s\n" "score files (.npz)"     "$(find predictions -name 'scores.model_idx_*.npz' 2>/dev/null | wc -l)"
  printf "%-34s %s\n" "MSAs (.aligned.pqt)"    "$(find predictions -name '*.aligned.pqt' 2>/dev/null | wc -l)"
  printf "%-34s %s\n" "per-target lDDT tables" "$(ls data/analysis/per_target 2>/dev/null | wc -l)"
  echo
  echo "--- sizes ---"
  du -sh predictions data/raw/cif data/analysis logs 2>/dev/null
  echo
  echo "--- derived analysis files ---"
  ls -l data/analysis/*.csv data/analysis/*.png data/targets/*.json 2>/dev/null
  echo
  echo "--- free space ---"
  df -h . | tail -1
} > "$MANIFEST"
cat "$MANIFEST"

log "Checksumming small derived outputs"
sha256sum data/analysis/*.csv data/analysis/*.png data/targets/msa_depth.json \
          data/targets/training_identity.json 2>/dev/null > "$OUT/checksums_analysis.txt"
log "  $(wc -l < "$OUT/checksums_analysis.txt") files checksummed"

# Per-target file list, so a partial restore is detectable target by target.
find predictions -type f \( -name '*.cif' -o -name '*.npz' -o -name '*.pqt' \) \
  -printf '%s\t%p\n' 2>/dev/null | sort -k2 > "$OUT/predictions_filelist.tsv"
log "  $(wc -l < "$OUT/predictions_filelist.tsv") prediction files listed"

# ---------------------------------------------------------------- 2. git
if [ "$DO_GIT" = "1" ]; then
  log "Committing small derived outputs (force-add past .gitignore)"
  # all_residues.csv is tens of MB; store it compressed.
  if [ -f data/analysis/all_residues.csv ] && [ ! -f data/analysis/all_residues.csv.gz ]; then
    gzip -kn data/analysis/all_residues.csv
  fi
  git add -f \
    data/analysis/all_residues.csv.gz \
    data/analysis/per_target_calibration.csv \
    data/analysis/ensemble_agreement.csv \
    data/analysis/reliability.png \
    data/targets/msa_depth.json \
    data/targets/training_identity.json \
    "$OUT/MANIFEST.txt" "$OUT/checksums_analysis.txt" \
    "$OUT/predictions_filelist.tsv" 2>/dev/null
  git add -f data/analysis/per_target 2>/dev/null
  if git diff --cached --quiet; then
    log "  nothing new to commit"
  else
    git commit -q -m "Archive run outputs: per-residue scores, per-target calibration, covariates

Derived outputs from the 511-target batch, committed so they survive the GPU
host. Predictions and reference structures are too large for git and are
archived separately; archive/MANIFEST.txt records what existed on the host."
    log "  committed -- now: git push -u origin \$(git rev-parse --abbrev-ref HEAD)"
  fi
fi

# ---------------------------------------------------------------- 3. tarballs
if [ "$DO_TAR" = "1" ]; then
  # Tarring needs room for a second copy. Check before starting.
  need_kb=$(du -sk predictions data/raw/cif 2>/dev/null | awk '{s+=$1} END {print s}')
  free_kb=$(df -Pk . | awk 'NR==2 {print $4}')
  log "predictions+refs: $((need_kb/1024)) MB, free: $((free_kb/1024)) MB"
  if [ "${need_kb:-0}" -gt "${free_kb:-0}" ]; then
    log "NOT ENOUGH DISK to tar locally. Upload the directories directly instead"
    log "  (hf upload / rsync), which needs no second copy. Skipping tarballs."
  else
    if command -v zstd >/dev/null 2>&1; then C="--zstd"; X="tar.zst"; else C="-z"; X="tar.gz"; fi
    log "Building archive/predictions_structures.$X (models + scores)"
    tar $C -cf "$OUT/predictions_structures.$X" \
      --exclude='*.pqt' predictions
    log "Building archive/predictions_msas.$X (retrieved alignments)"
    find predictions -name '*.aligned.pqt' -print0 \
      | tar $C -cf "$OUT/predictions_msas.$X" --null -T -
    log "Building archive/reference_structures.tar (already-gzipped mmCIF)"
    tar -cf "$OUT/reference_structures.tar" data/raw/cif
    ( cd "$OUT" && sha256sum ./*."$X" ./reference_structures.tar > checksums_archives.txt )
    ls -lh "$OUT"
  fi
fi

log "Done. Nothing has been deleted and nothing has been pushed."
log "Verify the copies are readable OFF this host before terminating it."
