#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(pwd)}"

if [[ -d "$ROOT/smac-jepa-wm" ]]; then
  COMBINED_ROOT="$(cd "$ROOT" && pwd)"
  JEPA_ROOT="$COMBINED_ROOT/smac-jepa-wm"
elif [[ -f "$ROOT/smac_jepa/train_jepa_exp40_dreamer.py" ]]; then
  JEPA_ROOT="$(cd "$ROOT" && pwd)"
  COMBINED_ROOT="$(dirname "$JEPA_ROOT")"
else
  echo "ERROR: ROOT must be combined-upload or smac-jepa-wm" >&2
  echo "Example: ROOT=~/workspace/dreamer/combined-upload ./install_exp45_pow2_direct.sh" >&2
  exit 1
fi

for required in \
  "$JEPA_ROOT/smac_jepa/train_jepa_exp40_dreamer.py" \
  "$JEPA_ROOT/smac_jepa/train_jepa_exp31_exp35.py"; do
  [[ -f "$required" ]] || { echo "ERROR: missing required Exp40 source: $required" >&2; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$COMBINED_ROOT/smac-jepa-wm_exp45_pow2_backup_$STAMP"
mkdir -p "$BACKUP/smac_jepa" "$BACKUP/scripts" "$BACKUP/tools" "$BACKUP/tests"

copy_with_backup() {
  local relative="$1"
  local source="$BUNDLE_ROOT/$relative"
  local destination="$JEPA_ROOT/$relative"
  [[ -f "$source" ]] || { echo "ERROR: bundle file missing: $source" >&2; exit 1; }
  if [[ -f "$destination" ]]; then
    mkdir -p "$BACKUP/$(dirname "$relative")"
    cp -a "$destination" "$BACKUP/$relative"
  fi
  mkdir -p "$(dirname "$destination")"
  cp -a "$source" "$destination"
}

FILES=(
  smac_jepa/pow2_direct_predictor.py
  smac_jepa/train_jepa_exp45_pow2_direct.py
  tools/make_exp40_eval_checkpoint.py
  tools/audit_exp45_pow2_checkpoint.py
  tools/eval_pow2_direct.py
  tools/eval_rnn_seqmem_dreamer_probe_r2aware_anchored.py
  scripts/run_exp45_pow2_direct_train.sh
  scripts/smoke_exp45_pow2_direct.sh
  scripts/eval_exp45_pow2_ordinary.sh
  scripts/eval_exp45_pow2_hidden.sh
  scripts/eval_exp45_pow2_all.sh
  scripts/static_audit_exp45_pow2.sh
  tests/test_pow2_direct_predictor.py
  tests/test_pow2_checkpoint_sanitizer.py
  README_EXP45_POW2_DIRECT.md
  COMMANDS_EXP45_POW2.md
  AUDIT_RESULTS.txt
)

for relative in "${FILES[@]}"; do copy_with_backup "$relative"; done
chmod +x "$JEPA_ROOT"/scripts/*exp45_pow2*.sh "$JEPA_ROOT/scripts/static_audit_exp45_pow2.sh"

cd "$JEPA_ROOT"
./scripts/static_audit_exp45_pow2.sh

echo
if [[ -f eval_rnn_seqmem_dreamer_probe_r2aware.py || -f tools/eval_rnn_seqmem_dreamer_probe_r2aware.py ]]; then
  echo "[OK] R2-aware ordinary evaluator found"
else
  echo "[WARN] eval_rnn_seqmem_dreamer_probe_r2aware.py not found; set ORDINARY_EVAL when evaluating"
fi
if [[ -f eval_jepa_exp31_exp33_anchored.py || -f tools/eval_jepa_exp31_exp33_anchored.py ]]; then
  echo "[OK] anchored ordinary/probe evaluator found"
else
  echo "[WARN] eval_jepa_exp31_exp33_anchored.py not found; set ANCHORED_EVAL when evaluating"
fi
if [[ -f eval_jepa_hidden_belief_exp31_exp33.py || -f tools/eval_jepa_hidden_belief_exp31_exp33.py ]]; then
  echo "[OK] hidden-belief evaluator found"
else
  echo "[WARN] eval_jepa_hidden_belief_exp31_exp33.py not found; set HIDDEN_EVAL when evaluating"
fi

echo
printf '[OK] Installed Exp45 Pow2 direct model into %s\n' "$JEPA_ROOT"
printf '[OK] Backup of replaced files: %s\n' "$BACKUP"
printf '[NEXT] Smoke: cd %s && EXP40_CHECKPOINT=<path> ./scripts/smoke_exp45_pow2_direct.sh\n' "$JEPA_ROOT"
