#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
REPO="${REPO:-$ROOT/smac-dreamer}"
PY="${PY:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the old best_val_macro_winrate.pt}"
SOURCE_RUN_META="${SOURCE_RUN_META:?Set SOURCE_RUN_META to the old run_meta.json}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture.yaml}"
FINAL_STEP="${FINAL_STEP:-2000000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN:-$REPO/logs/r2dreamer/exp40_best_tactical_mixture_2m_${STAMP}}"

"$PY" "$REPO/scripts/preflight_tactical_mixture.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META"

if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null 2>&1; then
  echo "[FAIL] another multimap trainer appears to be running:" >&2
  pgrep -af 'train_r2dreamer_smaclite_multimap.py' >&2 || true
  exit 2
fi

mkdir -p "$RUN"
printf '%s\n' \
  "checkpoint=$CHECKPOINT" \
  "checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')" \
  "source_run_meta=$SOURCE_RUN_META" \
  "config=$CONFIG" \
  "new_phase_start=0" \
  "new_phase_final_step=$FINAL_STEP" \
  "adaptive_priority=false" \
  "tactical_duration=1" \
  > "$RUN/TACTICAL_RUN_METADATA.txt"

echo "$RUN" > "$ROOT/CURRENT_TACTICAL_MIXTURE_RUN.txt"
cd "$REPO"

"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT" \
  --resume-start-step 0 \
  --logdir "$RUN" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN/train.log"
