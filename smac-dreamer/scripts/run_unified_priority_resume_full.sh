#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
REPO="${REPO:-$ROOT/smac-dreamer}"
PY="${PY:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained R2 latest.pt}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_unified_priority.yaml}"
SOURCE_RUN_META="${SOURCE_RUN_META:-$(dirname "$CHECKPOINT")/run_meta.json}"
RESUME_START_STEP="${RESUME_START_STEP:-}"
FINAL_STEP="${FINAL_STEP:-2000000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN:-$REPO/logs/r2dreamer/exp40_unified_priority_resume_2m_${STAMP}}"

"$PY" "$REPO/scripts/preflight_unified_priority.py" \
  --repo "$REPO" --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" --source-run-meta "$SOURCE_RUN_META"

START_STEP="$("$PY" - "$CHECKPOINT" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("step", 0)))
PY
)"
if [[ -n "$RESUME_START_STEP" ]]; then
  START_STEP="$RESUME_START_STEP"
fi

if [[ "$START_STEP" -le 0 ]]; then
  echo "[FAIL] checkpoint has no positive step." >&2
  exit 2
fi
if [[ "$START_STEP" -ge "$FINAL_STEP" ]]; then
  echo "[FAIL] checkpoint step $START_STEP is already >= FINAL_STEP $FINAL_STEP" >&2
  exit 2
fi

mkdir -p "$RUN"
printf '%s\n' \
  "checkpoint=$CHECKPOINT" \
  "checkpoint_step=$START_STEP" \
  "final_step=$FINAL_STEP" \
  "config=$CONFIG" \
  > "$RUN/RESUME_METADATA.txt"

cd "$REPO"
"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT" \
  --resume-start-step "$START_STEP" \
  --logdir "$RUN" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN/train.log"

"$PY" "$REPO/scripts/preflight_unified_priority.py" \
  --repo "$REPO" --checkpoint "$RUN/latest.pt"
"$PY" "$REPO/scripts/inspect_unified_priority_checkpoint.py" \
  "$RUN/latest.pt" --require-adaptive --min-step "$FINAL_STEP"
