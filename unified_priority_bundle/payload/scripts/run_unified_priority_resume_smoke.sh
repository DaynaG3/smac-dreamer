#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
REPO="${REPO:-$ROOT/smac-dreamer}"
PY="${PY:-$ROOT/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained R2 latest.pt}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_unified_priority.yaml}"
SOURCE_RUN_META="${SOURCE_RUN_META:-$(dirname "$CHECKPOINT")/run_meta.json}"
RESUME_START_STEP="${RESUME_START_STEP:-}"
SMOKE_ADDITIONAL_STEPS="${SMOKE_ADDITIONAL_STEPS:-5000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN:-$REPO/logs/r2dreamer/exp40_unified_priority_smoke_${STAMP}}"

"$PY" "$REPO/scripts/preflight_unified_priority.py" \
  --repo "$REPO" --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" --source-run-meta "$SOURCE_RUN_META" \
  --json-out "/tmp/unified_priority_preflight_${STAMP}.json"

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
  echo "[FAIL] checkpoint has no positive step; set RESUME_START_STEP manually." >&2
  exit 2
fi

END_STEP=$((START_STEP + SMOKE_ADDITIONAL_STEPS))
mkdir -p "$RUN"

echo "[SMOKE] checkpoint step : $START_STEP"
echo "[SMOKE] absolute end step: $END_STEP"
echo "[SMOKE] new run dir      : $RUN"

cd "$REPO"
"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT" \
  --resume-start-step "$START_STEP" \
  --logdir "$RUN" \
  --steps "$END_STEP" \
  2>&1 | tee "$RUN/smoke.log"

"$PY" scripts/preflight_unified_priority.py \
  --repo "$REPO" --checkpoint "$RUN/latest.pt"
"$PY" scripts/inspect_unified_priority_checkpoint.py \
  "$RUN/latest.pt" --require-adaptive --min-step "$END_STEP"
"$PY" scripts/assert_unified_priority_metrics.py \
  "$RUN" --require-nonuniform

grep -E \
  "priority/(sequence_raw|is_weight|effective_sample_size|map_entropy|map_probability)" \
  "$RUN"/metrics.jsonl "$RUN"/smoke.log 2>/dev/null | tail -50 || true

echo "[OK] smoke continuation finished. Do not launch the full run until its"
echo "     latest.pt resumes successfully and priority metrics are finite/non-uniform."
