#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$HOME/workspace/dreamer/combined-upload/smac-dreamer}"
PY="${PY:-$HOME/workspace/dreamer/combined-upload/.venv/bin/python}"

"$PY" -m py_compile \
  "$REPO/src/smacdreamer/adaptive_priority.py" \
  "$REPO/external/r2dreamer/adaptive_buffer.py" \
  "$REPO/scripts/preflight_unified_priority.py" \
  "$REPO/scripts/assert_unified_priority_metrics.py" \
  "$REPO/scripts/inspect_unified_priority_checkpoint.py" \
  "$REPO/scripts/infer_resume_step.py" \
  "$REPO/scripts/train_r2dreamer_smaclite_multimap.py" \
  "$REPO/src/smacdreamer/envs/map_sampler.py" \
  "$REPO/src/smacdreamer/r2dreamer_factory.py" \
  "$REPO/src/smacdreamer/checkpointing.py" \
  "$REPO/external/r2dreamer/trainer.py" \
  "$REPO/external/r2dreamer/dreamer.py"

"$PY" "$REPO/scripts/preflight_unified_priority.py" --repo "$REPO"

grep -q "sampling_mode: adaptive_priority" \
  "$REPO/configs/r2_2100_jepa_unified_priority.yaml"

echo "[OK] static audit passed"
