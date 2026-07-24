#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/workspace/dreamer/combined-upload/smac-dreamer}"
PY="${PY:-$HOME/workspace/dreamer/combined-upload/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
SOURCE_RUN_META="${SOURCE_RUN_META:-}"

"$PY" -m py_compile \
  "$REPO/external/r2dreamer/tactical_policy.py" \
  "$REPO/external/r2dreamer/dreamer.py" \
  "$REPO/scripts/train_r2dreamer_smaclite_multimap.py" \
  "$REPO/scripts/preflight_tactical_mixture.py" \
  "$REPO/scripts/assert_tactical_metrics.py"

bash -n "$REPO/scripts/run_tactical_mixture_2m.sh"

PYTHONPATH="$REPO/external/r2dreamer:$REPO/src" \
"$PY" -m pytest -q "$REPO/tests/test_tactical_policy.py"

PREFLIGHT=(
  "$PY" "$REPO/scripts/preflight_tactical_mixture.py"
  --repo "$REPO"
  --config "$CONFIG"
)
if [[ -n "$CHECKPOINT" ]]; then
  PREFLIGHT+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$SOURCE_RUN_META" ]]; then
  PREFLIGHT+=(--source-run-meta "$SOURCE_RUN_META")
fi
"${PREFLIGHT[@]}"

grep -q "from buffer import Buffer" \
  "$REPO/scripts/train_r2dreamer_smaclite_multimap.py"
grep -q "using original uniform SliceSampler" \
  "$REPO/scripts/train_r2dreamer_smaclite_multimap.py"
grep -q "modules\['tactical_policy'\]" \
  "$REPO/external/r2dreamer/dreamer.py"
grep -q "self._frozen_tactical_policy" \
  "$REPO/external/r2dreamer/dreamer.py"

git -C "$REPO" diff --check -- .

echo "[OK] tactical-mixture static audit passed"
