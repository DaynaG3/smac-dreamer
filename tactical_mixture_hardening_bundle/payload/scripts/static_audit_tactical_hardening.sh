#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
ROOT="${ROOT:-$(dirname "$REPO")}"
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_hardened.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
SOURCE_RUN_META="${SOURCE_RUN_META:-}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }

bash -n \
  "$REPO/scripts/static_audit_tactical_hardening.sh" \
  "$REPO/scripts/run_tactical_hardened_2m.sh"

"$PY" -m py_compile \
  "$REPO/external/r2dreamer/tactical_policy.py" \
  "$REPO/external/r2dreamer/dreamer.py" \
  "$REPO/scripts/train_r2dreamer_smaclite_multimap.py" \
  "$REPO/src/smacdreamer/validation_trainer.py" \
  "$REPO/scripts/audit_tactical_hardening.py" \
  "$REPO/scripts/assert_tactical_hardened_metrics.py"

PYTHONPATH="$REPO/external/r2dreamer:$REPO/src" \
"$PY" -m pytest -q \
  "$REPO/tests/test_tactical_policy_hardened.py"

AUDIT=(
  "$PY" "$REPO/scripts/audit_tactical_hardening.py"
  --repo "$REPO"
  --config "$CONFIG"
)
if [[ -n "$CHECKPOINT" ]]; then
  AUDIT+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$SOURCE_RUN_META" ]]; then
  AUDIT+=(--source-run-meta "$SOURCE_RUN_META")
fi
"${AUDIT[@]}"

git -C "$REPO" diff --check -- .

echo "[OK] tactical hardening static audit passed"
