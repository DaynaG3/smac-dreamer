#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-$(dirname "$REPO")/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -f "$REPO/$CONFIG" ]] || { echo "[FAIL] config missing: $REPO/$CONFIG" >&2; exit 1; }
[[ -f "$REPO/$SOURCE_CONFIG" ]] || { echo "[FAIL] source config missing: $REPO/$SOURCE_CONFIG" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "[FAIL] checkpoint missing: $CHECKPOINT" >&2; exit 1; }

bash -n "$REPO/scripts/run_actor_critic_h15_800k.sh"
bash -n "$REPO/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh"
"$PY" -m py_compile "$REPO/scripts/audit_actor_critic_h15_800k.py"
"$PY" "$REPO/scripts/audit_actor_critic_h15_800k.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --source-config "$SOURCE_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --expected-sha256 "$EXPECTED_SOURCE_CHECKPOINT_SHA256"

echo "[OK] Tactical-v1.2 ordinary actor-critic H=15 static audit passed"
