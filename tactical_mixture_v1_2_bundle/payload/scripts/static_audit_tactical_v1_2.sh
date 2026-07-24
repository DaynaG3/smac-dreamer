#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${SOURCE_RUN_META:?set SOURCE_RUN_META}"
cd "$REPO"
"$PY" -m py_compile \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/tactical_policy.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_tactical_v1_2.py \
  scripts/assert_tactical_v1_2_metrics.py
"$PY" -m pytest -q tests/test_tactical_policy_v1_2.py
"$PY" scripts/audit_tactical_v1_2.py \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META"
echo "[OK] Tactical Mixture v1.2 static audit passed"
