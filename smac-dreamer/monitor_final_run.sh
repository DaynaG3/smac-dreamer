#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-$(ls -td logs/r2dreamer/exp33_jepa_final_beliefmask_slotadapter_grad_2m_* 2>/dev/null | head -1)}"
if [[ -z "${RUN}" || ! -d "$RUN" ]]; then
  echo "No final run directory found. Pass a RUN path explicitly."
  exit 1
fi

echo "RUN=$RUN"
echo "--- launch meta ---"
cat "$RUN/launch_meta.txt" 2>/dev/null || true

echo "--- recent important log lines ---"
grep -Ei "step|checkpoint|eval|validation|battle_won|win|reward|ret |jepa/feature_norm|trainable_adapter|maskh0|maskh4|imag_empty|invalid|nan|inf|traceback|error|wandb" "$RUN/train.log" 2>/dev/null | tail -120 || true
