#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python make_final_2m_config.py
python preflight_final_r2_jepa.py --jepa-checkpoint checkpoints/jepa/model.pt

RUN="logs/r2dreamer/exp33_jepa_final_beliefmask_slotadapter_grad_2m_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

echo "RUN=$RUN" | tee "$RUN/launch_meta.txt"
echo "CONFIG=configs/tmp_r2_2100_jepa_final_2m_wandb.yaml" | tee -a "$RUN/launch_meta.txt"
echo "JEPA_CKPT=checkpoints/jepa/model.pt" | tee -a "$RUN/launch_meta.txt"

echo "Launching 2M R2-Dreamer JEPA run with wandb online..."
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/tmp_r2_2100_jepa_final_2m_wandb.yaml \
  --jepa-checkpoint checkpoints/jepa/model.pt \
  --steps 2000000 \
  --logdir "$RUN" \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT:-smac-dreamer-jepa}" \
  2>&1 | tee "$RUN/train.log"
