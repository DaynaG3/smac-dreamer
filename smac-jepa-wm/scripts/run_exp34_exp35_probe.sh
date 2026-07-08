#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CKPT=runs/rnn_seqmem_exp34_dreamer_two_mask_xxx/checkpoint.pt bash scripts/run_exp34_exp35_probe.sh
# or:
#   CKPT=runs/rnn_seqmem_exp35_dreamer_simple_loss_xxx/checkpoint.pt bash scripts/run_exp34_exp35_probe.sh

CKPT=${CKPT:?Set CKPT=/path/to/checkpoint.pt}
OUT_DIR=${OUT_DIR:-sanity_outputs/$(basename "$(dirname "$CKPT")")_decode_probe}
TRAIN_EP=${TRAIN_EP:-data/r2_general_2100_full/train/shard_02/r2g_train_0083.npz}
VAL_EP=${VAL_EP:-data/r2_general_2100_full/validation/shard_03/r2g_validation_1300.npz}
DEVICE=${DEVICE:-cuda}

python tools/sanity_decode_rollout_exp33.py \
  --checkpoint "$CKPT" \
  --train-episode "$TRAIN_EP" \
  --val-episode "$VAL_EP" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --episode-index 0 \
  --rollout-horizon 5 \
  --xy-timesteps 1 2 5 10 20 \
  --table-timesteps 1 2 5 10 20 \
  --value-table-features important

echo "[done] wrote probe to $OUT_DIR"
