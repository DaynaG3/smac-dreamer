#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST="${MANIFEST:-splits/generated_seed4_mapdims_only.json}"
OUT_DIR="${OUT_DIR:-runs/rnn_seqmem_exp33_dreamer_v1}"
EPOCHS="${EPOCHS:-5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp33-dreamer-compatible-v1}"

mkdir -p "$OUT_DIR"

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "$RESUME")
elif [[ -f "$OUT_DIR/checkpoint_recovery.pt" ]]; then
  resume_args=(--resume "$OUT_DIR/checkpoint_recovery.pt")
elif [[ -f "$OUT_DIR/checkpoint.pt" ]]; then
  resume_args=(--resume "$OUT_DIR/checkpoint.pt")
fi

wandb_args=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")
if [[ "${WANDB:-1}" == "0" ]]; then
  wandb_args=(--no-wandb)
fi

# Do not set SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO here. That flag is only an
# evaluation ablation and would disable the learned hidden update.
env \
  LD_LIBRARY_PATH="" \
  SMAC_JEPA_ANCHOR_GATE_INIT="${SMAC_JEPA_ANCHOR_GATE_INIT:--3.0}" \
  SMAC_JEPA_ANCHOR_DELTA_SCALE="${SMAC_JEPA_ANCHOR_DELTA_SCALE:-0.25}" \
  SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE="${SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE:-0.10}" \
  SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT="${SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT:-0.002}" \
python -m smac_jepa.train_jepa_exp33_dreamer \
  --manifest "$MANIFEST" \
  --split train \
  --out-dir "$OUT_DIR" \
  --model-size default \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --rollout-window 20 \
  --rollout-horizon 5 \
  --window-mode random \
  --samples-per-epoch "$SAMPLES_PER_EPOCH" \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --action-conditioned-memory \
  --rollout-memory-dim 322 \
  --one-step-weight 0.5 \
  --target-mode full \
  --r2-dyn-scale 1.0 \
  --r2-rep-scale 0.1 \
  --r2-barlow-scale 0.05 \
  --r2-barlow-lambda 0.0005 \
  --r2-latent-normalize \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --ema-target-encoder \
  --ema-momentum 0.996 \
  --delta-loss-weight 0.05 \
  --enemy-observation-dropout 0.0 \
  --occlusion-mode contiguous \
  --contiguous-occlusion-spans 1 3 5 \
  --occlusion-spans-per-sample 2 \
  --hidden-reconstruction-weight 0.05 \
  --last-seen-anchor-weight 0.05 \
  --last-seen-change-threshold 0.01 \
  --hidden-presence-weight 0.02 \
  --reappearance-consistency-weight 0.02 \
  --inverse-dynamics-weight 0.0 \
  --memory-barlow-scale 0.0 \
  --event-dynamics-weight 0.0 \
  --no-event-balanced-sampling \
  --no-residual-state-decoder \
  --no-direct-action-fusion \
  --aux-loss-warmup-steps 2000 \
  --lr-warmup-steps 1000 \
  --grad-clip 1.0 \
  --checkpoint-every-steps 250 \
  --seed "$SEED" \
  --device cuda \
  --amp \
  "${wandb_args[@]}" \
  "${resume_args[@]}"

python scripts/validate_exp33_dreamer_checkpoint.py \
  "$OUT_DIR/checkpoint.pt" \
  --jepa-root .
