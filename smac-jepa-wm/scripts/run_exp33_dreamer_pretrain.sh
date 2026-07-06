#!/usr/bin/env bash
set -Eeuo pipefail

# Dreamer-compatible full Exp33 pretraining.
#
# Safety changes:
#   - Never resumes automatically.
#   - RESUME must be explicitly supplied.
#   - A clean run refuses to use a non-empty OUT_DIR.
#   - Prints the exact imported dataset file/version before training.
#
# This script uses the active UV-created virtual environment.
# It does not call uv sync, uv lock, uv add, or modify uv.lock.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MANIFEST="${MANIFEST:-splits/r2_general_2100.json}"
OUT_DIR="${OUT_DIR:-runs/rnn_seqmem_exp33_dreamer_7ep_v2_clean}"
EPOCHS="${EPOCHS:-7}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp33-dreamer-compatible-7ep-v2-clean}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$MANIFEST" ]] || fail "Manifest not found: $MANIFEST"

python - <<'PY'
import inspect
import sys
import torch
import smac_jepa

from smac_jepa import train_jepa_exp31_exp33 as trainer

dataset_cls = trainer.VisibilityMarkovRolloutSMACJEPADataset
version = int(
    getattr(dataset_cls, "explicit_visibility_mask_version", 0)
)

print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("smac_jepa:", smac_jepa.__file__)
print("visibility dataset:", inspect.getfile(dataset_cls))
print("visibility dataset version:", version)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if version < 1:
    raise SystemExit(
        "Corrected visibility dataset is not active"
    )

print("Exp33 launch diagnostics: PASS")
PY

resume_args=()

if [[ -n "${RESUME:-}" ]]; then
  [[ -f "$RESUME" ]] || fail \
    "Requested resume checkpoint does not exist: $RESUME"

  echo "Explicit resume requested:"
  echo "  $RESUME"
  resume_args=(--resume "$RESUME")

  mkdir -p "$OUT_DIR"
else
  if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    fail \
      "OUT_DIR is not empty: $OUT_DIR
Use a fresh OUT_DIR, move the old directory, or explicitly set:
RESUME=/absolute/path/to/checkpoint.pt"
  fi

  mkdir -p "$OUT_DIR"
  echo "Starting a clean training run:"
  echo "  $OUT_DIR"
fi

wandb_args=(
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-name "$WANDB_NAME"
)

if [[ "${WANDB:-1}" == "0" ]]; then
  wandb_args=(--no-wandb)
fi

unset SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO || true

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
  --lr-warmup-steps 2000 \
  --grad-clip 1.0 \
  --checkpoint-every-steps 250 \
  --seed "$SEED" \
  --device cuda \
  --amp \
  "${wandb_args[@]}" \
  "${resume_args[@]}"

[[ -s "$OUT_DIR/checkpoint.pt" ]] || fail \
  "Training exited without producing $OUT_DIR/checkpoint.pt"

python scripts/validate_exp33_dreamer_checkpoint.py \
  "$OUT_DIR/checkpoint.pt" \
  --jepa-root . \
  --dreamer-root ../smac-dreamer
