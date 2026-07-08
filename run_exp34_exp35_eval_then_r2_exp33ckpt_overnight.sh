#!/usr/bin/env bash
set -Eeuo pipefail

# Overnight pipeline:
#   1) Train Exp34 JEPA for 5 epochs
#   2) Evaluate Exp34 JEPA
#   3) Train Exp35 JEPA for 5 epochs
#   4) Evaluate Exp35 JEPA
#   5) Run R2-Dreamer for 2,000,000 steps using the existing Exp33 JEPA checkpoint
#
# Important:
#   - This script DOES NOT overwrite smac-dreamer/checkpoints/jepa/model.pt.
#   - R2-Dreamer uses the original/previous Exp33-compatible checkpoint:
#       $DREAMER_ROOT/checkpoints/jepa/model.pt
#   - Exp34/35 are trained and evaluated as standalone JEPA checkpoints only.

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
JEPA_ROOT="${JEPA_ROOT:-$HOME/workspace/dreamer/combined-upload/smac-jepa-wm}"
DREAMER_ROOT="${DREAMER_ROOT:-$HOME/workspace/dreamer/combined-upload/smac-dreamer}"

# ---------------------------------------------------------------------
# JEPA training settings
# ---------------------------------------------------------------------
JEPA_EPOCHS="${JEPA_EPOCHS:-5}"
JEPA_WANDB="${JEPA_WANDB:-true}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP34_RUN="${EXP34_RUN:-runs/rnn_seqmem_exp34_two_mask_complex_5ep_${STAMP}}"
EXP35_RUN="${EXP35_RUN:-runs/rnn_seqmem_exp35_two_mask_simple_loss_5ep_${STAMP}}"

# ---------------------------------------------------------------------
# JEPA eval settings
# ---------------------------------------------------------------------
REQUIRE_JEPA_EVAL="${REQUIRE_JEPA_EVAL:-true}"

EVAL_TRAIN_EPISODE="${EVAL_TRAIN_EPISODE:-data/r2_general_2100_full/train/shard_02/r2g_train_0083.npz}"
EVAL_VAL_EPISODE="${EVAL_VAL_EPISODE:-data/r2_general_2100_full/validation/shard_03/r2g_validation_1300.npz}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_EPISODE_INDEX="${EVAL_EPISODE_INDEX:-0}"
EVAL_ROLLOUT_HORIZON="${EVAL_ROLLOUT_HORIZON:-5}"
EVAL_XY_TIMESTEPS="${EVAL_XY_TIMESTEPS:-1 2 5 10 20}"
EVAL_TABLE_TIMESTEPS="${EVAL_TABLE_TIMESTEPS:-1 2 5 10 20}"
EVAL_VALUE_TABLE_FEATURES="${EVAL_VALUE_TABLE_FEATURES:-important}"

# ---------------------------------------------------------------------
# R2-Dreamer settings
# ---------------------------------------------------------------------
R2_STEPS="${R2_STEPS:-2000000}"
DREAMER_CONFIG="${DREAMER_CONFIG:-configs/r2_2100_jepa_local.yaml}"
R2_JEPA_CKPT="${R2_JEPA_CKPT:-checkpoints/jepa/model.pt}"
R2_WANDB_MODE="${R2_WANDB_MODE:-disabled}"
R2_RUN="${R2_RUN:-logs/r2dreamer/exp33_jepa_original_ckpt_2m_${STAMP}}"

log_section() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

assert_file() {
  local f="$1"
  test -f "$f" || { echo "[missing] $f"; exit 1; }
}

assert_dir() {
  local d="$1"
  test -d "$d" || { echo "[missing dir] $d"; exit 1; }
}

verify_runner_supports_epochs() {
  local runner="$1"
  if ! grep -q "EPOCHS" "$runner"; then
    echo "[error] $runner does not appear to support EPOCHS env override."
    echo "Patch the runner or set --epochs manually. Required JEPA_EPOCHS=$JEPA_EPOCHS."
    exit 1
  fi
}

verify_epochs_from_config() {
  local run_dir="$1"
  local expected="$2"
  local cfg="$run_dir/config.json"

  if [ ! -f "$cfg" ]; then
    echo "[warn] Could not find $cfg to verify epochs. Continuing."
    return 0
  fi

  python - "$cfg" "$expected" <<'PY'
import json
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
expected = int(sys.argv[2])
data = json.loads(cfg.read_text())
actual = int(data.get("epochs", -1))
print(f"[check] {cfg}: epochs={actual}")
if actual != expected:
    raise SystemExit(f"[error] Expected epochs={expected}, got epochs={actual}")
PY
}

run_jepa_eval() {
  local ckpt="$1"
  local name="$2"
  local out_dir="sanity_outputs/${name}_decode_rollout"

  log_section "Evaluate ${name} JEPA checkpoint"

  cd "$JEPA_ROOT"
  assert_file "$ckpt"

  mkdir -p "$out_dir"

  if [ -f "tools/sanity_decode_rollout_exp33.py" ]; then
    echo "[eval] Using tools/sanity_decode_rollout_exp33.py"
    python tools/sanity_decode_rollout_exp33.py \
      --checkpoint "$ckpt" \
      --train-episode "$EVAL_TRAIN_EPISODE" \
      --val-episode "$EVAL_VAL_EPISODE" \
      --out-dir "$out_dir" \
      --device "$EVAL_DEVICE" \
      --episode-index "$EVAL_EPISODE_INDEX" \
      --rollout-horizon "$EVAL_ROLLOUT_HORIZON" \
      --xy-timesteps $EVAL_XY_TIMESTEPS \
      --table-timesteps $EVAL_TABLE_TIMESTEPS \
      --value-table-features "$EVAL_VALUE_TABLE_FEATURES" \
      2>&1 | tee "$out_dir/eval.log"
    return 0
  fi

  if [ -f "scripts/run_exp34_exp35_probe.sh" ]; then
    echo "[eval] tools/sanity_decode_rollout_exp33.py not found; using scripts/run_exp34_exp35_probe.sh"
    CKPT="$ckpt" \
    OUT_DIR="$out_dir" \
    bash scripts/run_exp34_exp35_probe.sh \
      2>&1 | tee "$out_dir/eval.log"
    return 0
  fi

  echo "[error] No JEPA eval/probe script found."
  echo "Checked:"
  echo "  $JEPA_ROOT/tools/sanity_decode_rollout_exp33.py"
  echo "  $JEPA_ROOT/scripts/run_exp34_exp35_probe.sh"
  echo
  echo "Port one of those scripts into this repo before running overnight."
  if [ "$REQUIRE_JEPA_EVAL" = "true" ]; then
    exit 1
  fi
}

log_section "0) Preflight"

assert_dir "$JEPA_ROOT"
assert_dir "$DREAMER_ROOT"

cd "$JEPA_ROOT"

assert_file "scripts/run_exp34_dreamer_two_mask.sh"
assert_file "scripts/run_exp35_dreamer_simple_loss.sh"
assert_file "smac_jepa/train_jepa_exp31_exp35.py"
assert_file "smac_jepa/train_jepa_exp34_dreamer.py"
assert_file "smac_jepa/train_jepa_exp35_dreamer.py"

python -m py_compile \
  smac_jepa/train_jepa_exp31_exp35.py \
  smac_jepa/train_jepa_exp34_dreamer.py \
  smac_jepa/train_jepa_exp35_dreamer.py

verify_runner_supports_epochs "scripts/run_exp34_dreamer_two_mask.sh"
verify_runner_supports_epochs "scripts/run_exp35_dreamer_simple_loss.sh"

if [ ! -f "tools/sanity_decode_rollout_exp33.py" ] && [ ! -f "scripts/run_exp34_exp35_probe.sh" ]; then
  echo "[error] JEPA eval script not found."
  echo "Need one of:"
  echo "  tools/sanity_decode_rollout_exp33.py"
  echo "  scripts/run_exp34_exp35_probe.sh"
  exit 1
fi

assert_file "$EVAL_TRAIN_EPISODE"
assert_file "$EVAL_VAL_EPISODE"

cd "$DREAMER_ROOT"
assert_file "$DREAMER_CONFIG"
assert_file "$R2_JEPA_CKPT"
assert_file "smac-dreamer/src/smacdreamer/jepa/world_model.py"
assert_file "validate_belief_mask_patch.py"
assert_file "scripts/train_r2dreamer_smaclite_multimap.py"

log_section "1) Train Exp34 JEPA, ${JEPA_EPOCHS} epochs"

cd "$JEPA_ROOT"
mkdir -p "$EXP34_RUN"

(
  cd "$JEPA_ROOT"
  OUT_DIR="$EXP34_RUN" \
  EPOCHS="$JEPA_EPOCHS" \
  WANDB="$JEPA_WANDB" \
  WANDB_NAME="exp34-two-mask-complex-5ep" \
  SMAC_JEPA_EXP34_TWO_MASK_LOSS=1 \
  SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT=3.0 \
  bash scripts/run_exp34_dreamer_two_mask.sh
) 2>&1 | tee "$JEPA_ROOT/$EXP34_RUN/overnight_exp34_train.log"

EXP34_CKPT="$JEPA_ROOT/$EXP34_RUN/checkpoint.pt"
assert_file "$EXP34_CKPT"
verify_epochs_from_config "$JEPA_ROOT/$EXP34_RUN" "$JEPA_EPOCHS"

run_jepa_eval "$EXP34_CKPT" "exp34_two_mask_complex_5ep"

log_section "2) Train Exp35 JEPA, ${JEPA_EPOCHS} epochs"

cd "$JEPA_ROOT"
mkdir -p "$EXP35_RUN"

(
  cd "$JEPA_ROOT"
  OUT_DIR="$EXP35_RUN" \
  EPOCHS="$JEPA_EPOCHS" \
  WANDB="$JEPA_WANDB" \
  WANDB_NAME="exp35-two-mask-simple-loss-5ep" \
  SMAC_JEPA_EXP34_TWO_MASK_LOSS=1 \
  SMAC_JEPA_EXP35_SIMPLE_LOSS=1 \
  SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT=3.0 \
  bash scripts/run_exp35_dreamer_simple_loss.sh
) 2>&1 | tee "$JEPA_ROOT/$EXP35_RUN/overnight_exp35_train.log"

EXP35_CKPT="$JEPA_ROOT/$EXP35_RUN/checkpoint.pt"
assert_file "$EXP35_CKPT"
verify_epochs_from_config "$JEPA_ROOT/$EXP35_RUN" "$JEPA_EPOCHS"

run_jepa_eval "$EXP35_CKPT" "exp35_two_mask_simple_loss_5ep"

log_section "3) R2-Dreamer preflight using original Exp33 JEPA checkpoint"

cd "$DREAMER_ROOT"

echo "[info] R2-Dreamer will use existing checkpoint:"
echo "  $DREAMER_ROOT/$R2_JEPA_CKPT"
echo "[info] This script does NOT copy Exp34/Exp35 over it."

python -m py_compile smac-dreamer/src/smacdreamer/jepa/world_model.py
python -m py_compile validate_belief_mask_patch.py
python validate_belief_mask_patch.py

log_section "4) Run R2-Dreamer for ${R2_STEPS} steps"

mkdir -p "$R2_RUN"

python scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$DREAMER_CONFIG" \
  --jepa-checkpoint "$R2_JEPA_CKPT" \
  --steps "$R2_STEPS" \
  --logdir "$R2_RUN" \
  --wandb-mode "$R2_WANDB_MODE" \
  2>&1 | tee "$R2_RUN/train.log"

log_section "DONE"

echo "Exp34 run: $JEPA_ROOT/$EXP34_RUN"
echo "Exp34 eval: $JEPA_ROOT/sanity_outputs/exp34_two_mask_complex_5ep_decode_rollout"
echo "Exp35 run: $JEPA_ROOT/$EXP35_RUN"
echo "Exp35 eval: $JEPA_ROOT/sanity_outputs/exp35_two_mask_simple_loss_5ep_decode_rollout"
echo "R2 run:    $DREAMER_ROOT/$R2_RUN"
echo "R2 JEPA checkpoint used: $DREAMER_ROOT/$R2_JEPA_CKPT"
