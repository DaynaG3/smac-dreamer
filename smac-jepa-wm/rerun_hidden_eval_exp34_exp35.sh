#!/usr/bin/env bash
set -uo pipefail

cd ~/workspace/dreamer/combined-upload
source .venv/bin/activate
cd smac-jepa-wm

MANIFEST="${MANIFEST:-splits/r2_general_2100.json}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

EXP34_CKPT="$(ls -td runs/rnn_seqmem_exp34_two_mask_complex_5ep_*/checkpoint.pt 2>/dev/null | head -1)"
EXP35_CKPT="$(ls -td runs/rnn_seqmem_exp35_two_mask_simple_loss_5ep_*/checkpoint.pt 2>/dev/null | head -1)"

find_probe_dir() {
  local ordinary_dir="$1"

  local probe_file
  probe_file="$(find "$ordinary_dir" -type f \
    \( -name "*probe*.pt" -o -name "*probe*.pth" -o -name "*decoder*.pt" -o -name "*.pt" \) \
    | head -1)"

  if [ -n "$probe_file" ]; then
    dirname "$probe_file"
  else
    echo "$ordinary_dir"
  fi
}

run_hidden() {
  local name="$1"
  local ckpt="$2"
  local ordinary_dir="eval_outputs/${name}/ordinary_rollout"
  local out_dir="eval_outputs/${name}/hidden_belief"
  local probe_dir

  if [ ! -s "$ckpt" ]; then
    echo "[skip] missing checkpoint for $name: $ckpt"
    return 0
  fi

  if [ ! -d "$ordinary_dir" ]; then
    echo "[skip] missing ordinary eval dir for $name: $ordinary_dir"
    return 0
  fi

  probe_dir="$(find_probe_dir "$ordinary_dir")"

  echo
  echo "======================================================================"
  echo "Hidden eval: $name"
  echo "checkpoint: $ckpt"
  echo "probe_dir:  $probe_dir"
  echo "out_dir:    $out_dir"
  echo "======================================================================"

  mkdir -p "$out_dir"

  python eval_jepa_hidden_belief_exp31_exp33.py \
    --checkpoint "$ckpt" \
    --manifest "$MANIFEST" \
    --split eval \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --out-dir "$out_dir" \
    --probe-dir "$probe_dir" \
    --eval-rollout-horizon 5 \
    --target-mode full \
    --thresholds 0.01 0.05 0.10 \
    --controlled-occlusion-eval \
    --controlled-occlusion-spans 1 3 5 \
    --natural-hidden-eval \
    2>&1 | tee "$out_dir/hidden_eval.log"
}

run_hidden "exp34_two_mask_complex_5ep" "$EXP34_CKPT"
run_hidden "exp35_two_mask_simple_loss_5ep" "$EXP35_CKPT"
