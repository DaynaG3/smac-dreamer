#!/usr/bin/env bash
set -uo pipefail

# Overnight pipeline, continue-on-failure.
#
# Runs:
#   1) Exp34 JEPA pretrain, 5 epochs
#   2) Ordinary rollout metrics eval for Exp34
#   3) Hidden-belief metrics eval for Exp34
#   4) Exp35 JEPA pretrain, 5 epochs
#   5) Ordinary rollout metrics eval for Exp35
#   6) Hidden-belief metrics eval for Exp35
#   7) R2-Dreamer 2M using the ORIGINAL Exp33 JEPA checkpoint
#
# It does NOT overwrite:
#   ~/workspace/dreamer/combined-upload/smac-dreamer/checkpoints/jepa/model.pt
#
# Override evaluator paths if your filenames differ:
#   ORDINARY_EVAL=eval_jepa_exp31_exp33_anchored.py
#   HIDDEN_EVAL=eval_jepa_hidden_belief_exp31_exp33.py

JEPA_ROOT="${JEPA_ROOT:-$HOME/workspace/dreamer/combined-upload/smac-jepa-wm}"
DREAMER_ROOT="${DREAMER_ROOT:-$HOME/workspace/dreamer/combined-upload/smac-dreamer}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-$JEPA_ROOT/overnight_logs/exp34_exp35_true_metrics_r2_${STAMP}}"
mkdir -p "$PIPELINE_LOG_DIR"

STATUS_FILE="$PIPELINE_LOG_DIR/status.tsv"
: > "$STATUS_FILE"

# ----------------------------
# Train settings
# ----------------------------
JEPA_EPOCHS="${JEPA_EPOCHS:-5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
JEPA_WANDB="${JEPA_WANDB:-1}"

EXP34_RUN="${EXP34_RUN:-runs/rnn_seqmem_exp34_two_mask_complex_5ep_${STAMP}}"
EXP35_RUN="${EXP35_RUN:-runs/rnn_seqmem_exp35_two_mask_simple_loss_5ep_${STAMP}}"

# ----------------------------
# Evaluator selection
# ----------------------------
ORDINARY_EVAL="${ORDINARY_EVAL:-}"
HIDDEN_EVAL="${HIDDEN_EVAL:-}"

# Common eval settings
EVAL_MANIFEST="${EVAL_MANIFEST:-splits/r2_general_2100.json}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_ROLLOUT_HORIZON="${EVAL_ROLLOUT_HORIZON:-5}"

# Optional caps. Leave empty for full eval.
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
EVAL_PROBE_EPOCHS="${EVAL_PROBE_EPOCHS:-20}"
EVAL_PROBE_MAX_BATCHES_PER_EPOCH="${EVAL_PROBE_MAX_BATCHES_PER_EPOCH:-300}"
EVAL_PROBE_SAMPLES_PER_EPOCH="${EVAL_PROBE_SAMPLES_PER_EPOCH:-20000}"

HIDDEN_OCCLUSION_SPANS="${HIDDEN_OCCLUSION_SPANS:-1 3 5}"
HIDDEN_MAX_BATCHES="${HIDDEN_MAX_BATCHES:-}"
HIDDEN_PROBE_EPOCHS="${HIDDEN_PROBE_EPOCHS:-20}"

# ----------------------------
# R2-Dreamer settings
# ----------------------------
R2_STEPS="${R2_STEPS:-2000000}"
DREAMER_CONFIG="${DREAMER_CONFIG:-configs/r2_2100_jepa_local.yaml}"
R2_JEPA_CKPT="${R2_JEPA_CKPT:-checkpoints/jepa/model.pt}"
R2_WANDB_MODE="${R2_WANDB_MODE:-disabled}"
R2_RUN="${R2_RUN:-logs/r2dreamer/exp33_jepa_original_ckpt_2m_${STAMP}}"

section() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

record_status() {
  local name="$1"
  local status="$2"
  local note="$3"
  printf "%s\t%s\t%s\n" "$name" "$status" "$note" | tee -a "$STATUS_FILE"
}

run_stage() {
  local name="$1"
  local logfile="$2"
  shift 2

  section "$name"
  echo "[log] $logfile"
  mkdir -p "$(dirname "$logfile")"

  ( "$@" ) 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    record_status "$name" "OK" "rc=0"
  else
    record_status "$name" "FAIL" "rc=$rc; see $logfile"
  fi

  # Always continue to next stage.
  return 0
}

find_first_existing() {
  for p in "$@"; do
    if [ -f "$p" ]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

resolve_eval_files() {
  cd "$JEPA_ROOT" || return 1

  if [ -z "$ORDINARY_EVAL" ]; then
    ORDINARY_EVAL="$(find_first_existing \
      eval_jepa_exp31_exp33_anchored.py \
      eval_jepa_exp31_exp33.py \
      eval_jepa_blocker_fixed_experiments.py \
      smac_jepa/eval_jepa_exp31_exp33_anchored.py \
      smac_jepa/eval_jepa_exp31_exp33.py \
      smac_jepa/eval_jepa_blocker_fixed_experiments.py \
      || true)"
  fi

  if [ -z "$HIDDEN_EVAL" ]; then
    HIDDEN_EVAL="$(find_first_existing \
      eval_jepa_hidden_belief_exp31_exp33.py \
      eval_jepa_hidden_belief_v3.py \
      eval_jepa_event_conditioned_v2.py \
      eval_jepa_event_conditioned.py \
      smac_jepa/eval_jepa_hidden_belief_exp31_exp33.py \
      smac_jepa/eval_jepa_hidden_belief_v3.py \
      smac_jepa/eval_jepa_event_conditioned_v2.py \
      smac_jepa/eval_jepa_event_conditioned.py \
      || true)"
  fi

  echo "[eval] ORDINARY_EVAL=${ORDINARY_EVAL:-NOT_FOUND}"
  echo "[eval] HIDDEN_EVAL=${HIDDEN_EVAL:-NOT_FOUND}"

  return 0
}

module_or_file_cmd() {
  local path="$1"
  if [[ "$path" == smac_jepa/*.py ]]; then
    local mod="${path%.py}"
    mod="${mod//\//.}"
    echo "python -m $mod"
  else
    echo "python $path"
  fi
}

add_opt_if_supported() {
  local help_text="$1"
  local opt="$2"
  local val="${3:-__FLAG__}"
  local -n arr="$4"

  if grep -q -- "$opt" <<< "$help_text"; then
    arr+=("$opt")
    if [ "$val" != "__FLAG__" ]; then
      arr+=("$val")
    fi
  fi
}

run_eval_python_script() {
  local script_path="$1"
  local ckpt="$2"
  local out_dir="$3"
  local kind="$4"

  cd "$JEPA_ROOT" || return 1

  if [ ! -f "$script_path" ]; then
    echo "[missing eval script] $script_path"
    return 2
  fi
  if [ ! -s "$ckpt" ]; then
    echo "[missing checkpoint] $ckpt"
    return 2
  fi

  mkdir -p "$out_dir"

  local cmd_str
  cmd_str="$(module_or_file_cmd "$script_path")"

  # shellcheck disable=SC2206
  local cmd=( $cmd_str )

  local help_text
  help_text="$("${cmd[@]}" --help 2>&1 || true)"

  local args=()

  if grep -q -- "--checkpoint" <<< "$help_text"; then
    args+=(--checkpoint "$ckpt")
  elif grep -q -- "--checkpoints" <<< "$help_text"; then
    args+=(--checkpoints "$ckpt")
  else
    args+=(--checkpoint "$ckpt")
  fi

  add_opt_if_supported "$help_text" "--manifest" "$EVAL_MANIFEST" args
  add_opt_if_supported "$help_text" "--split" "$EVAL_SPLIT" args
  add_opt_if_supported "$help_text" "--device" "$EVAL_DEVICE" args
  add_opt_if_supported "$help_text" "--batch-size" "$EVAL_BATCH_SIZE" args
  add_opt_if_supported "$help_text" "--num-workers" "$EVAL_NUM_WORKERS" args
  add_opt_if_supported "$help_text" "--out-dir" "$out_dir" args
  add_opt_if_supported "$help_text" "--out" "$out_dir/eval_metrics.json" args
  add_opt_if_supported "$help_text" "--summary-out" "$out_dir/summary.csv" args

  add_opt_if_supported "$help_text" "--eval-rollout-horizon" "$EVAL_ROLLOUT_HORIZON" args
  add_opt_if_supported "$help_text" "--rollout-horizon" "$EVAL_ROLLOUT_HORIZON" args
  add_opt_if_supported "$help_text" "--target-mode" "full" args
  add_opt_if_supported "$help_text" "--window-mode" "sequential" args
  add_opt_if_supported "$help_text" "--enemy-visibility-mask" "__FLAG__" args

  # Ordinary metric extras.
  if [ "$kind" = "ordinary" ]; then
    add_opt_if_supported "$help_text" "--diagnostics" "__FLAG__" args
    add_opt_if_supported "$help_text" "--probe-decoder" "__FLAG__" args
    add_opt_if_supported "$help_text" "--probe-epochs" "$EVAL_PROBE_EPOCHS" args
    add_opt_if_supported "$help_text" "--probe-max-batches-per-epoch" "$EVAL_PROBE_MAX_BATCHES_PER_EPOCH" args
    add_opt_if_supported "$help_text" "--probe-samples-per-epoch" "$EVAL_PROBE_SAMPLES_PER_EPOCH" args

    if grep -q -- "--thresholds" <<< "$help_text"; then
      args+=(--thresholds 0.01 0.05 0.10)
    fi

    if [ -n "$EVAL_MAX_BATCHES" ]; then
      add_opt_if_supported "$help_text" "--max-batches" "$EVAL_MAX_BATCHES" args
    fi
  fi

  # Hidden-belief extras.
  if [ "$kind" = "hidden" ]; then
    add_opt_if_supported "$help_text" "--occlusion-mode" "contiguous" args

    if grep -q -- "--contiguous-occlusion-spans" <<< "$help_text"; then
      args+=(--contiguous-occlusion-spans $HIDDEN_OCCLUSION_SPANS)
    elif grep -q -- "--occlusion-spans" <<< "$help_text"; then
      args+=(--occlusion-spans $HIDDEN_OCCLUSION_SPANS)
    fi

    add_opt_if_supported "$help_text" "--hidden-probe-epochs" "$HIDDEN_PROBE_EPOCHS" args
    add_opt_if_supported "$help_text" "--probe-epochs" "$HIDDEN_PROBE_EPOCHS" args

    if [ -n "$HIDDEN_MAX_BATCHES" ]; then
      add_opt_if_supported "$help_text" "--max-batches" "$HIDDEN_MAX_BATCHES" args
    fi
  fi

  echo "[eval-cmd] ${cmd[*]} ${args[*]}"
  "${cmd[@]}" "${args[@]}"
}

verify_jepa_generated_files() {
  cd "$JEPA_ROOT" || return 1

  test -f scripts/run_exp34_dreamer_two_mask.sh || { echo "[missing] scripts/run_exp34_dreamer_two_mask.sh"; return 1; }
  test -f scripts/run_exp35_dreamer_simple_loss.sh || { echo "[missing] scripts/run_exp35_dreamer_simple_loss.sh"; return 1; }
  test -f smac_jepa/train_jepa_exp31_exp35.py || { echo "[missing] smac_jepa/train_jepa_exp31_exp35.py"; return 1; }
  test -f smac_jepa/train_jepa_exp34_dreamer.py || { echo "[missing] smac_jepa/train_jepa_exp34_dreamer.py"; return 1; }
  test -f smac_jepa/train_jepa_exp35_dreamer.py || { echo "[missing] smac_jepa/train_jepa_exp35_dreamer.py"; return 1; }

  python -m py_compile \
    smac_jepa/train_jepa_exp31_exp35.py \
    smac_jepa/train_jepa_exp34_dreamer.py \
    smac_jepa/train_jepa_exp35_dreamer.py || return 1

  resolve_eval_files
  return 0
}

train_exp34() {
  cd "$JEPA_ROOT" || return 1
  verify_jepa_generated_files || return 1

  OUT_DIR="$EXP34_RUN" \
  EPOCHS="$JEPA_EPOCHS" \
  SAMPLES_PER_EPOCH="$SAMPLES_PER_EPOCH" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  SEED="$SEED" \
  WANDB="$JEPA_WANDB" \
  WANDB_NAME="exp34-two-mask-complex-5ep" \
  SMAC_JEPA_EXP34_TWO_MASK_LOSS=1 \
  SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT=3.0 \
  bash scripts/run_exp34_dreamer_two_mask.sh || return 1

  test -s "$EXP34_RUN/checkpoint.pt" || { echo "[missing] $EXP34_RUN/checkpoint.pt"; return 1; }
}

train_exp35() {
  cd "$JEPA_ROOT" || return 1
  verify_jepa_generated_files || return 1

  OUT_DIR="$EXP35_RUN" \
  EPOCHS="$JEPA_EPOCHS" \
  SAMPLES_PER_EPOCH="$SAMPLES_PER_EPOCH" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  SEED="$SEED" \
  WANDB="$JEPA_WANDB" \
  WANDB_NAME="exp35-two-mask-simple-loss-5ep" \
  SMAC_JEPA_EXP34_TWO_MASK_LOSS=1 \
  SMAC_JEPA_EXP35_SIMPLE_LOSS=1 \
  SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT=3.0 \
  bash scripts/run_exp35_dreamer_simple_loss.sh || return 1

  test -s "$EXP35_RUN/checkpoint.pt" || { echo "[missing] $EXP35_RUN/checkpoint.pt"; return 1; }
}

eval_exp34_ordinary() {
  resolve_eval_files
  run_eval_python_script "$ORDINARY_EVAL" "$JEPA_ROOT/$EXP34_RUN/checkpoint.pt" \
    "$JEPA_ROOT/eval_outputs/exp34_two_mask_complex_5ep/ordinary_rollout" ordinary
}

eval_exp34_hidden() {
  resolve_eval_files
  run_eval_python_script "$HIDDEN_EVAL" "$JEPA_ROOT/$EXP34_RUN/checkpoint.pt" \
    "$JEPA_ROOT/eval_outputs/exp34_two_mask_complex_5ep/hidden_belief" hidden
}

eval_exp35_ordinary() {
  resolve_eval_files
  run_eval_python_script "$ORDINARY_EVAL" "$JEPA_ROOT/$EXP35_RUN/checkpoint.pt" \
    "$JEPA_ROOT/eval_outputs/exp35_two_mask_simple_loss_5ep/ordinary_rollout" ordinary
}

eval_exp35_hidden() {
  resolve_eval_files
  run_eval_python_script "$HIDDEN_EVAL" "$JEPA_ROOT/$EXP35_RUN/checkpoint.pt" \
    "$JEPA_ROOT/eval_outputs/exp35_two_mask_simple_loss_5ep/hidden_belief" hidden
}

r2_preflight() {
  cd "$DREAMER_ROOT" || return 1

  test -f "$DREAMER_CONFIG" || { echo "[missing] $DREAMER_CONFIG"; return 1; }
  test -f "$R2_JEPA_CKPT" || { echo "[missing] $R2_JEPA_CKPT"; return 1; }
  test -f smac-dreamer/src/smacdreamer/jepa/world_model.py || { echo "[missing] smac-dreamer/src/smacdreamer/jepa/world_model.py"; return 1; }
  test -f validate_belief_mask_patch.py || { echo "[missing] validate_belief_mask_patch.py"; return 1; }
  test -f scripts/train_r2dreamer_smaclite_multimap.py || { echo "[missing] scripts/train_r2dreamer_smaclite_multimap.py"; return 1; }

  echo "[info] R2-Dreamer uses original existing Exp33 checkpoint:"
  echo "  $DREAMER_ROOT/$R2_JEPA_CKPT"

  python -m py_compile smac-dreamer/src/smacdreamer/jepa/world_model.py || return 1
  python -m py_compile validate_belief_mask_patch.py || return 1
  python validate_belief_mask_patch.py || return 1
}

run_r2dreamer() {
  cd "$DREAMER_ROOT" || return 1
  r2_preflight || return 1

  mkdir -p "$R2_RUN"

  python scripts/train_r2dreamer_smaclite_multimap.py \
    --config "$DREAMER_CONFIG" \
    --jepa-checkpoint "$R2_JEPA_CKPT" \
    --steps "$R2_STEPS" \
    --logdir "$R2_RUN" \
    --wandb-mode "$R2_WANDB_MODE"
}

section "Pipeline settings"
cat <<EOF
JEPA_ROOT=$JEPA_ROOT
DREAMER_ROOT=$DREAMER_ROOT
PIPELINE_LOG_DIR=$PIPELINE_LOG_DIR
JEPA_EPOCHS=$JEPA_EPOCHS
SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH
EXP34_RUN=$EXP34_RUN
EXP35_RUN=$EXP35_RUN
ORDINARY_EVAL=${ORDINARY_EVAL:-auto}
HIDDEN_EVAL=${HIDDEN_EVAL:-auto}
R2_STEPS=$R2_STEPS
DREAMER_CONFIG=$DREAMER_CONFIG
R2_JEPA_CKPT=$R2_JEPA_CKPT
R2_RUN=$R2_RUN
EOF

run_stage "Preflight JEPA generated files and eval discovery" "$PIPELINE_LOG_DIR/preflight_jepa.log" verify_jepa_generated_files

run_stage "Train Exp34 JEPA" "$PIPELINE_LOG_DIR/train_exp34.log" train_exp34
run_stage "Ordinary rollout metrics Exp34" "$PIPELINE_LOG_DIR/eval_exp34_ordinary.log" eval_exp34_ordinary
run_stage "Hidden belief metrics Exp34" "$PIPELINE_LOG_DIR/eval_exp34_hidden.log" eval_exp34_hidden

run_stage "Train Exp35 JEPA" "$PIPELINE_LOG_DIR/train_exp35.log" train_exp35
run_stage "Ordinary rollout metrics Exp35" "$PIPELINE_LOG_DIR/eval_exp35_ordinary.log" eval_exp35_ordinary
run_stage "Hidden belief metrics Exp35" "$PIPELINE_LOG_DIR/eval_exp35_hidden.log" eval_exp35_hidden

run_stage "R2-Dreamer 2M with original Exp33 JEPA checkpoint" "$PIPELINE_LOG_DIR/r2dreamer_2m.log" run_r2dreamer

section "SUMMARY"
cat "$STATUS_FILE"

echo
echo "Logs: $PIPELINE_LOG_DIR"
echo "Exp34 run: $JEPA_ROOT/$EXP34_RUN"
echo "Exp34 ordinary eval: $JEPA_ROOT/eval_outputs/exp34_two_mask_complex_5ep/ordinary_rollout"
echo "Exp34 hidden eval:   $JEPA_ROOT/eval_outputs/exp34_two_mask_complex_5ep/hidden_belief"
echo "Exp35 run: $JEPA_ROOT/$EXP35_RUN"
echo "Exp35 ordinary eval: $JEPA_ROOT/eval_outputs/exp35_two_mask_simple_loss_5ep/ordinary_rollout"
echo "Exp35 hidden eval:   $JEPA_ROOT/eval_outputs/exp35_two_mask_simple_loss_5ep/hidden_belief"
echo "R2 run: $DREAMER_ROOT/$R2_RUN"
