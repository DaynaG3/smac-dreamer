#!/usr/bin/env bash
# Run inside tmux. This script optionally installs Exp45, trains it fully,
# then runs ordinary and hidden-belief evaluation in sequence.
set -Eeuo pipefail

###############################################################################
# User-overridable settings
###############################################################################
ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
MANIFEST="${MANIFEST:-splits/r2_general_2100.json}"
EXP40_CHECKPOINT="${EXP40_CHECKPOINT:-runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt}"
BUNDLE_ZIP="${BUNDLE_ZIP:-$ROOT/exp45_pow2_direct_from_exp40_bundle.zip}"

# Training defaults. Override on the command line as environment variables.
EPOCHS="${EPOCHS:-5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp45-pow2-direct-1-2-4-8-16}"
AMP="${AMP:-1}"

# Exp45 loss/model defaults.
POW2_DIRECT_WEIGHT="${POW2_DIRECT_WEIGHT:-0.10}"
POW2_COMPOSITION_WEIGHT="${POW2_COMPOSITION_WEIGHT:-0.05}"
POW2_SHARED_HEAD_WEIGHT="${POW2_SHARED_HEAD_WEIGHT:-0.10}"
POW2_HIDDEN_DIM="${POW2_HIDDEN_DIM:-384}"
POW2_WARMUP_STEPS="${POW2_WARMUP_STEPS:-2000}"

# Evaluation defaults.
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-300}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-100}"
PROBE_SAMPLES="${PROBE_SAMPLES:-8000}"
DIRECT_MAX_BATCHES="${DIRECT_MAX_BATCHES:-300}"
NATURAL_HIDDEN_TARGETS="${NATURAL_HIDDEN_TARGETS:-3000}"
NATURAL_HIDDEN_SCAN_BATCHES="${NATURAL_HIDDEN_SCAN_BATCHES:-5000}"
CONTROLLED_MAX_BATCHES="${CONTROLLED_MAX_BATCHES:-150}"
CONTROLLED_TARGETS="${CONTROLLED_TARGETS:-10000}"

# Set RESUME to an Exp45 checkpoint to resume training. When resuming, also set
# RUN_DIR to the original Exp45 run directory so checkpoint files stay together.
RESUME="${RESUME:-}"

###############################################################################
# Paths and logging
###############################################################################
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="${PIPE_DIR:-$ROOT/exp45_pow2_full_pipeline_$STAMP}"
RUN_DIR="${RUN_DIR:-$JEPA_ROOT/runs/rnn_seqmem_exp45_pow2_direct_$STAMP}"
EVAL_ROOT="${EVAL_ROOT:-$PIPE_DIR/eval}"
ORDINARY_OUT_DIR="$EVAL_ROOT/ordinary"
HIDDEN_OUT_DIR="$EVAL_ROOT/hidden"
LOG_DIR="$PIPE_DIR/logs"
STATUS_FILE="$PIPE_DIR/status.tsv"

mkdir -p "$PIPE_DIR" "$LOG_DIR" "$ORDINARY_OUT_DIR" "$HIDDEN_OUT_DIR"
printf 'stage\tstatus\ttimestamp\tdetail\n' > "$STATUS_FILE"

now() { date '+%Y-%m-%d %H:%M:%S'; }
status() {
  local stage="$1" state="$2" detail="${3:-}"
  printf '%s\t%s\t%s\t%s\n' "$stage" "$state" "$(now)" "$detail" | tee -a "$STATUS_FILE"
}
fail() {
  status "pipeline" "FAILED" "$*"
  echo "ERROR: $*" >&2
  exit 1
}

CURRENT_STAGE="setup"
on_error() {
  local code=$?
  status "$CURRENT_STAGE" "FAILED" "exit=$code line=${BASH_LINENO[0]:-unknown}"
  echo >&2
  echo "[FAILED] Stage: $CURRENT_STAGE" >&2
  echo "[FAILED] Pipeline directory: $PIPE_DIR" >&2
  echo "[FAILED] Inspect logs in: $LOG_DIR" >&2
  exit "$code"
}
trap on_error ERR

###############################################################################
# Validation and optional installation
###############################################################################
[[ -d "$ROOT" ]] || fail "ROOT does not exist: $ROOT"
[[ -d "$JEPA_ROOT" ]] || fail "JEPA repo does not exist: $JEPA_ROOT"
[[ -f "$VENV/bin/activate" ]] || fail "Virtual environment not found: $VENV"

# Install the Exp45 bundle only when its trainer is not already present.
if [[ ! -f "$JEPA_ROOT/smac_jepa/train_jepa_exp45_pow2_direct.py" ]]; then
  CURRENT_STAGE="install"
  status "$CURRENT_STAGE" "STARTED" "Exp45 trainer is not installed"
  [[ -f "$BUNDLE_ZIP" ]] || fail "Bundle ZIP not found: $BUNDLE_ZIP"

  INSTALL_TMP="$PIPE_DIR/install_bundle"
  rm -rf "$INSTALL_TMP"
  mkdir -p "$INSTALL_TMP"
  unzip -q "$BUNDLE_ZIP" -d "$INSTALL_TMP"
  INSTALLER="$(find "$INSTALL_TMP" -type f -name install_exp45_pow2_direct.sh -print -quit)"
  [[ -n "$INSTALLER" ]] || fail "Installer not found inside $BUNDLE_ZIP"
  chmod +x "$INSTALLER"
  ROOT="$ROOT" "$INSTALLER" 2>&1 | tee "$LOG_DIR/install.log"
  status "$CURRENT_STAGE" "COMPLETED" "$JEPA_ROOT"
else
  status "install" "SKIPPED" "Exp45 trainer already installed"
fi

source "$VENV/bin/activate"
cd "$JEPA_ROOT"
export PYTHONPATH="$JEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$MANIFEST" ]] || fail "Manifest not found relative to JEPA root: $JEPA_ROOT/$MANIFEST"
[[ -f "$EXP40_CHECKPOINT" ]] || fail "Exp40 checkpoint not found relative to JEPA root: $JEPA_ROOT/$EXP40_CHECKPOINT"
[[ -x scripts/run_exp45_pow2_direct_train.sh ]] || fail "Missing training launcher"
[[ -x scripts/eval_exp45_pow2_ordinary.sh ]] || fail "Missing ordinary evaluator launcher"
[[ -x scripts/eval_exp45_pow2_hidden.sh ]] || fail "Missing hidden evaluator launcher"

# Record the exact launch configuration before starting a long run.
cat > "$PIPE_DIR/config.env" <<CONFIG
ROOT=$ROOT
JEPA_ROOT=$JEPA_ROOT
VENV=$VENV
MANIFEST=$MANIFEST
EXP40_CHECKPOINT=$EXP40_CHECKPOINT
RUN_DIR=$RUN_DIR
EVAL_ROOT=$EVAL_ROOT
EPOCHS=$EPOCHS
SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH
BATCH_SIZE=$BATCH_SIZE
NUM_WORKERS=$NUM_WORKERS
SEED=$SEED
DEVICE=$DEVICE
WANDB=$WANDB
WANDB_PROJECT=$WANDB_PROJECT
WANDB_NAME=$WANDB_NAME
AMP=$AMP
POW2_DIRECT_WEIGHT=$POW2_DIRECT_WEIGHT
POW2_COMPOSITION_WEIGHT=$POW2_COMPOSITION_WEIGHT
POW2_SHARED_HEAD_WEIGHT=$POW2_SHARED_HEAD_WEIGHT
POW2_HIDDEN_DIM=$POW2_HIDDEN_DIM
POW2_WARMUP_STEPS=$POW2_WARMUP_STEPS
RESUME=$RESUME
CONFIG

###############################################################################
# Static audit
###############################################################################
CURRENT_STAGE="static_audit"
status "$CURRENT_STAGE" "STARTED" ""
./scripts/static_audit_exp45_pow2.sh 2>&1 | tee "$LOG_DIR/static_audit.log"
status "$CURRENT_STAGE" "COMPLETED" ""

###############################################################################
# Full training
###############################################################################
CURRENT_STAGE="training"
status "$CURRENT_STAGE" "STARTED" "$RUN_DIR"

train_env=(
  "MANIFEST=$MANIFEST"
  "EXP40_CHECKPOINT=$EXP40_CHECKPOINT"
  "OUT_DIR=$RUN_DIR"
  "EPOCHS=$EPOCHS"
  "SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH"
  "BATCH_SIZE=$BATCH_SIZE"
  "NUM_WORKERS=$NUM_WORKERS"
  "SEED=$SEED"
  "DEVICE=$DEVICE"
  "WANDB=$WANDB"
  "WANDB_PROJECT=$WANDB_PROJECT"
  "WANDB_NAME=$WANDB_NAME"
  "AMP=$AMP"
  "POW2_DIRECT_WEIGHT=$POW2_DIRECT_WEIGHT"
  "POW2_COMPOSITION_WEIGHT=$POW2_COMPOSITION_WEIGHT"
  "POW2_SHARED_HEAD_WEIGHT=$POW2_SHARED_HEAD_WEIGHT"
  "POW2_HIDDEN_DIM=$POW2_HIDDEN_DIM"
  "POW2_WARMUP_STEPS=$POW2_WARMUP_STEPS"
)
if [[ -n "$RESUME" ]]; then
  [[ -f "$RESUME" ]] || fail "RESUME checkpoint not found: $RESUME"
  train_env+=("RESUME=$RESUME")
fi

env "${train_env[@]}" \
  ./scripts/run_exp45_pow2_direct_train.sh \
  2>&1 | tee "$LOG_DIR/train.log"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
[[ -s "$CHECKPOINT" ]] || fail "Training completed without checkpoint: $CHECKPOINT"
status "$CURRENT_STAGE" "COMPLETED" "$CHECKPOINT"

###############################################################################
# Ordinary evaluation: recursive H=5 + direct powers/exact/binary compositions
###############################################################################
CURRENT_STAGE="ordinary_eval"
status "$CURRENT_STAGE" "STARTED" "$ORDINARY_OUT_DIR"

env \
  CHECKPOINT="$CHECKPOINT" \
  MANIFEST="$MANIFEST" \
  SPLIT=eval \
  OUT_DIR="$ORDINARY_OUT_DIR" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  MAX_BATCHES="$EVAL_MAX_BATCHES" \
  PROBE_EPOCHS="$PROBE_EPOCHS" \
  PROBE_MAX_BATCHES="$PROBE_MAX_BATCHES" \
  PROBE_SAMPLES="$PROBE_SAMPLES" \
  ./scripts/eval_exp45_pow2_ordinary.sh \
  2>&1 | tee "$LOG_DIR/eval_ordinary.log"

status "$CURRENT_STAGE" "COMPLETED" "$ORDINARY_OUT_DIR"

###############################################################################
# Hidden evaluation: established recursive suite + direct natural-hidden metrics
###############################################################################
CURRENT_STAGE="hidden_eval"
status "$CURRENT_STAGE" "STARTED" "$HIDDEN_OUT_DIR"

env \
  CHECKPOINT="$CHECKPOINT" \
  MANIFEST="$MANIFEST" \
  SPLIT=eval \
  OUT_DIR="$HIDDEN_OUT_DIR" \
  ORDINARY_OUT_DIR="$ORDINARY_OUT_DIR" \
  DEVICE="$DEVICE" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  DIRECT_MAX_BATCHES="$DIRECT_MAX_BATCHES" \
  NATURAL_HIDDEN_TARGETS="$NATURAL_HIDDEN_TARGETS" \
  NATURAL_HIDDEN_SCAN_BATCHES="$NATURAL_HIDDEN_SCAN_BATCHES" \
  CONTROLLED_MAX_BATCHES="$CONTROLLED_MAX_BATCHES" \
  CONTROLLED_TARGETS="$CONTROLLED_TARGETS" \
  ./scripts/eval_exp45_pow2_hidden.sh \
  2>&1 | tee "$LOG_DIR/eval_hidden.log"

status "$CURRENT_STAGE" "COMPLETED" "$HIDDEN_OUT_DIR"

###############################################################################
# Finish
###############################################################################
CURRENT_STAGE="pipeline"
status "$CURRENT_STAGE" "COMPLETED" "$PIPE_DIR"

cat <<SUMMARY

============================================================
Exp45 full training + evaluation completed
============================================================
Pipeline:          $PIPE_DIR
Checkpoint:        $CHECKPOINT
Ordinary results:  $ORDINARY_OUT_DIR
Hidden results:    $HIDDEN_OUT_DIR
Logs:              $LOG_DIR
Status:            $STATUS_FILE
============================================================
SUMMARY
