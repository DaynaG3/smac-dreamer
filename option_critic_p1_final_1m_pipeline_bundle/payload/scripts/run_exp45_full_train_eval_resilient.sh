#!/usr/bin/env bash
# Resilient Exp45 forecast-JEPA pipeline derived from the user-provided
# run_exp45_full_train_eval_tmux.sh. Each stage records its own outcome; a
# failed stage does not terminate the wrapper or the outer RL->forecast job.
set -uo pipefail

ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
MANIFEST="${MANIFEST:-splits/r2_general_2100.json}"
EXP40_CHECKPOINT="${EXP40_CHECKPOINT:-runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt}"
BUNDLE_ZIP="${BUNDLE_ZIP:-$ROOT/exp45_pow2_direct_from_exp40_bundle.zip}"

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

POW2_DIRECT_WEIGHT="${POW2_DIRECT_WEIGHT:-0.10}"
POW2_COMPOSITION_WEIGHT="${POW2_COMPOSITION_WEIGHT:-0.05}"
POW2_SHARED_HEAD_WEIGHT="${POW2_SHARED_HEAD_WEIGHT:-0.10}"
POW2_HIDDEN_DIM="${POW2_HIDDEN_DIM:-384}"
POW2_WARMUP_STEPS="${POW2_WARMUP_STEPS:-2000}"

EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-300}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-100}"
PROBE_SAMPLES="${PROBE_SAMPLES:-8000}"
DIRECT_MAX_BATCHES="${DIRECT_MAX_BATCHES:-300}"
NATURAL_HIDDEN_TARGETS="${NATURAL_HIDDEN_TARGETS:-3000}"
NATURAL_HIDDEN_SCAN_BATCHES="${NATURAL_HIDDEN_SCAN_BATCHES:-5000}"
CONTROLLED_MAX_BATCHES="${CONTROLLED_MAX_BATCHES:-150}"
CONTROLLED_TARGETS="${CONTROLLED_TARGETS:-10000}"

RESUME="${RESUME:-}"
EVAL_PARTIAL_ON_TRAIN_FAILURE="${EVAL_PARTIAL_ON_TRAIN_FAILURE:-1}"
STRICT_EXIT="${STRICT_EXIT:-0}"

STAMP="${FORECAST_STAMP:-$(date +%Y%m%d_%H%M%S)}"
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

FAILURES=0
run_stage() {
  local stage="$1" log="$2"
  shift 2
  status "$stage" "STARTED" ""
  set +e
  "$@" 2>&1 | tee "$log"
  local code="${PIPESTATUS[0]}"
  set -u
  if (( code == 0 )); then
    status "$stage" "COMPLETED" "exit=0"
  else
    status "$stage" "FAILED" "exit=$code"
    FAILURES=$((FAILURES + 1))
  fi
  return "$code"
}

skip_stage() {
  status "$1" "SKIPPED" "$2"
}

setup_ok=1
for path in "$ROOT" "$JEPA_ROOT"; do
  if [[ ! -d "$path" ]]; then
    status "setup" "FAILED" "missing directory: $path"
    setup_ok=0
    FAILURES=$((FAILURES + 1))
  fi
done
if [[ ! -f "$VENV/bin/activate" ]]; then
  status "setup" "FAILED" "virtual environment missing: $VENV"
  setup_ok=0
  FAILURES=$((FAILURES + 1))
fi

if (( setup_ok == 0 )); then
  skip_stage "install" "setup failed"
  skip_stage "static_audit" "setup failed"
  skip_stage "training" "setup failed"
  skip_stage "ordinary_eval" "setup failed"
  skip_stage "hidden_eval" "setup failed"
else
  status "setup" "COMPLETED" "$JEPA_ROOT"
fi

install_exp45() {
  local install_tmp="$PIPE_DIR/install_bundle"
  [[ -f "$BUNDLE_ZIP" ]] || {
    echo "[FAIL] Exp45 bundle missing: $BUNDLE_ZIP" >&2
    return 2
  }
  rm -rf "$install_tmp"
  mkdir -p "$install_tmp"
  unzip -q "$BUNDLE_ZIP" -d "$install_tmp" || return $?
  local installer
  installer="$(find "$install_tmp" -type f -name install_exp45_pow2_direct.sh -print -quit)"
  [[ -n "$installer" ]] || {
    echo "[FAIL] install_exp45_pow2_direct.sh not found in bundle" >&2
    return 3
  }
  chmod +x "$installer"
  ROOT="$ROOT" "$installer"
}

forecast_ready="$setup_ok"
if (( setup_ok == 1 )); then
  if [[ -f "$JEPA_ROOT/smac_jepa/train_jepa_exp45_pow2_direct.py" ]]; then
    skip_stage "install" "Exp45 trainer already installed"
  else
    if ! run_stage "install" "$LOG_DIR/install.log" install_exp45; then
      forecast_ready=0
    fi
  fi
fi

if (( forecast_ready == 1 )); then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  cd "$JEPA_ROOT" || forecast_ready=0
  export PYTHONPATH="$JEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

if (( forecast_ready == 1 )); then
  missing=()
  [[ -f "$MANIFEST" ]] || missing+=("$JEPA_ROOT/$MANIFEST")
  [[ -f "$EXP40_CHECKPOINT" ]] || missing+=("$JEPA_ROOT/$EXP40_CHECKPOINT")
  [[ -x scripts/run_exp45_pow2_direct_train.sh ]] || missing+=("scripts/run_exp45_pow2_direct_train.sh")
  [[ -x scripts/eval_exp45_pow2_ordinary.sh ]] || missing+=("scripts/eval_exp45_pow2_ordinary.sh")
  [[ -x scripts/eval_exp45_pow2_hidden.sh ]] || missing+=("scripts/eval_exp45_pow2_hidden.sh")
  if (( ${#missing[@]} > 0 )); then
    status "prerequisites" "FAILED" "missing: ${missing[*]}"
    FAILURES=$((FAILURES + 1))
    forecast_ready=0
  else
    status "prerequisites" "COMPLETED" ""
  fi
fi

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
EVAL_PARTIAL_ON_TRAIN_FAILURE=$EVAL_PARTIAL_ON_TRAIN_FAILURE
CONFIG

if (( forecast_ready == 1 )); then
  if ! run_stage "static_audit" "$LOG_DIR/static_audit.log" \
      ./scripts/static_audit_exp45_pow2.sh; then
    # Fail closed: do not launch forecast training against code that failed its
    # own static audit. The outer RL->forecast wrapper still continues to its
    # summary instead of dying.
    forecast_ready=0
  fi
else
  skip_stage "static_audit" "forecast prerequisites unavailable"
fi

CHECKPOINT="$RUN_DIR/checkpoint.pt"
training_ok=0
if (( forecast_ready == 1 )); then
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
    if [[ -f "$RESUME" ]]; then
      train_env+=("RESUME=$RESUME")
    else
      status "training" "FAILED" "RESUME checkpoint missing: $RESUME"
      FAILURES=$((FAILURES + 1))
      forecast_ready=0
    fi
  fi
  if (( forecast_ready == 1 )); then
    if run_stage "training" "$LOG_DIR/train.log" \
        env "${train_env[@]}" ./scripts/run_exp45_pow2_direct_train.sh; then
      training_ok=1
    fi
  fi
else
  skip_stage "training" "static audit or prerequisites failed"
fi

checkpoint_available=0
if [[ -s "$CHECKPOINT" ]]; then
  checkpoint_available=1
  status "checkpoint" "AVAILABLE" "$CHECKPOINT"
elif (( training_ok == 1 )); then
  status "checkpoint" "FAILED" "training returned 0 but checkpoint missing"
  FAILURES=$((FAILURES + 1))
fi

allow_eval=0
if (( checkpoint_available == 1 && training_ok == 1 )); then
  allow_eval=1
elif (( checkpoint_available == 1 && EVAL_PARTIAL_ON_TRAIN_FAILURE == 1 )); then
  allow_eval=1
  status "checkpoint" "PARTIAL" "evaluating checkpoint after training failure"
fi

if (( allow_eval == 1 )); then
  run_stage "ordinary_eval" "$LOG_DIR/eval_ordinary.log" \
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
      ./scripts/eval_exp45_pow2_ordinary.sh || true

  # Hidden evaluation is attempted even if ordinary evaluation failed. It may
  # still produce useful standalone metrics; any dependency failure is logged.
  run_stage "hidden_eval" "$LOG_DIR/eval_hidden.log" \
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
      ./scripts/eval_exp45_pow2_hidden.sh || true
else
  skip_stage "ordinary_eval" "no usable checkpoint"
  skip_stage "hidden_eval" "no usable checkpoint"
fi

if (( FAILURES == 0 )); then
  status "pipeline" "COMPLETED" "$PIPE_DIR"
else
  status "pipeline" "COMPLETED_WITH_FAILURES" "count=$FAILURES"
fi

cat <<SUMMARY

============================================================
Exp45 resilient training + evaluation pipeline finished
============================================================
Pipeline:          $PIPE_DIR
Checkpoint:        $CHECKPOINT
Ordinary results:  $ORDINARY_OUT_DIR
Hidden results:    $HIDDEN_OUT_DIR
Logs:              $LOG_DIR
Status:            $STATUS_FILE
Recorded failures: $FAILURES
============================================================
SUMMARY

if [[ "$STRICT_EXIT" == "1" && "$FAILURES" -gt 0 ]]; then
  exit 1
fi
exit 0
