#!/usr/bin/env bash
# One-command sequential pipeline:
#   1) audited 1M-step Option-Critic P1-final run
#   2) Exp45 forecast-JEPA training + ordinary/hidden evaluation
# Top-level failures are recorded and do not prevent the next stage from being
# attempted when CONTINUE_ON_FAILURE=1 (default).
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO")}" 
PY="${PY:-$ROOT/.venv/bin/python}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
STRICT_EXIT="${STRICT_EXIT:-0}"
PIPE_STAMP="${PIPE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-oc1m_then_exp45_$PIPE_STAMP}"
PIPE_DIR="${MASTER_PIPE_DIR:-$ROOT/option_critic_1m_then_exp45_$PIPE_STAMP}"
LOG_DIR="$PIPE_DIR/logs"
STATUS_FILE="$PIPE_DIR/status.tsv"

# Make this genuinely one-command and SSH-safe. The script relaunches itself in
# a detached tmux session unless it is already inside tmux or AUTO_TMUX=0.
AUTO_TMUX="${AUTO_TMUX:-1}"
if [[ "$AUTO_TMUX" == "1" && -z "${TMUX:-}" && "${PIPELINE_INSIDE_TMUX:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || {
    echo "[FAIL] tmux is required for AUTO_TMUX=1" >&2
    exit 1
  }
  mkdir -p "$PIPE_DIR"
  tmux new-session -d -s "$SESSION_NAME" \
    "PIPELINE_INSIDE_TMUX=1 PIPE_STAMP='$PIPE_STAMP' SESSION_NAME='$SESSION_NAME' MASTER_PIPE_DIR='$PIPE_DIR' ROOT='$ROOT' REPO='$REPO' PY='$PY' CONTINUE_ON_FAILURE='$CONTINUE_ON_FAILURE' STRICT_EXIT='$STRICT_EXIT' bash '$SELF'"
  printf '%s\n' "$PIPE_DIR" > "$ROOT/CURRENT_OPTION_CRITIC_AND_EXP45_PIPELINE.txt"
  echo "[STARTED] tmux session: $SESSION_NAME"
  echo "[PIPELINE] $PIPE_DIR"
  echo "[ATTACH] tmux attach -t $SESSION_NAME"
  exit 0
fi

mkdir -p "$PIPE_DIR" "$LOG_DIR"
printf 'stage\tstatus\ttimestamp\tdetail\n' > "$STATUS_FILE"
printf '%s\n' "$PIPE_DIR" > "$ROOT/CURRENT_OPTION_CRITIC_AND_EXP45_PIPELINE.txt"

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

[[ -d "$REPO" ]] || {
  status "setup" "FAILED" "missing repo: $REPO"
  exit 1
}
[[ -x "$PY" ]] || {
  status "setup" "FAILED" "missing Python: $PY"
  exit 1
}
status "setup" "COMPLETED" "$REPO"

TACTICAL_V12_RUN="${TACTICAL_V12_RUN:-$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt" 2>/dev/null || true)}"
if [[ -z "$TACTICAL_V12_RUN" ]]; then
  status "rl_1m" "FAILED" "CURRENT_TACTICAL_V1_2_RUN.txt is missing"
  FAILURES=$((FAILURES + 1))
  RL_STATUS=2
else
  # Prevent stale launch exports from overriding the fail-closed launcher.
  unset SOURCE_CHECKPOINT SOURCE_RUN_META RUN_DIR CONFIG
  if run_stage "rl_1m" "$LOG_DIR/rl_1m.log" \
      env \
        ROOT="$ROOT" \
        REPO="$REPO" \
        PY="$PY" \
        TACTICAL_V12_RUN="$TACTICAL_V12_RUN" \
        FINAL_STEP=1000000 \
        "$REPO/scripts/run_option_critic_p1_final_1m.sh"; then
    RL_STATUS=0
  else
    RL_STATUS=$?
  fi
fi

if (( RL_STATUS != 0 && CONTINUE_ON_FAILURE != 1 )); then
  status "forecast_exp45" "SKIPPED" "RL failed and CONTINUE_ON_FAILURE=0"
else
  FORECAST_PIPE_DIR="$PIPE_DIR/forecast_exp45"
  # The resilient forecast script records install/audit/train/eval failures and
  # returns zero by default, so the master pipeline always reaches its summary.
  run_stage "forecast_exp45" "$LOG_DIR/forecast_exp45_wrapper.log" \
    env \
      ROOT="$ROOT" \
      JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}" \
      VENV="${VENV:-$ROOT/.venv}" \
      PIPE_DIR="$FORECAST_PIPE_DIR" \
      FORECAST_STAMP="$PIPE_STAMP" \
      STRICT_EXIT=1 \
      EVAL_PARTIAL_ON_TRAIN_FAILURE="${EVAL_PARTIAL_ON_TRAIN_FAILURE:-1}" \
      "$REPO/scripts/run_exp45_full_train_eval_resilient.sh" || true
fi

if (( FAILURES == 0 )); then
  status "pipeline" "COMPLETED" "$PIPE_DIR"
else
  status "pipeline" "COMPLETED_WITH_FAILURES" "count=$FAILURES"
fi

cat <<SUMMARY

================================================================
Option-Critic 1M -> Exp45 forecast-JEPA pipeline finished
================================================================
Pipeline:          $PIPE_DIR
Status:            $STATUS_FILE
Logs:              $LOG_DIR
RL pointer:         $ROOT/CURRENT_OPTION_CRITIC_V4_P1_1M_RUN.txt
Forecast pipeline: $PIPE_DIR/forecast_exp45
Recorded failures: $FAILURES
================================================================
SUMMARY

if [[ "$STRICT_EXIT" == "1" && "$FAILURES" -gt 0 ]]; then
  exit 1
fi
exit 0
