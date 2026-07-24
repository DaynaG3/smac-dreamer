#!/usr/bin/env bash
# Forecast JEPA first, then the controlled H=15, 800k interruptible option run.
set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO")}"
PY="${PY:-$ROOT/.venv/bin/python}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
BUNDLE_ZIP="${BUNDLE_ZIP:-$ROOT/exp45_pow2_direct_from_exp40_bundle.zip}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
STRICT_EXIT="${STRICT_EXIT:-0}"
AUTO_TMUX="${AUTO_TMUX:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION_NAME:-forecast_then_ocv9_$STAMP}"
PIPE_DIR="${PIPE_DIR:-$ROOT/forecast_then_option_critic_v9_$STAMP}"
LOG_DIR="$PIPE_DIR/logs"
STATUS="$PIPE_DIR/status.tsv"

if [[ "$AUTO_TMUX" == 1 && -z "${TMUX:-}" && "${INSIDE_PIPELINE_TMUX:-0}" != 1 ]]; then
  command -v tmux >/dev/null || { echo '[FAIL] tmux is not installed' >&2; exit 1; }
  mkdir -p "$PIPE_DIR"
  if ! tmux new-session -d -s "$SESSION" \
    "INSIDE_PIPELINE_TMUX=1 AUTO_TMUX=0 PIPE_DIR='$PIPE_DIR' ROOT='$ROOT' REPO='$REPO' PY='$PY' JEPA_ROOT='$JEPA_ROOT' VENV='$VENV' BUNDLE_ZIP='$BUNDLE_ZIP' CONTINUE_ON_FAILURE='$CONTINUE_ON_FAILURE' STRICT_EXIT='$STRICT_EXIT' bash '$SELF'"; then
    echo "[FAIL] could not create tmux session: $SESSION" >&2
    exit 1
  fi
  printf '%s\n' "$PIPE_DIR" > "$ROOT/CURRENT_FORECAST_THEN_OPTION_CRITIC_V9_PIPELINE.txt"
  echo "[STARTED] tmux session: $SESSION"
  echo "[ATTACH] tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$LOG_DIR"
printf 'stage\tstatus\ttimestamp\texit\n' > "$STATUS"
FAILURES=0
run_stage() {
  local name="$1" log="$2"
  shift 2
  printf '%s\tSTARTED\t%s\t-\n' "$name" "$(date -Is)" | tee -a "$STATUS"
  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if (( code != 0 )); then
    FAILURES=$((FAILURES + 1))
    printf '%s\tFAILED\t%s\t%s\n' "$name" "$(date -Is)" "$code" | tee -a "$STATUS"
  else
    printf '%s\tCOMPLETED\t%s\t0\n' "$name" "$(date -Is)" | tee -a "$STATUS"
  fi
  return "$code"
}

# STRICT_EXIT=1 makes internal forecast failures visible to this outer status,
# while `|| true` preserves the user's requirement that RL is still attempted.
run_stage forecast_jepa "$LOG_DIR/forecast.log" env \
  ROOT="$ROOT" JEPA_ROOT="$JEPA_ROOT" VENV="$VENV" BUNDLE_ZIP="$BUNDLE_ZIP" \
  PIPE_DIR="$PIPE_DIR/forecast" STRICT_EXIT=1 \
  bash "$REPO/scripts/run_exp45_full_train_eval_resilient.sh" || true

if (( FAILURES > 0 && CONTINUE_ON_FAILURE == 0 )); then
  printf 'rl_800k\tSKIPPED_FORECAST_FAILURE\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
elif pgrep -af 'train_jepa_exp45_pow2_direct.py|run_exp45_pow2_direct_train.sh' >/dev/null; then
  # Never start the RL learner on the same GPU while a failed forecast wrapper
  # has left a training child alive. Record and finish rather than hanging or
  # creating an OOM race; the RL launcher can be run manually after cleanup.
  FAILURES=$((FAILURES + 1))
  printf 'rl_800k\tSKIPPED_ACTIVE_FORECAST_PROCESS\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
  pgrep -af 'train_jepa_exp45_pow2_direct.py|run_exp45_pow2_direct_train.sh' \
    | tee "$LOG_DIR/active_forecast_processes.log" || true
else
  run_stage rl_800k "$LOG_DIR/rl.log" env \
    ROOT="$ROOT" REPO="$REPO" PY="$PY" FINAL_STEP=800000 \
    bash "$REPO/scripts/run_option_critic_v9_anchor_safe_800k.sh" || true
fi

state=COMPLETED
(( FAILURES > 0 )) && state=COMPLETED_WITH_FAILURES
printf 'pipeline\t%s\t%s\t%s\n' "$state" "$(date -Is)" "$FAILURES" | tee -a "$STATUS"
echo "[PIPELINE] $PIPE_DIR"
echo "[STATUS] $STATUS"
(( STRICT_EXIT == 1 && FAILURES > 0 )) && exit 1
exit 0
