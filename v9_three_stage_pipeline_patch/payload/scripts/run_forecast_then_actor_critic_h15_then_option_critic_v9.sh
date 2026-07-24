#!/usr/bin/env bash
# Three independent experiments in one sequential GPU-safe pipeline:
# forecast JEPA -> ordinary Tactical-v1.2 AC H=15/800k -> Option-Critic v9 H=15/800k.
set -uo pipefail
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO") }"; ROOT="${ROOT% }"
PY="${PY:-$ROOT/.venv/bin/python}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
BUNDLE_ZIP="${BUNDLE_ZIP:-$ROOT/exp45_pow2_direct_from_exp40_bundle.zip}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
STRICT_EXIT="${STRICT_EXIT:-0}"
AUTO_TMUX="${AUTO_TMUX:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION_NAME:-forecast_ac15_ocv9_$STAMP}"
PIPE_DIR="${PIPE_DIR:-$ROOT/forecast_ac15_ocv9_$STAMP}"
LOG_DIR="$PIPE_DIR/logs"
STATUS="$PIPE_DIR/status.tsv"

if [[ "$AUTO_TMUX" == 1 && -z "${TMUX:-}" && "${INSIDE_PIPELINE_TMUX:-0}" != 1 ]]; then
  command -v tmux >/dev/null || { echo '[FAIL] tmux is not installed' >&2; exit 1; }
  mkdir -p "$PIPE_DIR"
  tmux new-session -d -s "$SESSION" \
    "INSIDE_PIPELINE_TMUX=1 AUTO_TMUX=0 PIPE_DIR='$PIPE_DIR' ROOT='$ROOT' REPO='$REPO' PY='$PY' JEPA_ROOT='$JEPA_ROOT' VENV='$VENV' BUNDLE_ZIP='$BUNDLE_ZIP' CONTINUE_ON_FAILURE='$CONTINUE_ON_FAILURE' STRICT_EXIT='$STRICT_EXIT' bash '$SELF'" || {
      echo "[FAIL] could not create tmux session: $SESSION" >&2; exit 1;
    }
  printf '%s\n' "$PIPE_DIR" > "$ROOT/CURRENT_FORECAST_AC15_OCV9_PIPELINE.txt"
  echo "[STARTED] tmux session: $SESSION"
  echo "[ATTACH] tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$LOG_DIR"
printf 'stage\tstatus\ttimestamp\texit\n' > "$STATUS"
FAILURES=0
run_stage() {
  local name="$1" log="$2"; shift 2
  printf '%s\tSTARTED\t%s\t-\n' "$name" "$(date -Is)" | tee -a "$STATUS"
  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if (( code == 0 )); then
    printf '%s\tCOMPLETED\t%s\t0\n' "$name" "$(date -Is)" | tee -a "$STATUS"
  else
    FAILURES=$((FAILURES + 1))
    printf '%s\tFAILED\t%s\t%s\n' "$name" "$(date -Is)" "$code" | tee -a "$STATUS"
  fi
  return "$code"
}
active_forecast() { pgrep -af 'train_jepa_exp45_pow2_direct.py|run_exp45_pow2_direct_train.sh' >/dev/null; }
active_rl() { pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; }
should_continue() { (( CONTINUE_ON_FAILURE == 1 || FAILURES == 0 )); }

run_stage forecast_jepa "$LOG_DIR/forecast.log" env \
  ROOT="$ROOT" JEPA_ROOT="$JEPA_ROOT" VENV="$VENV" BUNDLE_ZIP="$BUNDLE_ZIP" \
  PIPE_DIR="$PIPE_DIR/forecast" STRICT_EXIT=1 \
  bash "$REPO/scripts/run_exp45_full_train_eval_resilient.sh" || true

if ! should_continue; then
  printf 'actor_critic_h15_800k\tSKIPPED_PREVIOUS_FAILURE\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
elif active_forecast; then
  FAILURES=$((FAILURES + 1))
  printf 'actor_critic_h15_800k\tSKIPPED_ACTIVE_FORECAST_PROCESS\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
else
  run_stage actor_critic_h15_800k "$LOG_DIR/actor_critic_h15.log" env \
    ROOT="$ROOT" REPO="$REPO" PY="$PY" FINAL_STEP=800000 \
    bash "$REPO/scripts/run_actor_critic_h15_800k.sh" || true
fi

# The Option-Critic experiment starts independently from the same original
# Tactical-v1.2 checkpoint, not from the 800k baseline result. This keeps the
# architecture comparison fair.
if ! should_continue; then
  printf 'option_critic_v9_h15_800k\tSKIPPED_PREVIOUS_FAILURE\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
elif active_forecast || active_rl; then
  FAILURES=$((FAILURES + 1))
  printf 'option_critic_v9_h15_800k\tSKIPPED_ACTIVE_TRAIN_PROCESS\t%s\t-\n' "$(date -Is)" | tee -a "$STATUS"
  pgrep -af 'train_jepa_exp45_pow2_direct.py|run_exp45_pow2_direct_train.sh|train_r2dreamer_smaclite_multimap.py' \
    | tee "$LOG_DIR/active_processes_before_option_critic.log" || true
else
  run_stage option_critic_v9_h15_800k "$LOG_DIR/option_critic_v9.log" env \
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
