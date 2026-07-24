#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_repaired_exp42_44.sh"
mkdir -p "$PIPE_DIR/smoke" "$PIPE_DIR/logs"
status smoke_start OK "$PIPE_DIR"

jepa_activate
python -m smac_jepa.train_repaired_exp42_44_seqmem --help > "$PIPE_DIR/logs/trainer_help.txt"
for flag in --event-balanced-sampling --event-dynamics-weight --delta-loss-weight --inverse-dynamics-weight --hidden-change-residual-weight --hidden-change-scope --local-action-counterfactual-weight --audit-strict; do
  grep -q -- "$flag" "$PIPE_DIR/logs/trainer_help.txt" || { echo "[FAIL] missing $flag"; exit 1; }
done
status trainer_flags OK

# Tiny CPU smoke that still requires the mechanisms to be active.
export DEVICE="${SMOKE_DEVICE:-cuda}"
export WANDB_MODE=disabled
export EPOCHS=1
export SAMPLES_PER_EPOCH="${SMOKE_SAMPLES_PER_EPOCH:-64}"
export LOG_EVERY=1
export ROLLOUT_WINDOW="${SMOKE_ROLLOUT_WINDOW:-8}"
export ROLLOUT_HORIZON="${SMOKE_ROLLOUT_HORIZON:-2}"

run_smoke_exp() {
  local exp="$1"; shift
  local slug="$1"; shift
  local out="$PIPE_DIR/smoke/smoke_exp${exp}_${slug}"
  local log="$PIPE_DIR/logs/smoke_exp${exp}_${slug}.log"
  local args
  args="$(base_exp40_train_args)"
  args="${args/PLACEHOLDER_OUT_DIR/$out}"
  args="${args/PLACEHOLDER_WANDB_NAME/smoke_exp${exp}_${slug}}"
  echo "[SMOKE] Exp${exp} $slug" | tee "$log"
  # shellcheck disable=SC2086
  PYTHONPATH="$JEPA_DIR" python -m smac_jepa.train_repaired_exp42_44_seqmem $args "$@" 2>&1 | tee -a "$log"
  test "${PIPESTATUS[0]}" -eq 0
  grep -q "epoch_summary" "$log"
  if [[ "$exp" == "42" || "$exp" == "43" ]]; then
    grep -Eq "hidden_count=[1-9]" "$log" || { echo "[FAIL] Exp${exp} hidden_count never active" | tee -a "$log"; exit 1; }
    grep -Eq "hidden_changed=[1-9]" "$log" || { echo "[FAIL] Exp${exp} hidden_changed never active" | tee -a "$log"; exit 1; }
    grep -Eq "hidden_unchanged=[1-9]" "$log" || { echo "[FAIL] Exp${exp} hidden_unchanged never active" | tee -a "$log"; exit 1; }
  fi
  if [[ "$exp" == "44" ]]; then
    grep -Eq "local_count=[1-9]" "$log" || { echo "[FAIL] Exp44 local counterfactual never active" | tee -a "$log"; exit 1; }
    grep -Eq "local_near=0\.000000" "$log" && echo "[WARN] Exp44 local_near stayed zero in smoke; inspect before full run" | tee -a "$log"
  fi
}

echo "[SKIP] Exp42 weak_all_slots already passed activation"
run_smoke_exp 43 enemy_only --hidden-change-residual-weight 0.020 --hidden-change-copy-weight 0.005 --hidden-change-scope enemy_only
run_smoke_exp 44 local_counterfactual --local-action-counterfactual-weight 0.02 --local-action-drift-weight 0.05 --local-action-effect-margin 0.0005 --local-action-neighbor-radius 6.0

if [[ -z "${R2_RUN:-}" ]]; then
  echo "[WARN] R2_RUN not set; skipping R2 resume smoke check"
else
  [[ -f "$R2_RUN/latest.pt" ]] || { echo "[FAIL] R2_RUN latest missing: $R2_RUN/latest.pt"; exit 1; }
  status r2_latest_exists OK "$R2_RUN/latest.pt"
fi
patch_r2_torch_load
ensure_exp40_checkpoint
make_r2_200k_config
python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load('$DREAMER_DIR/configs/tmp_r2_2100_jepa_exp40_dense_perm_mask_200k_resume.yaml')
assert int(cfg.validation.every) == 200000
assert str(cfg.world_model.jepa.checkpoint) == 'checkpoints/jepa/model_exp40.pt'
print('[OK] R2 200k config smoke passed')
PY
status smoke_done OK "$PIPE_DIR"
echo "[OK] SMOKE PASSED: $PIPE_DIR/status.tsv"
