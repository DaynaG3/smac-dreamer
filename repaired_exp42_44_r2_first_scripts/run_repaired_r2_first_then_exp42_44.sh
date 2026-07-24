#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_repaired_exp42_44.sh"
mkdir -p "$PIPE_DIR/logs"
status pipeline_start OK "$PIPE_DIR"

if [[ -z "${R2_RUN:-}" ]]; then
  echo "[FAIL] R2_RUN must be set to stopped Exp40 R2 run directory" >&2
  exit 1
fi
if [[ ! -f "$R2_RUN/latest.pt" ]]; then
  echo "[FAIL] R2 latest checkpoint missing: $R2_RUN/latest.pt" >&2
  exit 1
fi

# 1) R2-Dreamer resume first. This is intentionally at the front.
patch_r2_torch_load
ensure_exp40_checkpoint
make_r2_200k_config
status r2_resume_start RUN "$R2_RUN/latest.pt"
dreamer_activate
WANDB_PROJECT="${WANDB_PROJECT:-smac-dreamer-jepa}" \
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/tmp_r2_2100_jepa_exp40_dense_perm_mask_200k_resume.yaml \
  --jepa-checkpoint checkpoints/jepa/model_exp40.pt \
  --steps 2000000 \
  --logdir "$R2_RUN" \
  --resume "$R2_RUN/latest.pt" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT:-smac-dreamer-jepa}" \
  2>&1 | tee -a "$PIPE_DIR/logs/r2_resume_to_2m.log"
status r2_resume_done OK "$R2_RUN"

# 2) Run repaired Exp42-44 only after R2 is done.
export DEVICE="${JEPA_DEVICE:-cuda}"
export EPOCHS="${EPOCHS:-5}"
export SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
export LOG_EVERY="${LOG_EVERY:-100}"
jepa_activate

run_train() {
  local exp="$1"; shift
  local slug="$1"; shift
  local out="$JEPA_DIR/runs/rnn_seqmem_exp${exp}_${slug}_repaired_${STAMP}"
  local log="$PIPE_DIR/logs/train_exp${exp}_${slug}.log"
  local args
  args="$(base_exp40_train_args)"
  args="${args/PLACEHOLDER_OUT_DIR/$out}"
  args="${args/PLACEHOLDER_WANDB_NAME/exp${exp}_${slug}_repaired_${STAMP}}"
  status "exp${exp}_train_start" RUN "$out"
  echo "[TRAIN] Exp${exp} $slug out=$out" | tee "$log"
  # shellcheck disable=SC2086
  PYTHONPATH="$JEPA_DIR" python -m smac_jepa.train_repaired_exp42_44_seqmem $args "$@" 2>&1 | tee -a "$log"
  test "${PIPESTATUS[0]}" -eq 0
  status "exp${exp}_train_done" OK "$out/checkpoint.pt"
  echo "$out/checkpoint.pt" > "$PIPE_DIR/exp${exp}_checkpoint.txt"
}

run_train 42 weak_copy_update --hidden-change-residual-weight 0.015 --hidden-change-copy-weight 0.005 --hidden-change-scope all_slots
run_train 43 enemy_only_copy_update --hidden-change-residual-weight 0.020 --hidden-change-copy-weight 0.005 --hidden-change-scope enemy_only
run_train 44 local_action_counterfactual --local-action-counterfactual-weight 0.02 --local-action-drift-weight 0.05 --local-action-effect-margin 0.0005 --local-action-neighbor-radius 6.0

# 3) Ordinary evals with logs. Hidden eval is not in this critical path; run manually/full later.
run_ordinary_eval() {
  local exp="$1"
  local ckpt
  ckpt="$(cat "$PIPE_DIR/exp${exp}_checkpoint.txt")"
  local out="$JEPA_DIR/eval_outputs/exp${exp}_repaired_${STAMP}/ordinary_rollout"
  local log="$PIPE_DIR/logs/eval_ordinary_exp${exp}.log"
  local help="$PIPE_DIR/logs/eval_rnn_seqmem_dreamer_probe_help.txt"
  mkdir -p "$out"
  status "exp${exp}_ordinary_eval_start" RUN "$out"
  PYTHONPATH="$JEPA_DIR" python eval_rnn_seqmem_dreamer_probe.py --help > "$help" 2>&1 || true
  has_flag() { grep -Eq "(^|[[:space:],])$1([[:space:],=]|$)" "$help"; }
  CMD=(python eval_rnn_seqmem_dreamer_probe.py --checkpoint "$ckpt" --manifest "$MANIFEST" --split eval --device "$DEVICE")
  if has_flag --out-dir; then CMD+=(--out-dir "$out");
  elif has_flag --output-dir; then CMD+=(--output-dir "$out");
  elif has_flag --output; then CMD+=(--output "$out/eval_metrics.json");
  else echo "[FAIL] ordinary eval output flag not found" | tee -a "$log"; cat "$help" | tee -a "$log"; exit 1; fi
  if has_flag --eval-rollout-horizon; then CMD+=(--eval-rollout-horizon 5); fi
  if has_flag --target-mode; then CMD+=(--target-mode full); fi
  if has_flag --max-batches; then CMD+=(--max-batches "${EVAL_MAX_BATCHES:-300}"); fi
  if has_flag --probe-decoder; then CMD+=(--probe-decoder); fi
  if has_flag --probe-decoder-epochs; then CMD+=(--probe-decoder-epochs "${PROBE_EPOCHS:-5}"); fi
  if has_flag --probe-decoder-max-batches-per-epoch; then CMD+=(--probe-decoder-max-batches-per-epoch "${PROBE_MAX_BATCHES_PER_EPOCH:-100}"); fi
  echo "[CMD] PYTHONPATH=$JEPA_DIR ${CMD[*]}" | tee "$log"
  PYTHONPATH="$JEPA_DIR" "${CMD[@]}" 2>&1 | tee -a "$log"
  test "${PIPESTATUS[0]}" -eq 0
  status "exp${exp}_ordinary_eval_done" OK "$out"
}

run_ordinary_eval 42
run_ordinary_eval 43
run_ordinary_eval 44

status pipeline_done OK "$PIPE_DIR"
echo "[OK] pipeline done: $PIPE_DIR/status.tsv"
