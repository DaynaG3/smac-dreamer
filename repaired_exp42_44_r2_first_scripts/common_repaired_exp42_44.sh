#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
JEPA_DIR="$ROOT/smac-jepa-wm"
DREAMER_DIR="$ROOT/smac-dreamer"
MANIFEST="${MANIFEST:-splits/r2_general_2100.json}"
DEVICE="${DEVICE:-cuda}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="${PIPE_DIR:-$ROOT/repaired_exp42_44_r2_first_$STAMP}"
mkdir -p "$PIPE_DIR/logs"

jepa_activate() {
  cd "$JEPA_DIR"
  source "$ROOT/.venv/bin/activate"
  export PYTHONPATH="$JEPA_DIR:${PYTHONPATH:-}"
}

dreamer_activate() {
  cd "$DREAMER_DIR"
  source "$ROOT/.venv/bin/activate"
}

status() {
  mkdir -p "$PIPE_DIR"
  printf "%s\t%s\t%s\t%s\n" "$(date +%F_%T)" "$1" "$2" "${3:-}" | tee -a "$PIPE_DIR/status.tsv"
}

patch_r2_torch_load() {
  dreamer_activate
  python - <<'PY'
from pathlib import Path
p = Path('scripts/train_r2dreamer_smaclite_multimap.py')
s = p.read_text()
s = s.replace('torch.load(args.resume, map_location=str(cfg.device))', 'torch.load(args.resume, map_location=str(cfg.device), weights_only=False)')
s = s.replace('torch.load(path, map_location="cpu")', 'torch.load(path, map_location="cpu", weights_only=False)')
p.write_text(s)
print('[OK] patched torch.load weights_only=False if needed')
PY
}

make_r2_200k_config() {
  dreamer_activate
  python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
candidates = [
    Path('configs/tmp_r2_2100_jepa_exp40_dense_perm_mask_100k.yaml'),
    Path('configs/r2_2100_jepa_slot_adapter_default_reward.yaml'),
]
src = next((p for p in candidates if p.exists()), None)
if src is None:
    raise SystemExit('No R2 config source found')
cfg = OmegaConf.load(src)
# Keep the known good Exp40 R2 setup, but force 200k validation cadence.
if 'world_model' not in cfg or 'jepa' not in cfg.world_model:
    raise SystemExit('Config does not look like JEPA R2 config')
cfg.world_model.jepa.checkpoint = 'checkpoints/jepa/model_exp40.pt'
cfg.steps = 2000000
cfg.validation.every = 200000
cfg.validation.run_at_start = False
cfg.mask_threshold = 0.05
if 'reward' in cfg:
    cfg.reward.name = 'dense_v3'
    cfg.reward.params = {}
if 'wandb' in cfg:
    cfg.wandb.project = 'smac-dreamer-jepa'
    cfg.wandb.run_name = 'exp40_jepa_dense_v3_perm_imagmask_200k_resume_2m'
out = Path('configs/tmp_r2_2100_jepa_exp40_dense_perm_mask_200k_resume.yaml')
OmegaConf.save(cfg, out)
print('[OK] wrote', out)
print('validation.every =', cfg.validation.every)
print('jepa =', cfg.world_model.jepa.checkpoint)
print('reward =', cfg.reward.name if 'reward' in cfg else '<missing>')
PY
}

ensure_exp40_checkpoint() {
  dreamer_activate
  mkdir -p checkpoints/jepa
  if [[ ! -f checkpoints/jepa/model_exp40.pt ]]; then
    SRC=$(ls -t "$ROOT"/smac-jepa-wm/runs/rnn_seqmem_exp40_event_balanced_5ep_*/checkpoint.pt 2>/dev/null | head -1 || true)
    if [[ -z "$SRC" ]]; then
      echo "[FAIL] checkpoints/jepa/model_exp40.pt missing and no Exp40 checkpoint found" >&2
      exit 1
    fi
    cp "$SRC" checkpoints/jepa/model_exp40.pt
  fi
  ls -lh checkpoints/jepa/model_exp40.pt
}

base_exp40_train_args() {
  local wandb_flag="--wandb"
  if [[ "${WANDB_MODE:-online}" == "disabled" ]]; then
    wandb_flag=""
  fi
  cat <<EOF
--manifest $MANIFEST --split train --model-size default --epochs ${EPOCHS:-5} --out-dir PLACEHOLDER_OUT_DIR --rollout-window ${ROLLOUT_WINDOW:-20} --rollout-horizon ${ROLLOUT_HORIZON:-5} --window-mode random --samples-per-epoch ${SAMPLES_PER_EPOCH:-50000} --enemy-visibility-mask --enemy-sight-range 9.0 --temporal-loss lambda --td-lambda 0.9 --sigreg-weight 0.005 --decoder-weight 0.005 --presence-weight 0.01 --one-step-weight 0.5 --target-mode full --action-conditioned-memory --event-balanced-sampling --event-fraction 0.50 --event-change-threshold 0.01 --event-dynamics-weight 1.25 --delta-loss-weight 0.06 --inverse-dynamics-weight 0.01 --device $DEVICE --amp --seed ${SEED:-1} --log-every ${LOG_EVERY:-100} --audit-strict $wandb_flag --wandb-project ${WANDB_PROJECT:-SMAC-JEPA-losses} --wandb-name PLACEHOLDER_WANDB_NAME --wandb-mode ${WANDB_MODE:-online}
EOF
}
