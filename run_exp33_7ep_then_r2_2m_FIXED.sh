#!/usr/bin/env bash
set -Eeuo pipefail

# Full guarded pipeline:
#   1. Sanity checks.
#   2. Train full Exp33 JEPA for 7 total epochs on the R2-2100 contract.
#   3. Strictly validate it with the patched Dreamer loader.
#   4. Install it as smac-dreamer/checkpoints/jepa/model.pt.
#   5. Run a 5,000-step R2-Dreamer smoke test.
#   6. Start the full 2,000,000-step R2-Dreamer run.
#
# Run this from tmux:
#   tmux new -s exp33-r2-2m
#   cd /home/jovyan/workspace/dreamer/combined-upload
#   ./run_exp33_7ep_then_r2_2m_FIXED.sh
#
# This script uses the existing UV-created .venv and does not touch uv.lock.

ROOT="${ROOT:-/home/jovyan/workspace/dreamer/combined-upload}"
JEPA_DIR="$ROOT/smac-jepa-wm"
DREAMER_DIR="$ROOT/smac-dreamer"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON="$VENV_DIR/bin/python"

JEPA_MANIFEST="${JEPA_MANIFEST:-splits/r2_general_2100.json}"
JEPA_OUT_DIR="${JEPA_OUT_DIR:-$JEPA_DIR/runs/rnn_seqmem_exp33_dreamer_7ep_v1}"
JEPA_CHECKPOINT="$JEPA_OUT_DIR/checkpoint.pt"

DREAMER_CHECKPOINT_DIR="$DREAMER_DIR/checkpoints/jepa"
DREAMER_CHECKPOINT="$DREAMER_CHECKPOINT_DIR/model.pt"
DREAMER_CONFIG="${DREAMER_CONFIG:-configs/r2_2100_jepa_local.yaml}"

SMOKE_STEPS="${SMOKE_STEPS:-5000}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SMOKE_LOGDIR="${SMOKE_LOGDIR:-$DREAMER_DIR/logs/r2dreamer/exp33_7ep_smoke_${TIMESTAMP}}"
FULL_LOGDIR="${FULL_LOGDIR:-$DREAMER_DIR/logs/r2dreamer/exp33_7ep_2m_${TIMESTAMP}}"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/exp33_7ep_to_r2_2m_${TIMESTAMP}.log}"

JEPA_WANDB="${JEPA_WANDB:-1}"
DREAMER_WANDB_MODE="${DREAMER_WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$(dirname "$PIPELINE_LOG")"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

trap 's=$?; echo; echo "FAILED at line $LINENO: $BASH_COMMAND"; echo "Log: $PIPELINE_LOG"; exit $s' ERR

echo "============================================================"
echo "Exp33 7 epochs -> R2-Dreamer 2M"
echo "ROOT:              $ROOT"
echo "JEPA manifest:     $JEPA_MANIFEST"
echo "JEPA output:       $JEPA_OUT_DIR"
echo "Installed ckpt:    $DREAMER_CHECKPOINT"
echo "Dreamer config:    $DREAMER_CONFIG"
echo "Full Dreamer logs: $FULL_LOGDIR"
echo "============================================================"

# ---------- Fail-fast checks ----------
[[ -x "$PYTHON" ]] || fail "Missing UV-created interpreter: $PYTHON"
[[ -d "$JEPA_DIR" ]] || fail "Missing $JEPA_DIR"
[[ -d "$DREAMER_DIR" ]] || fail "Missing $DREAMER_DIR"

required=(
  "$JEPA_DIR/$JEPA_MANIFEST"
  "$JEPA_DIR/smac_jepa/train_jepa_exp31_exp33.py"
  "$JEPA_DIR/smac_jepa/anchored_belief_memory.py"
  "$JEPA_DIR/smac_jepa/train_jepa_exp33_dreamer.py"
  "$JEPA_DIR/scripts/run_exp33_dreamer_pretrain.sh"
  "$JEPA_DIR/scripts/validate_exp33_dreamer_checkpoint.py"
  "$JEPA_DIR/tests/test_exp33_memory_contract.py"
  "$DREAMER_DIR/src/smacdreamer/jepa/checkpoint.py"
  "$DREAMER_DIR/src/smacdreamer/jepa/world_model.py"
  "$DREAMER_DIR/scripts/train_r2dreamer_smaclite_multimap.py"
  "$DREAMER_DIR/$DREAMER_CONFIG"
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || fail "Missing required file: $f"
done

export PATH="$VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$VENV_DIR"
export PYTHONUNBUFFERED=1
unset SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO || true

echo
echo "=== Python/CUDA/import check ==="
"$PYTHON" - <<'PY'
import torch
import smac_jepa
import smacdreamer

print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("smac_jepa:", smac_jepa.__file__)
print("smacdreamer:", smacdreamer.__file__)

assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda")
assert torch.isfinite(x @ x).all()
print("CUDA smoke test: PASS")
PY

grep -q "AnchoredActionConditionedEntityRolloutGRUMemory" \
  "$DREAMER_DIR/src/smacdreamer/jepa/checkpoint.py" \
  || fail "Dreamer anchored-memory loader patch is missing"

grep -q "presence_rollout_mode" \
  "$DREAMER_DIR/src/smacdreamer/jepa/world_model.py" \
  || fail "Dreamer soft-presence patch is missing"

"$PYTHON" -m py_compile \
  "$JEPA_DIR/smac_jepa/anchored_belief_memory.py" \
  "$JEPA_DIR/smac_jepa/train_jepa_exp33_dreamer.py" \
  "$JEPA_DIR/scripts/validate_exp33_dreamer_checkpoint.py" \
  "$DREAMER_DIR/src/smacdreamer/jepa/checkpoint.py" \
  "$DREAMER_DIR/src/smacdreamer/jepa/world_model.py"

(
  cd "$JEPA_DIR"
  "$PYTHON" tests/test_exp33_memory_contract.py
)

echo
echo "=== Verify Dreamer CLI before expensive JEPA training ==="
HELP="$("$PYTHON" "$DREAMER_DIR/scripts/train_r2dreamer_smaclite_multimap.py" --help 2>&1 || true)"
grep -q -- "--config" <<<"$HELP" || fail "Dreamer trainer lacks --config"
grep -q -- "--jepa-checkpoint" <<<"$HELP" || fail "Dreamer trainer lacks --jepa-checkpoint"
grep -q -- "--steps" <<<"$HELP" || fail "Dreamer trainer lacks --steps"
grep -q -- "--logdir" <<<"$HELP" || fail "Dreamer trainer lacks --logdir"
grep -q -- "--wandb-mode" <<<"$HELP" || fail "Dreamer trainer lacks --wandb-mode"
echo "Dreamer CLI check: PASS"

# ---------- JEPA training ----------
echo
echo "=== Train full Exp33 JEPA for 7 total epochs ==="
(
  cd "$JEPA_DIR"
  EPOCHS=7 \
  MANIFEST="$JEPA_MANIFEST" \
  OUT_DIR="$JEPA_OUT_DIR" \
  WANDB="$JEPA_WANDB" \
  WANDB_NAME="exp33-dreamer-compatible-7ep-v1" \
  NUM_WORKERS="$NUM_WORKERS" \
  BATCH_SIZE="$BATCH_SIZE" \
  ./scripts/run_exp33_dreamer_pretrain.sh
)

[[ -s "$JEPA_CHECKPOINT" ]] || fail "Missing final JEPA checkpoint: $JEPA_CHECKPOINT"

echo
echo "=== Validate final JEPA checkpoint ==="
"$PYTHON" "$JEPA_DIR/scripts/validate_exp33_dreamer_checkpoint.py" \
  "$JEPA_CHECKPOINT" \
  --jepa-root "$JEPA_DIR" \
  --dreamer-root "$DREAMER_DIR"

"$PYTHON" - "$JEPA_CHECKPOINT" <<'PY'
import sys
import torch

p = sys.argv[1]
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
except TypeError:
    c = torch.load(p, map_location="cpu")

cfg = c.get("resolved_config", c.get("config", {}))
state = c.get("memory_module_state", {})

assert c.get("model_state"), "model_state missing"
assert state, "memory_module_state missing"
assert bool(cfg.get("anchored_belief_memory", False)), "anchored flag missing"
assert any(str(k).startswith("hidden_gate_net.") for k in state), "hidden gate missing"
assert int(cfg.get("rollout_horizon", -1)) == 5, "rollout horizon is not 5"

saved_epoch = c.get("epoch")
epoch_complete = c.get("epoch_complete")
print("saved epoch:", saved_epoch)
print("epoch complete:", epoch_complete)
print("global step:", c.get("global_step"))
print("memory architecture:", cfg.get("memory_architecture"))
print("presence rollout:", cfg.get("presence_rollout_mode"))

if saved_epoch is not None and int(saved_epoch) < 7:
    raise SystemExit(f"Expected a 7-epoch checkpoint, got epoch={saved_epoch}")
if epoch_complete is False:
    raise SystemExit("Final checkpoint reports epoch_complete=False")
PY

# ---------- Install checkpoint ----------
echo
echo "=== Install checkpoint as smac-dreamer/checkpoints/jepa/model.pt ==="
mkdir -p "$DREAMER_CHECKPOINT_DIR"

if [[ -f "$DREAMER_CHECKPOINT" ]]; then
  cp -a "$DREAMER_CHECKPOINT" \
    "${DREAMER_CHECKPOINT}.backup_${TIMESTAMP}"
fi

tmp="${DREAMER_CHECKPOINT}.tmp.$$"
cp --reflink=auto "$JEPA_CHECKPOINT" "$tmp"
mv -f "$tmp" "$DREAMER_CHECKPOINT"
cmp -s "$JEPA_CHECKPOINT" "$DREAMER_CHECKPOINT" \
  || fail "Checkpoint copy verification failed"

"$PYTHON" "$JEPA_DIR/scripts/validate_exp33_dreamer_checkpoint.py" \
  "$DREAMER_CHECKPOINT" \
  --jepa-root "$JEPA_DIR" \
  --dreamer-root "$DREAMER_DIR"

sha256sum "$JEPA_CHECKPOINT" "$DREAMER_CHECKPOINT"

# ---------- Dreamer smoke ----------
if (( SMOKE_STEPS > 0 )); then
  echo
  echo "=== R2-Dreamer smoke test: ${SMOKE_STEPS} steps ==="
  (
    cd "$DREAMER_DIR"
    "$PYTHON" -u scripts/train_r2dreamer_smaclite_multimap.py \
      --config "$DREAMER_CONFIG" \
      --jepa-checkpoint "$DREAMER_CHECKPOINT" \
      --steps "$SMOKE_STEPS" \
      --logdir "$SMOKE_LOGDIR" \
      --wandb-mode disabled
  )
  [[ -s "$SMOKE_LOGDIR/latest.pt" ]] \
    || fail "Smoke run did not produce $SMOKE_LOGDIR/latest.pt"
  echo "Smoke test: PASS"
fi

# ---------- Full 2M Dreamer ----------
echo
echo "=== Start full R2-Dreamer run: 2,000,000 steps ==="
(
  cd "$DREAMER_DIR"
  "$PYTHON" -u scripts/train_r2dreamer_smaclite_multimap.py \
    --config "$DREAMER_CONFIG" \
    --jepa-checkpoint "$DREAMER_CHECKPOINT" \
    --steps 2000000 \
    --logdir "$FULL_LOGDIR" \
    --wandb-mode "$DREAMER_WANDB_MODE"
)

echo
echo "Pipeline finished."
echo "JEPA checkpoint:    $JEPA_CHECKPOINT"
echo "Installed checkpoint: $DREAMER_CHECKPOINT"
echo "Dreamer logs:       $FULL_LOGDIR"
echo "Pipeline log:       $PIPELINE_LOG"
