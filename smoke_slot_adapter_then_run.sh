#!/usr/bin/env bash
set -euo pipefail

# Smoke-test and optionally launch the slot-preserving JEPA feature-adapter R2-Dreamer run.
#
# Usage:
#   cd /home/jovyan/workspace/dreamer/combined-upload
#   bash smoke_slot_adapter_then_run.sh
#
# Start full run after smoke passes:
#   RUN_FULL=1 bash smoke_slot_adapter_then_run.sh
#
# Optional overrides:
#   SMOKE_STEPS=2000 RUN_FULL=1 bash smoke_slot_adapter_then_run.sh
#   WANDB_FULL_MODE=disabled RUN_FULL=1 bash smoke_slot_adapter_then_run.sh

COMBINED_ROOT="${COMBINED_ROOT:-/home/jovyan/workspace/dreamer/combined-upload}"
PROJECT_ROOT="${PROJECT_ROOT:-$COMBINED_ROOT/smac-dreamer}"

CONFIG="${CONFIG:-configs/r2_2100_jepa_slot_adapter.yaml}"
CKPT="${CKPT:-checkpoints/jepa/model.pt}"
SMOKE_STEPS="${SMOKE_STEPS:-1000}"
FULL_STEPS="${FULL_STEPS:-2000000}"
WANDB_FULL_MODE="${WANDB_FULL_MODE:-online}"

echo "=== Activate venv ==="
cd "$COMBINED_ROOT"
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: venv not found at $COMBINED_ROOT/.venv/bin/activate"
  exit 1
fi
source .venv/bin/activate

echo "python: $(which python)"
python - <<'PY'
import sys
import torch

print("python exe:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo
echo "=== Enter project ==="
cd "$PROJECT_ROOT"
echo "cwd: $(pwd)"

echo
echo "=== Check required files ==="
for f in \
  "$CONFIG" \
  "$CKPT" \
  "src/smacdreamer/jepa/feature_adapter.py" \
  "src/smacdreamer/jepa/world_model.py" \
  "scripts/train_r2dreamer_smaclite_multimap.py"
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file: $f"
    exit 1
  fi
done
echo "config: $CONFIG"
echo "checkpoint: $CKPT"
ls -lh "$CKPT"

echo
echo "=== Verify slot-adapter patch is active ==="
python - <<'PY'
from pathlib import Path

feature = Path("src/smacdreamer/jepa/feature_adapter.py").read_text()
world = Path("src/smacdreamer/jepa/world_model.py").read_text()

checks = [
    ("feature_adapter has max_agents argument", "max_agents" in feature),
    ("feature_adapter has slot_mlp", "slot_mlp" in feature),
    ("feature_adapter preserves ally slots", "allies_flat" in feature),
    ("world_model passes max_agents", "max_agents=self.max_agents" in world),
]

failed = False
for name, ok in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    failed = failed or (not ok)

idx = world.find("self.feature_adapter = JEPAFeatureAdapter(")
end = world.find("self.feat_size", idx)
if idx == -1 or end == -1:
    print("\nCould not locate feature_adapter construction block in world_model.py")
    failed = True
else:
    print("\nworld_model.py adapter block:")
    print(world[idx:end])

if failed:
    raise SystemExit("Patch verification failed.")
PY

echo
echo "=== Compile patched files ==="
python -m py_compile \
  src/smacdreamer/jepa/feature_adapter.py \
  src/smacdreamer/jepa/world_model.py \
  scripts/train_r2dreamer_smaclite_multimap.py

echo
echo "=== Load JEPA checkpoint ==="
python - <<'PY'
from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint

core, memory, info = load_frozen_jepa_checkpoint(
    "checkpoints/jepa/model.pt",
    map_location="cpu",
    strict=True,
)

print("memory class:", type(memory).__name__)
print("latent_dim:", info.latent_dim)
print("memory_dim:", info.memory_dim)
print("presence:", info.resolved_config.get("presence_rollout_mode"))
print("training_regime:", info.resolved_config.get("training_regime"))
PY

echo
echo "=== Direct adapter tensor smoke test ==="
python - <<'PY'
import torch
from smacdreamer.jepa.feature_adapter import JEPAFeatureAdapter

B = 4
E = 12
A = 5

adapter = JEPAFeatureAdapter(
    latent_dim=192,
    memory_dim=322,
    static_dim=16,
    out_dim=2304,
    max_agents=A,
)

z = torch.randn(B, E, 192)
mem = torch.randn(B, E, 322)
mask = torch.ones(B, E)
static = torch.randn(B, 16)

feat = adapter(z, mem, mask, static)

print("feat shape:", tuple(feat.shape))
print("finite:", bool(torch.isfinite(feat).all()))
print("adapter params:", sum(p.numel() for p in adapter.parameters()))

assert feat.shape == (B, 2304), tuple(feat.shape)
assert torch.isfinite(feat).all()
PY

echo
echo "=== Tiny R2-Dreamer smoke training ==="
SMOKE_RUN="logs/r2dreamer/smoke_slot_adapter_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE_RUN"

echo "smoke run: $SMOKE_RUN"
python -u scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --jepa-checkpoint "$CKPT" \
  --steps "$SMOKE_STEPS" \
  --logdir "$SMOKE_RUN" \
  --wandb-mode disabled \
  2>&1 | tee "$SMOKE_RUN/train.log"

echo
echo "=== Smoke log tail ==="
tail -n 40 "$SMOKE_RUN/train.log"

echo
echo "SMOKE PASSED."

if [[ "${RUN_FULL:-0}" != "1" ]]; then
  echo
  echo "Full run not started."
  echo "Start full overnight run with:"
  echo "  cd $COMBINED_ROOT"
  echo "  RUN_FULL=1 bash smoke_slot_adapter_then_run.sh"
  exit 0
fi

echo
echo "=== Start full overnight R2-Dreamer run ==="
FULL_RUN="logs/r2dreamer/exp34_slot_adapter_2m_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FULL_RUN"

nohup python -u scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --jepa-checkpoint "$CKPT" \
  --steps "$FULL_STEPS" \
  --logdir "$FULL_RUN" \
  --wandb-mode "$WANDB_FULL_MODE" \
  > "$FULL_RUN/train.log" 2>&1 &

echo $! > "$FULL_RUN/pid.txt"

echo "FULL_RUN=$FULL_RUN"
echo "PID=$(cat "$FULL_RUN/pid.txt")"
echo "LOG=$FULL_RUN/train.log"
echo
echo "Watch with:"
echo "  cd $PROJECT_ROOT"
echo "  tail -f $FULL_RUN/train.log"
