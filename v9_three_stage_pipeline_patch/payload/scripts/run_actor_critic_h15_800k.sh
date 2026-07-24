#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO") }"
ROOT="${ROOT% }"
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
FINAL_STEP="${FINAL_STEP:-800000}"
TACTICAL_V12_RUN="${TACTICAL_V12_RUN:-$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt") }"
TACTICAL_V12_RUN="${TACTICAL_V12_RUN% }"
CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "[FAIL] source checkpoint missing: $CHECKPOINT" >&2; exit 1; }
[[ -s "$SOURCE_RUN_META" ]] || { echo "[FAIL] source metadata missing: $SOURCE_RUN_META" >&2; exit 1; }
[[ "$FINAL_STEP" =~ ^[0-9]+$ ]] && (( FINAL_STEP == 800000 )) || {
  echo "[FAIL] comparison baseline requires exactly 800000 new environment steps" >&2
  exit 1
}
ACTUAL_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_SHA" == "$EXPECTED_SOURCE_CHECKPOINT_SHA256" ]] || {
  echo "[FAIL] wrong Tactical-v1.2 source checkpoint hash" >&2
  echo "expected=$EXPECTED_SOURCE_CHECKPOINT_SHA256" >&2
  echo "actual=$ACTUAL_SHA" >&2
  exit 1
}
if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; then
  echo "[FAIL] another multimap trainer is already active" >&2
  pgrep -af 'train_r2dreamer_smaclite_multimap.py' >&2 || true
  exit 1
fi

cd "$REPO"
REPO="$REPO" PY="$PY" CONFIG="$CONFIG" SOURCE_CONFIG="$SOURCE_CONFIG" \
CHECKPOINT="$CHECKPOINT" EXPECTED_SOURCE_CHECKPOINT_SHA256="$EXPECTED_SOURCE_CHECKPOINT_SHA256" \
bash scripts/static_audit_actor_critic_h15_800k.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO/logs/r2dreamer/tactical_v12_actor_critic_h15_800k_$STAMP}"
[[ ! -e "$RUN_DIR" ]] || { echo "[FAIL] run directory already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$ROOT/CURRENT_TACTICAL_V12_ACTOR_CRITIC_H15_800K_RUN.txt"
cp "$CONFIG" "$RUN_DIR/launch_config.yaml"
sha256sum "$CHECKPOINT" > "$RUN_DIR/source_checkpoint_sha256.txt"
cat > "$RUN_DIR/SOURCE_LINEAGE.txt" <<EOF
source_checkpoint=$CHECKPOINT
source_run_meta=$SOURCE_RUN_META
architecture=tactical_mixture_v1_2_ordinary_actor_critic
option_critic=disabled
imag_horizon=15
new_environment_steps=$FINAL_STEP
validation=startup_and_every_200k
comparison_source=identical_to_option_critic_v9
EOF

echo "[START] ordinary Tactical-v1.2 actor-critic baseline: $RUN_DIR"
echo "[START] horizon: 15; new steps: $FINAL_STEP"
set -o pipefail
"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT" \
  --resume-start-step 0 \
  --logdir "$RUN_DIR" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN_DIR/train.log"
