#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO")}"
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml}"
FINAL_STEP="${FINAL_STEP:-800000}"
TACTICAL_V12_RUN="${TACTICAL_V12_RUN:-$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")}"
CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "[FAIL] checkpoint missing: $CHECKPOINT" >&2; exit 1; }
[[ -s "$SOURCE_RUN_META" ]] || { echo "[FAIL] source metadata missing: $SOURCE_RUN_META" >&2; exit 1; }
ACTUAL_SOURCE_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_CHECKPOINT_SHA256" == "$EXPECTED_SOURCE_CHECKPOINT_SHA256" ]] || {
  echo "[FAIL] wrong Tactical-v1.2 source checkpoint hash" >&2
  echo "expected=$EXPECTED_SOURCE_CHECKPOINT_SHA256" >&2
  echo "actual=$ACTUAL_SOURCE_CHECKPOINT_SHA256" >&2
  exit 1
}
[[ "$FINAL_STEP" =~ ^[0-9]+$ ]] && (( FINAL_STEP == 800000 )) || {
  echo '[FAIL] v9 comparison run requires exactly 800000 new environment steps' >&2
  exit 1
}
if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; then
  echo '[FAIL] another multimap trainer is already active' >&2
  pgrep -af 'train_r2dreamer_smaclite_multimap.py' >&2 || true
  exit 1
fi

cd "$REPO"
REPO="$REPO" PY="$PY" CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
EXPECTED_SOURCE_CHECKPOINT_SHA256="$EXPECTED_SOURCE_CHECKPOINT_SHA256" \
bash scripts/static_audit_option_critic_v9_anchor_safe.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO/logs/r2dreamer/tactical_v12_option_critic_v9_anchor_safe_h15_800k_$STAMP}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$ROOT/CURRENT_OPTION_CRITIC_V9_ANCHOR_SAFE_8SLOT_800K_RUN.txt"
cp "$CONFIG" "$RUN_DIR/launch_config.yaml"
sha256sum "$CHECKPOINT" > "$RUN_DIR/source_checkpoint_sha256.txt"

cat > "$RUN_DIR/SOURCE_LINEAGE.txt" <<EOF
source_checkpoint=$CHECKPOINT
source_run_meta=$SOURCE_RUN_META
architecture=dreamer_option_critic_v9_anchor_safe_8slot
num_options=8
source_group_router=frozen_deterministic_tactical_v1_2
source_worker=frozen_exact_anchors
slot_identity_initialization=fixed_40pct_anchor_floor_plus_trainable_remaining_mass
child_slots=6_gradient_alive_zero_output_bounded_deltas
slot_anchor_floor=0.40
option_critic_consistency=decays_with_worker_warmup
worker_objective=child_transitions_only_single_pg_ramp
source_interrupt=enabled_every_state
bellman_target=exact_interruptible_group_restricted_smdp
min_duration=1
max_duration=4
learned_termination=disabled
imag_horizon=15
final_step=$FINAL_STEP
validation=startup_and_every_200k
world_model=frozen_controlled_comparison
EOF

echo "[START] Option-Critic v9 run: $RUN_DIR"
echo "[START] source checkpoint: $CHECKPOINT"
echo "[START] horizon: 15; steps: $FINAL_STEP; options: 8"
set -o pipefail
"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT" \
  --resume-start-step 0 \
  --logdir "$RUN_DIR" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN_DIR/train.log"
