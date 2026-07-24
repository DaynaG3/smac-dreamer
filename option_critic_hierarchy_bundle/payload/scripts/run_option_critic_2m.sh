#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
ROOT="${ROOT:-$(dirname "$REPO")}" 
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_option_critic_8_v2.yaml}"
FINAL_STEP="${FINAL_STEP:-2000000}"

if [[ -z "${TACTICAL_V12_RUN:-}" ]]; then
  POINTER="$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt"
  [[ -s "$POINTER" ]] || {
    echo "[FAIL] set TACTICAL_V12_RUN or provide $POINTER" >&2
    exit 1
  }
  TACTICAL_V12_RUN="$(cat "$POINTER")"
fi
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$TACTICAL_V12_RUN/best_val_macro_winrate.pt}"
SOURCE_RUN_META="${SOURCE_RUN_META:-$TACTICAL_V12_RUN/run_meta.json}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }
if [[ "$CONFIG" = /* ]]; then CONFIG_PATH="$CONFIG"; else CONFIG_PATH="$REPO/$CONFIG"; fi
[[ -s "$CONFIG_PATH" ]] || { echo "[FAIL] config missing: $CONFIG_PATH" >&2; exit 1; }
[[ -s "$SOURCE_CHECKPOINT" ]] || { echo "[FAIL] source checkpoint missing: $SOURCE_CHECKPOINT" >&2; exit 1; }
[[ -s "$SOURCE_RUN_META" ]] || { echo "[FAIL] source run_meta missing: $SOURCE_RUN_META" >&2; exit 1; }
[[ "$FINAL_STEP" =~ ^[0-9]+$ ]] && (( FINAL_STEP > 0 )) || {
  echo "[FAIL] FINAL_STEP must be a positive integer" >&2; exit 1;
}

SOURCE_RUN_REAL="$(readlink -f "$TACTICAL_V12_RUN")"
CHECKPOINT_REAL="$(readlink -f "$SOURCE_CHECKPOINT")"
META_REAL="$(readlink -f "$SOURCE_RUN_META")"
[[ "$(dirname "$CHECKPOINT_REAL")" == "$SOURCE_RUN_REAL" ]] || {
  echo "[FAIL] checkpoint is not inside selected v1.2 run" >&2; exit 1;
}
[[ "$(dirname "$META_REAL")" == "$SOURCE_RUN_REAL" ]] || {
  echo "[FAIL] run_meta is not inside selected v1.2 run" >&2; exit 1;
}
[[ "$(basename "$CHECKPOINT_REAL")" == "best_val_macro_winrate.pt" ]] || {
  echo "[FAIL] hierarchy must start from v1.2 best_val_macro_winrate.pt" >&2; exit 1;
}

if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; then
  echo "[FAIL] another multimap trainer is already running:" >&2
  pgrep -af 'train_r2dreamer_smaclite_multimap.py' >&2 || true
  exit 1
fi

cd "$REPO"

"$PY" - "$CHECKPOINT_REAL" <<'PY'
import sys, torch
p=sys.argv[1]
ckpt=torch.load(p,map_location='cpu',weights_only=False)
state=ckpt.get('agent_state_dict')
meta=ckpt.get('tactical_mixture_metadata') or {}
if not isinstance(state,dict): raise SystemExit('[FAIL] source lacks agent_state_dict')
if meta.get('architecture')!='tactical_mixture_v1_2':
    raise SystemExit(f"[FAIL] source architecture is {meta.get('architecture')!r}")
if int(meta.get('num_tactics',-1))!=2:
    raise SystemExit('[FAIL] source is not the two-mode v1.2 checkpoint')
if any(k.startswith('hierarchical_options.') for k in state):
    raise SystemExit('[FAIL] source already contains hierarchical option parameters')
win=float(ckpt.get('val_macro_win_rate',-1))
if win < 0.3749: raise SystemExit(f'[FAIL] source macro win rate {win} < 0.375')
print('[SOURCE] step:',ckpt.get('step'))
print('[SOURCE] macro win rate:',win)
print('[SOURCE] original return:',ckpt.get('val_macro_original_return'))
print('[OK] selected Tactical Mixture v1.2 best checkpoint')
PY

REPO="$REPO" PY="$PY" CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT_REAL" \
SOURCE_RUN_META="$META_REAL" bash scripts/static_audit_option_critic_hierarchy.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO/logs/r2dreamer/tactical_v12_option_critic_v2_8_3to20_2m_$STAMP}"
if [[ -e "$RUN_DIR" ]]; then
  [[ -d "$RUN_DIR" ]] || { echo "[FAIL] RUN_DIR is not a directory" >&2; exit 1; }
  if find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "[FAIL] refusing to reuse non-empty RUN_DIR: $RUN_DIR" >&2
    exit 1
  fi
else
  mkdir -p "$RUN_DIR"
fi

CHECKPOINT_SHA="$($PY - "$CHECKPOINT_REAL" <<'PY'
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
print(h.hexdigest())
PY
)"
{
  echo "source_v1_2_run=$SOURCE_RUN_REAL"
  echo "source_checkpoint=$CHECKPOINT_REAL"
  echo "source_checkpoint_sha256=$CHECKPOINT_SHA"
  echo "source_run_meta=$META_REAL"
  echo "config=$(readlink -f "$CONFIG_PATH")"
  echo "fresh_phase_start_step=0"
  echo "fresh_phase_final_step=$FINAL_STEP"
  echo "option_capacity=8"
  echo "duration_range=3-20"
  echo "learned_termination=true"
  echo "hierarchy_architecture=dreamer_option_critic_v2"
} | tee "$RUN_DIR/SOURCE_LINEAGE.txt"

printf '%s\n' "$RUN_DIR" | tee "$ROOT/CURRENT_OPTION_CRITIC_V2_RUN.txt"

echo "[START] Option-Critic run: $RUN_DIR"
echo "[START] source checkpoint: $CHECKPOINT_REAL"
echo "[START] source SHA-256: $CHECKPOINT_SHA"

"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT_REAL" \
  --resume-start-step 0 \
  --logdir "$RUN_DIR" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN_DIR/train.log"
