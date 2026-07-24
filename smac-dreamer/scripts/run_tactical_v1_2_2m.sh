#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
ROOT="${ROOT:-$(dirname "$REPO")}"
PY="${PY:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
FINAL_STEP="${FINAL_STEP:-2000000}"

if [[ -z "${ADAPTIVE_RUN:-}" ]]; then
  POINTER="$ROOT/CURRENT_UNIFIED_PRIORITY_RUN.txt"
  [[ -s "$POINTER" ]] || {
    echo "[FAIL] set ADAPTIVE_RUN or provide $POINTER" >&2
    exit 1
  }
  ADAPTIVE_RUN="$(cat "$POINTER")"
fi
# Deliberately do not read a generic CHECKPOINT environment variable: users
# often still have an older Exp40 path exported. Only SOURCE_CHECKPOINT can
# override the adaptive run's best validation checkpoint.
CHECKPOINT="${SOURCE_CHECKPOINT:-$ADAPTIVE_RUN/best_val_macro_winrate.pt}"
SOURCE_RUN_META="${SOURCE_RUN_META:-$ADAPTIVE_RUN/run_meta.json}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }
if [[ "$CONFIG" = /* ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="$REPO/$CONFIG"
fi
[[ -s "$CONFIG_PATH" ]] || {
  echo "[FAIL] v1.2 config missing: $CONFIG_PATH" >&2
  exit 1
}
[[ -s "$CHECKPOINT" ]] || {
  echo "[FAIL] adaptive-PER best checkpoint missing: $CHECKPOINT" >&2
  exit 1
}
ADAPTIVE_REAL="$(readlink -f "$ADAPTIVE_RUN")"
CHECKPOINT_PARENT="$(dirname "$(readlink -f "$CHECKPOINT")")"
if [[ "$CHECKPOINT_PARENT" != "$ADAPTIVE_REAL" ]]; then
  echo "[FAIL] source checkpoint is not inside the selected adaptive run" >&2
  echo "  adaptive run: $ADAPTIVE_REAL" >&2
  echo "  checkpoint : $(readlink -f "$CHECKPOINT")" >&2
  exit 1
fi
if [[ "$(basename "$CHECKPOINT")" != "best_val_macro_winrate.pt" ]]; then
  echo "[FAIL] Tactical v1.2 must start from best_val_macro_winrate.pt" >&2
  exit 1
fi
[[ -s "$SOURCE_RUN_META" ]] || {
  echo "[FAIL] source run metadata missing: $SOURCE_RUN_META" >&2
  exit 1
}
SOURCE_META_PARENT="$(dirname "$(readlink -f "$SOURCE_RUN_META")")"
if [[ "$SOURCE_META_PARENT" != "$ADAPTIVE_REAL" ]]; then
  echo "[FAIL] source run metadata is not inside the selected adaptive run" >&2
  echo "  adaptive run: $ADAPTIVE_REAL" >&2
  echo "  metadata    : $(readlink -f "$SOURCE_RUN_META")" >&2
  exit 1
fi
[[ "$FINAL_STEP" =~ ^[0-9]+$ ]] && (( FINAL_STEP > 0 )) || {
  echo "[FAIL] FINAL_STEP must be a positive integer" >&2
  exit 1
}

if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; then
  echo "[FAIL] another multimap trainer is already running:" >&2
  pgrep -af 'train_r2dreamer_smaclite_multimap.py' >&2 || true
  exit 1
fi

cd "$REPO"

"$PY" - "$CHECKPOINT" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt.get("agent_state_dict")
if not isinstance(state, dict):
    raise SystemExit("[FAIL] adaptive source lacks agent_state_dict")
if any(key.startswith("tactical_policy.") for key in state):
    raise SystemExit("[FAIL] source checkpoint already contains tactical parameters")
print("[SOURCE] stored step:", ckpt.get("step"))
print("[SOURCE] val macro win rate:", ckpt.get("val_macro_win_rate"))
print("[SOURCE] val macro original return:", ckpt.get("val_macro_original_return"))
print("[OK] selected adaptive-PER best non-tactical checkpoint")
PY

REPO="$REPO" \
PY="$PY" \
CONFIG="$CONFIG" \
CHECKPOINT="$CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
bash "$REPO/scripts/static_audit_tactical_v1_2.sh"

"$PY" "$REPO/scripts/audit_tactical_v1_2.py" \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO/logs/r2dreamer/adaptive_best_tactical_v1_2_2m_$STAMP}"
if [[ -e "$RUN_DIR" ]]; then
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "[FAIL] RUN_DIR exists but is not a directory: $RUN_DIR" >&2
    exit 1
  fi
  if find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "[FAIL] refusing to reuse non-empty RUN_DIR: $RUN_DIR" >&2
    echo "       This protects against stale TorchRL replay/memmap state." >&2
    exit 1
  fi
else
  mkdir -p "$RUN_DIR"
fi

CHECKPOINT_REAL="$(readlink -f "$CHECKPOINT")"
CHECKPOINT_SHA="$($PY - "$CHECKPOINT_REAL" <<'PY'
import hashlib
import sys
from pathlib import Path
path = Path(sys.argv[1])
h = hashlib.sha256()
with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)"

{
  echo "adaptive_run=$(readlink -f "$ADAPTIVE_RUN")"
  echo "source_checkpoint=$CHECKPOINT_REAL"
  echo "source_checkpoint_sha256=$CHECKPOINT_SHA"
  echo "source_run_meta=$(readlink -f "$SOURCE_RUN_META")"
  echo "config=$(readlink -f "$CONFIG_PATH")"
  echo "fresh_phase_start_step=0"
  echo "fresh_phase_final_step=$FINAL_STEP"
} | tee "$RUN_DIR/SOURCE_LINEAGE.txt"

printf '%s\n' "$RUN_DIR" | tee "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt"

echo "[START] Tactical Mixture v1.2 run: $RUN_DIR"
echo "[START] source adaptive checkpoint: $CHECKPOINT_REAL"
echo "[START] source checkpoint SHA-256: $CHECKPOINT_SHA"

"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$CHECKPOINT_REAL" \
  --resume-start-step 0 \
  --logdir "$RUN_DIR" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN_DIR/train.log"
