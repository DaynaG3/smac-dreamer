#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JEPA_ROOT="${JEPA_ROOT:-$HOME/workspace/dreamer/combined-upload/smac-jepa-wm}"
CHECKPOINT="${CHECKPOINT:-$JEPA_ROOT/runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt}"
MANIFEST="${MANIFEST:-$JEPA_ROOT/splits/r2_general_2100.json}"
SPLIT="${SPLIT:-eval}"
MAX_BATCHES="${MAX_BATCHES:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOP_K="${TOP_K:-5}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cuda}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$JEPA_ROOT/analysis_outputs/exp40_h15_gallery_$STAMP}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/run.log}"

if [[ ! -d "$JEPA_ROOT" ]]; then
  echo "ERROR: JEPA repository not found: $JEPA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  FALLBACK="$HOME/workspace/dreamer/combined-upload/smac-dreamer/checkpoints/jepa/model_exp40.pt"
  if [[ -f "$FALLBACK" ]]; then
    echo "[warning] Primary checkpoint missing; using R2-installed Exp40 copy: $FALLBACK"
    CHECKPOINT="$FALLBACK"
  else
    echo "ERROR: Exp40 checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: evaluation manifest not found: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$JEPA_ROOT/eval_jepa_exp31_exp33_anchored.py" ]]; then
  echo "ERROR: anchored evaluator missing: $JEPA_ROOT/eval_jepa_exp31_exp33_anchored.py" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$JEPA_ROOT"

if [[ -x "$JEPA_ROOT/.venv/bin/python" ]]; then
  PYTHON=("$JEPA_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  PYTHON=(python)
fi

printf '[repo]       %s\n' "$JEPA_ROOT"
printf '[checkpoint] %s\n' "$CHECKPOINT"
printf '[manifest]   %s (%s)\n' "$MANIFEST" "$SPLIT"
printf '[output]     %s\n' "$OUT_DIR"
printf '[scale]      max_batches=%s batch_size=%s rollout_starts_per_item=20 horizon=15\n' "$MAX_BATCHES" "$BATCH_SIZE"
printf '[python]     %q ' "${PYTHON[@]}"; echo

env LD_LIBRARY_PATH="" \
    PYTHONPATH="$JEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYTHON[@]}" "$SCRIPT_DIR/eval_exp40_rollout_gallery.py" \
      --checkpoint "$CHECKPOINT" \
      --manifest "$MANIFEST" \
      --split "$SPLIT" \
      --out-dir "$OUT_DIR" \
      --horizon 15 \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-batches "$MAX_BATCHES" \
      --top-k "$TOP_K" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --amp \
      2>&1 | tee "$LOG_FILE"

printf '\nDONE. Upload this file back into the chat:\n%s\n' "$OUT_DIR/UPLOAD_THIS_BACK_TO_CHAT.zip"
