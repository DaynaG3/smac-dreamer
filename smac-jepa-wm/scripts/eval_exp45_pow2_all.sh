#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${CHECKPOINT:-${1:-runs/rnn_seqmem_exp45_pow2_direct_1_2_4_8_16/checkpoint.pt}}"
CHECKPOINT="$CHECKPOINT" bash scripts/eval_exp45_pow2_ordinary.sh
CHECKPOINT="$CHECKPOINT" bash scripts/eval_exp45_pow2_hidden.sh
echo "[OK] all Exp45 evaluations completed"
