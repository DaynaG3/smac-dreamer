#!/usr/bin/env bash
set -euo pipefail
REPO="${1:?usage: $0 REPO SOURCE_CHECKPOINT}"
CHECKPOINT="${2:?usage: $0 REPO SOURCE_CHECKPOINT}"
ROOT="$(dirname "$(readlink -f "$REPO")")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/preserve_before_option_critic_$STAMP"
mkdir -p "$OUT"
GIT_ROOT="$(git -C "$REPO" rev-parse --show-toplevel)"
git -C "$GIT_ROOT" status --short > "$OUT/git_status.txt"
git -C "$GIT_ROOT" diff --binary > "$OUT/working_tree.patch"
git -C "$GIT_ROOT" diff --cached --binary > "$OUT/staged.patch"
git -C "$GIT_ROOT" rev-parse HEAD > "$OUT/git_head.txt"
# Preserve the exact integration files and all untracked source/tests/scripts,
# while deliberately excluding multi-GB logs/replay/checkpoints.
tar -C "$REPO" -czf "$OUT/smac-dreamer-source-snapshot.tgz" \
  --exclude='./logs' --exclude='./wandb' --exclude='./checkpoints' \
  --exclude='./**/__pycache__' --exclude='./.pytest_cache' .
cp -a "$CHECKPOINT" "$OUT/source_checkpoint.pt"
sha256sum "$CHECKPOINT" "$OUT/source_checkpoint.pt" > "$OUT/checkpoint_sha256.txt"
echo "[OK] preservation directory: $OUT"
