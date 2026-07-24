#!/usr/bin/env bash
set -euo pipefail

REPO="${1:?Usage: preserve_before_tactical_mixture.sh REPO CHECKPOINT}"
CHECKPOINT="${2:?Usage: preserve_before_tactical_mixture.sh REPO CHECKPOINT}"

REPO="$(cd "$REPO" && pwd)"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
if ! git -C "$REPO" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "[FAIL] $REPO is not inside a Git working tree" >&2
  exit 2
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "[FAIL] checkpoint missing or empty: $CHECKPOINT" >&2
  exit 2
fi

GIT_ROOT="$(git -C "$REPO" rev-parse --show-toplevel)"
ROOT="$(dirname "$REPO")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/preserve_before_tactical_mixture_$STAMP"
mkdir -p "$OUT"

printf '%s\n' \
  "git_root=$GIT_ROOT" \
  "target_subtree=$REPO" \
  "checkpoint=$CHECKPOINT" \
  "branch=$(git -C "$GIT_ROOT" branch --show-current)" \
  "commit=$(git -C "$GIT_ROOT" rev-parse HEAD)" \
  > "$OUT/METADATA.txt"

git -C "$GIT_ROOT" status --short > "$OUT/STATUS.txt"
git -C "$GIT_ROOT" diff > "$OUT/UNSTAGED.patch"
git -C "$GIT_ROOT" diff --cached > "$OUT/STAGED.patch"
git -C "$GIT_ROOT" bundle create "$OUT/repo_all_refs.bundle" --all

# Preserve all current files in the smac-dreamer subtree, including untracked
# adaptive-priority integration files, without copying large logs/replays.
mkdir -p "$OUT/smac-dreamer-working-tree"
rsync -a \
  --exclude='logs/' \
  --exclude='wandb/' \
  --exclude='replay/' \
  --exclude='replay_stale_*/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  "$REPO/" "$OUT/smac-dreamer-working-tree/"

cp -a "$CHECKPOINT" "$OUT/source_checkpoint.pt"
sha256sum "$CHECKPOINT" "$OUT/source_checkpoint.pt" \
  > "$OUT/CHECKPOINT_SHA256.txt"

cat > "$OUT/RESTORE_WORKTREE.txt" <<EOF
To restore the preserved smac-dreamer subtree:
  rsync -a '$OUT/smac-dreamer-working-tree/' '$REPO/'

Git refs are preserved in:
  $OUT/repo_all_refs.bundle
EOF

printf '[OK] preservation snapshot: %s\n' "$OUT"
printf '[OK] Git root: %s\n' "$GIT_ROOT"
printf '[OK] Target subtree: %s\n' "$REPO"
