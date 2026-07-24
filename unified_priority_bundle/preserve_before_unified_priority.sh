#!/usr/bin/env bash
set -euo pipefail

REPO="${1:?Usage: preserve_before_unified_priority.sh REPO CHECKPOINT [BACKUP_ROOT]}"
CHECKPOINT="${2:?Usage: preserve_before_unified_priority.sh REPO CHECKPOINT [BACKUP_ROOT]}"
BACKUP_ROOT="${3:-$(dirname "$REPO")}" 
STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE="$BACKUP_ROOT/preserve_before_unified_priority_$STAMP"

REPO="$(cd "$REPO" && pwd)"
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
mkdir -p "$SAFE"

if ! git -C "$REPO" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "[FAIL] $REPO is not inside a Git working tree" >&2
  exit 2
fi
GIT_ROOT="$(git -C "$REPO" rev-parse --show-toplevel)"
echo "[INFO] Git root: $GIT_ROOT"
echo "[INFO] Target subtree: $REPO"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[FAIL] checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

# Preserve Git refs and every working-tree code/config change, including untracked files.
git -C "$REPO" rev-parse HEAD > "$SAFE/HEAD.txt"
git -C "$REPO" branch --show-current > "$SAFE/BRANCH.txt"
git -C "$REPO" status --short --untracked-files=all > "$SAFE/STATUS.txt"
git -C "$REPO" diff --binary > "$SAFE/UNSTAGED.patch"
git -C "$REPO" diff --cached --binary > "$SAFE/STAGED.patch"
git -C "$REPO" bundle create "$SAFE/repo_all_refs.bundle" --all

mkdir -p "$SAFE/repo_worktree"
rsync -a \
  --exclude='.git/' \
  --exclude='logs/' \
  --exclude='**/replay/' \
  --exclude='**/replay_stale_*/' \
  --exclude='__pycache__/' \
  "$REPO/" "$SAFE/repo_worktree/"

# Preserve the exact continuation source and its reconstruction metadata.
cp -a "$CHECKPOINT" "$SAFE/source_latest.pt"
SOURCE_RUN="$(dirname "$CHECKPOINT")"
for name in run_meta.json run_config.json; do
  [[ -f "$SOURCE_RUN/$name" ]] && cp -a "$SOURCE_RUN/$name" "$SAFE/$name"
done

(
  cd "$SAFE"
  sha256sum source_latest.pt > SHA256SUMS.txt
  [[ -f run_meta.json ]] && sha256sum run_meta.json >> SHA256SUMS.txt
  [[ -f run_config.json ]] && sha256sum run_config.json >> SHA256SUMS.txt
)

cat > "$SAFE/RESTORE_NOTES.txt" <<EOF
Repository: $REPO
Checkpoint: $CHECKPOINT

Restore code/config snapshot (does not touch .git):
  rsync -a '$SAFE/repo_worktree/' '$REPO/'

Restore Git refs into a separate clone:
  git clone '$SAFE/repo_all_refs.bundle' restored-smac-dreamer

The installer also creates a second, smaller backup containing every file it patches.
EOF

echo "[OK] preservation snapshot: $SAFE"
echo "$SAFE"
