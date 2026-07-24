#!/usr/bin/env bash
set -euo pipefail
BACKUP="${1:?Usage: restore_from_installer_backup.sh BACKUP_DIR REPO_DIR}"
REPO="${2:?Usage: restore_from_installer_backup.sh BACKUP_DIR REPO_DIR}"

for rel in \
  src/smacdreamer/envs/map_sampler.py \
  src/smacdreamer/r2dreamer_factory.py \
  src/smacdreamer/checkpointing.py \
  external/r2dreamer/trainer.py \
  external/r2dreamer/dreamer.py \
  scripts/train_r2dreamer_smaclite_multimap.py
do
  install -D -m 0644 "$BACKUP/$rel" "$REPO/$rel"
done

echo "[OK] restored patched existing files from $BACKUP"
echo "[NOTE] newly added files remain; remove them manually only after verifying restore."
