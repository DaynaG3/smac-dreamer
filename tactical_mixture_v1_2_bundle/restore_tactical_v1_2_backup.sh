#!/usr/bin/env bash
set -euo pipefail
BACKUP="${1:?usage: restore_tactical_v1_2_backup.sh BACKUP_DIR REPO}"
REPO="${2:?usage: restore_tactical_v1_2_backup.sh BACKUP_DIR REPO}"
MANIFEST="$BACKUP/v1_2_backup_manifest.json"
[[ -s "$MANIFEST" ]] || { echo "[FAIL] missing manifest: $MANIFEST" >&2; exit 1; }
python - "$BACKUP" "$REPO" "$MANIFEST" <<'PY'
import json, shutil, sys
from pathlib import Path
backup,repo,manifest=map(Path,sys.argv[1:])
data=json.loads(manifest.read_text())
for rel in data["backed_up_files"]:
    src=backup/rel; dst=repo/rel
    if not src.is_file(): raise SystemExit(f"[FAIL] missing backup {src}")
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
for rel in data["introduced_files"]:
    path=repo/rel
    if path.exists(): path.unlink()
print("[OK] restored pre-v1.2 files")
PY
