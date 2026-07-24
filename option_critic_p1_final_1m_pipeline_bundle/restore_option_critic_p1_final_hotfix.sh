#!/usr/bin/env bash
set -euo pipefail

BACKUP="${1:?usage: restore_option_critic_p1_final_hotfix.sh BACKUP_DIR REPO}"
REPO="${2:?usage: restore_option_critic_p1_final_hotfix.sh BACKUP_DIR REPO}"
PY="${PY:-python}"
MANIFEST="$BACKUP/option_critic_p1_final_backup_manifest.json"

[[ -f "$MANIFEST" ]] || { echo "[FAIL] backup manifest missing: $MANIFEST" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }

"$PY" - "$BACKUP" "$REPO" "$MANIFEST" <<'PY'
from __future__ import annotations
import hashlib, json, pathlib, shutil, sys
backup=pathlib.Path(sys.argv[1]).resolve()
repo=pathlib.Path(sys.argv[2]).resolve()
manifest=json.loads(pathlib.Path(sys.argv[3]).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

for rel, expected in manifest['backed_up_sha256'].items():
    src=backup/rel
    if not src.is_file(): raise SystemExit(f'[FAIL] missing backup file: {rel}')
    actual=sha(src)
    if actual!=expected: raise SystemExit(f'[FAIL] backup hash mismatch: {rel}')
for rel in manifest['backed_up_files']:
    src=backup/rel; dst=repo/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
for rel in manifest['introduced_files']:
    path=repo/rel
    if path.exists():
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
for rel, expected in manifest['backed_up_sha256'].items():
    actual=sha(repo/rel)
    if actual!=expected: raise SystemExit(f'[FAIL] restored hash mismatch: {rel}')
print('[OK] restored pre-P1-final files and removed introduced files')
PY
