#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BACKUP_DIR REPO" >&2
  exit 2
fi
BACKUP="$(readlink -f "$1")"
REPO="$(readlink -f "$2")"
MANIFEST="$BACKUP/option_critic_p0p1_hotfix_backup_manifest.json"
[[ -f "$MANIFEST" ]] || { echo "[FAIL] missing backup manifest: $MANIFEST" >&2; exit 1; }
PY="${PY:-python3}"

"$PY" - "$BACKUP" "$REPO" "$MANIFEST" <<'PY'
import hashlib, json, pathlib, shutil, sys
backup=pathlib.Path(sys.argv[1])
repo=pathlib.Path(sys.argv[2])
manifest=json.loads(pathlib.Path(sys.argv[3]).read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

for rel in manifest['backed_up_files']:
    src=backup/rel
    dst=repo/rel
    if not src.is_file(): raise SystemExit(f'[FAIL] backup file missing: {rel}')
    expected=manifest['backed_up_sha256'][rel]
    if sha(src)!=expected: raise SystemExit(f'[FAIL] backup hash changed: {rel}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    if sha(dst)!=expected: raise SystemExit(f'[FAIL] restore hash mismatch: {rel}')
for rel in manifest['introduced_files']:
    path=repo/rel
    if path.exists():
        if not path.is_file(): raise SystemExit(f'[FAIL] introduced path is not a file: {rel}')
        path.unlink()
print('[OK] restored all replaced files and removed hotfix-introduced files')
PY
