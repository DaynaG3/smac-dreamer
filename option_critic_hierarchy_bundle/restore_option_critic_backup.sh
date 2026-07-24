#!/usr/bin/env bash
set -euo pipefail
BACKUP="${1:?usage: $0 BACKUP_DIR REPO}"
REPO="${2:?usage: $0 BACKUP_DIR REPO}"
MANIFEST="$BACKUP/option_critic_backup_manifest.json"
[[ -s "$MANIFEST" ]] || { echo "[FAIL] missing $MANIFEST" >&2; exit 1; }
PY="${PY:-python}"
"$PY" - "$BACKUP" "$REPO" <<'PY'
import json,shutil,sys
from pathlib import Path
backup=Path(sys.argv[1]).resolve(); repo=Path(sys.argv[2]).resolve()
m=json.loads((backup/'option_critic_backup_manifest.json').read_text())
if int(m.get('schema_version', -1)) != 2:
    raise SystemExit('[FAIL] unsupported backup manifest schema')
if Path(m['repo']).resolve()!=repo: raise SystemExit('[FAIL] manifest repo mismatch')
expected_hashes=m.get('backed_up_sha256') or {}
for rel in m['backed_up_files']:
    src=backup/rel; dst=repo/rel
    if not src.is_file(): raise SystemExit(f'[FAIL] backup missing {rel}')
    import hashlib
    h=hashlib.sha256(src.read_bytes()).hexdigest()
    if h != expected_hashes.get(rel):
        raise SystemExit(f'[FAIL] backup hash mismatch for {rel}')
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
for rel in m['introduced_files']:
    p=repo/rel
    if p.exists():
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
print('[OK] restored pre-Option-Critic files')
PY
