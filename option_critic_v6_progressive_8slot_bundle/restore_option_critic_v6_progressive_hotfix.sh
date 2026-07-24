#!/usr/bin/env bash
set -euo pipefail
BACKUP="${1:?backup dir}"; REPO="${2:?repo}"; PY="${PY:-python}"
MANIFEST="$BACKUP/option_critic_v6_progressive_backup_manifest.json"
[[ -f "$MANIFEST" ]] || { echo "[FAIL] missing $MANIFEST" >&2; exit 1; }
"$PY" - "$BACKUP" "$REPO" "$MANIFEST" <<'PY'
import hashlib,json,pathlib,shutil,sys
b=pathlib.Path(sys.argv[1]); r=pathlib.Path(sys.argv[2]); m=json.loads(pathlib.Path(sys.argv[3]).read_text())
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
for rel,expected in m['backed_up_sha256'].items():
 p=b/rel
 assert p.is_file() and sha(p)==expected,(rel,'backup hash mismatch')
 d=r/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,d)
 assert sha(d)==expected,(rel,'restore hash mismatch')
for rel in m['introduced_files']:
 p=r/rel
 if p.exists(): p.unlink()
print('[OK] restored v5 files and removed v6 introduced files')
PY
