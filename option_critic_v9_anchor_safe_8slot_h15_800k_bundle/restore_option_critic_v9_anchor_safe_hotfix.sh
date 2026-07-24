#!/usr/bin/env bash
set -euo pipefail
BACKUP="${1:?backup directory required}"; REPO="${2:?repo required}"; PY="${PY:-python}"
MANIFEST="$BACKUP/option_critic_v9_anchor_safe_backup_manifest.json"
[[ -f "$MANIFEST" ]] || { echo '[FAIL] backup manifest missing' >&2; exit 1; }
"$PY" - "$BACKUP" "$REPO" <<'PY'
import hashlib,json,pathlib,shutil,sys
b=pathlib.Path(sys.argv[1]); r=pathlib.Path(sys.argv[2]); m=json.loads((b/'option_critic_v9_anchor_safe_backup_manifest.json').read_text())
for rel,expected in m['sha256'].items():
 p=b/rel
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise SystemExit('[FAIL] backup hash mismatch: '+rel)
for rel in m['replaced']:
 d=r/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(b/rel,d)
for rel in m['introduced']:
 p=r/rel
 if p.exists(): p.unlink()
print('[OK] restored pre-v9 files and removed v9-introduced files')
PY
