#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <hardening-backup-dir> <repo-dir>" >&2
  exit 2
fi

BACKUP="$(readlink -f "$1")"
REPO="$(readlink -f "$2")"
[[ -d "$BACKUP" ]] || { echo "[FAIL] backup missing: $BACKUP" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }

"${PY:-python}" - "$BACKUP" "$REPO" <<'PY'
import json
import shutil
import sys
from pathlib import Path

backup = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
manifest_path = backup / "hardening_backup_manifest.json"

if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise SystemExit("[FAIL] unsupported hardening backup manifest schema")
    recorded_repo = manifest.get("repo")
    if recorded_repo and Path(recorded_repo).resolve() != repo:
        raise SystemExit(
            "[FAIL] backup belongs to a different repo: "
            f"{Path(recorded_repo).resolve()} != {repo}"
        )

    def safe_relative(value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"[FAIL] unsafe backup manifest path: {value!r}")
        return relative

    backed_up = [safe_relative(value) for value in manifest.get("backed_up_files", [])]
    introduced = [safe_relative(value) for value in manifest.get("introduced_files", [])]
    if len(backed_up) != len(set(backed_up)) or len(introduced) != len(set(introduced)):
        raise SystemExit("[FAIL] backup manifest contains duplicate paths")

    # Validate the complete restore plan before changing any file.
    for relative in backed_up:
        source = backup / relative
        if not source.is_file():
            raise SystemExit(f"[FAIL] backup manifest file missing: {source}")

    for relative in backed_up:
        source = backup / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[RESTORE] {relative}")
    for relative in introduced:
        destination = repo / relative
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
            print(f"[REMOVE] {relative}")
    print(f"[OK] restored pre-hardening state from {backup}")
    raise SystemExit(0)

# Compatibility fallback for backups created by an earlier bundle draft.
restore_paths = [
    "external/r2dreamer/dreamer.py",
    "external/r2dreamer/tactical_policy.py",
    "scripts/train_r2dreamer_smaclite_multimap.py",
    "src/smacdreamer/validation_trainer.py",
    "configs/r2_2100_jepa_tactical_mixture.yaml",
    "configs/r2_2100_jepa_tactical_mixture_hardened.yaml",
]
for relative_text in restore_paths:
    relative = Path(relative_text)
    source = backup / relative
    if source.is_file():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[RESTORE] {relative}")

remove_paths = [
    "scripts/audit_tactical_hardening.py",
    "scripts/assert_tactical_hardened_metrics.py",
    "scripts/static_audit_tactical_hardening.sh",
    "scripts/run_tactical_hardened_2m.sh",
    "tests/test_tactical_policy_hardened.py",
]
for relative_text in remove_paths:
    destination = repo / relative_text
    if destination.is_file() or destination.is_symlink():
        destination.unlink()
        print(f"[REMOVE] {relative_text}")
if not (backup / "configs/r2_2100_jepa_tactical_mixture_hardened.yaml").is_file():
    target = repo / "configs/r2_2100_jepa_tactical_mixture_hardened.yaml"
    if target.is_file():
        target.unlink()
        print("[REMOVE] configs/r2_2100_jepa_tactical_mixture_hardened.yaml")
print(f"[OK] restored pre-hardening tactical files from {backup}")
PY
