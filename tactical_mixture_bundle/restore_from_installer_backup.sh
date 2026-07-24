#!/usr/bin/env bash
set -euo pipefail

BACKUP="${1:?Usage: restore_from_installer_backup.sh BACKUP_DIR REPO_DIR}"
REPO="${2:?Usage: restore_from_installer_backup.sh BACKUP_DIR REPO_DIR}"

for relative in \
  external/r2dreamer/dreamer.py \
  scripts/train_r2dreamer_smaclite_multimap.py
do
  source="$BACKUP/$relative"
  target="$REPO/$relative"
  if [[ ! -f "$source" ]]; then
    echo "[FAIL] backup missing: $source" >&2
    exit 2
  fi
  install -D -m 0644 "$source" "$target"
done

rm -f \
  "$REPO/external/r2dreamer/tactical_policy.py" \
  "$REPO/scripts/preflight_tactical_mixture.py" \
  "$REPO/scripts/static_audit_tactical_mixture.sh" \
  "$REPO/scripts/run_tactical_mixture_2m.sh" \
  "$REPO/scripts/assert_tactical_metrics.py" \
  "$REPO/tests/test_tactical_policy.py" \
  "$REPO/configs/r2_2100_jepa_tactical_mixture.yaml"

echo "[OK] restored replaced files and removed tactical additions"
