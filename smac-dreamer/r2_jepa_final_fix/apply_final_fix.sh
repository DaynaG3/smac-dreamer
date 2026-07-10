#!/usr/bin/env bash
set -euo pipefail

# Run this from repo root: ~/workspace/dreamer/combined-upload/smac-dreamer
ROOT="$PWD"
if [[ -d "$ROOT/src/smacdreamer/jepa" ]]; then
  TARGET="$ROOT/src/smacdreamer/jepa"
elif [[ -d "$ROOT/smac-dreamer/src/smacdreamer/jepa" ]]; then
  TARGET="$ROOT/smac-dreamer/src/smacdreamer/jepa"
else
  echo "Could not find src/smacdreamer/jepa from $ROOT"
  exit 1
fi

BACKUP="patch_backups/final_r2_jepa_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
cp "$TARGET/world_model.py" "$BACKUP/world_model.py.before"
cp "$TARGET/feature_adapter.py" "$BACKUP/feature_adapter.py.before"
cp "r2_jepa_final_fix/world_model.py" "$TARGET/world_model.py"
cp "r2_jepa_final_fix/feature_adapter.py" "$TARGET/feature_adapter.py"
cp "r2_jepa_final_fix/preflight_final_r2_jepa.py" .
cp "r2_jepa_final_fix/make_final_2m_config.py" .
cp "r2_jepa_final_fix/launch_final_2m_wandb.sh" .
cp "r2_jepa_final_fix/monitor_final_run.sh" .
chmod +x launch_final_2m_wandb.sh monitor_final_run.sh

echo "Patched files under: $TARGET"
echo "Backup saved under: $BACKUP"
