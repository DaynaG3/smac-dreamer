#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JEPA_DIR="$ROOT/smac-jepa-wm"
SCRIPTS_DIR="$ROOT/repaired_exp42_44_r2_first_scripts"

if [[ ! -d "$JEPA_DIR/smac_jepa" ]]; then
  echo "[FAIL] JEPA repo not found at $JEPA_DIR" >&2
  exit 1
fi
mkdir -p "$SCRIPTS_DIR"
cp "$BUNDLE_DIR/smac_jepa/train_repaired_exp42_44_seqmem.py" "$JEPA_DIR/smac_jepa/train_repaired_exp42_44_seqmem.py"
cp "$BUNDLE_DIR/repaired_scripts"/*.sh "$SCRIPTS_DIR/"
cp "$BUNDLE_DIR/repaired_scripts"/*.py "$SCRIPTS_DIR/" 2>/dev/null || true
chmod +x "$SCRIPTS_DIR"/*.sh

echo "[OK] Installed repaired trainer: $JEPA_DIR/smac_jepa/train_repaired_exp42_44_seqmem.py"
echo "[OK] Installed scripts: $SCRIPTS_DIR"
echo "Next: cd $SCRIPTS_DIR && ROOT=$ROOT R2_RUN=<run_dir> ./smoke_repaired_exp42_44_r2_first.sh"
