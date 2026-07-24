#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/workspace/dreamer/combined-upload}"
export ROOT

JEPA_DIR="$ROOT/smac-jepa-wm"
TRAINER="$JEPA_DIR/smac_jepa/train_repaired_exp42_44_seqmem.py"
export TRAINER

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

echo "[INFO] ROOT=$ROOT"
echo "[INFO] TRAINER=$TRAINER"
echo "[INFO] PY=$PY"

test -f "$TRAINER" || { echo "[FAIL] trainer missing: $TRAINER"; exit 1; }

"$PY" -m py_compile "$TRAINER"

"$PY" - <<'PY'
import os
import symtable
from pathlib import Path

p = Path(os.environ["TRAINER"])
source = p.read_text()

# Hard string checks for previous known bad bugs.
for bad in [
    "current_entity_mask",
    "pred_latent.detach()",
]:
    if bad in source:
        raise SystemExit(f"[FAIL] forbidden stale/buggy snippet still present: {bad}")

# target_latent.detach() is allowed: target/EMA side should not receive inverse/action gradients.
# What we care about is that pred_latent is not detached.

required_snippets = [
    "current_condition_mask = entity_mask_seq[:, start_idx]",
    "current_predict_mask = target_entity_mask_seq_full[:, start_idx]",
    "pred_cf = pred_cf * current_predict_mask.unsqueeze(-1)",
    "build_last_seen_cache",
    "hidden_changed_count",
    "hidden_unchanged_count",
    "local_cf_count",
]
for req in required_snippets:
    if req not in source:
        raise SystemExit(f"[FAIL] required repaired snippet missing: {req}")

st = symtable.symtable(source, str(p), "exec")

allowed_global_refs = {
    "F",
    "ValueError",
    "build_last_seen_cache",
    "dynamic_dims_from_metadata",
    "float",
    "getattr",
    "int",
    "make_entity_scope_mask",
    "pooled_action_context",
    "range",
    "sigreg_loss",
    "temporal_time_weights",
    "torch",
    "weighted_bce",
    "weighted_mse",
}

bad_refs = []

def walk(table):
    if table.get_name() == "repaired_rollout_losses":
        for sym in table.get_symbols():
            if sym.is_referenced() and sym.is_global() and sym.get_name() not in allowed_global_refs:
                bad_refs.append(sym.get_name())
    for child in table.get_children():
        walk(child)

walk(st)

if bad_refs:
    raise SystemExit(
        "[FAIL] unexpected global refs in repaired_rollout_losses: "
        + ", ".join(sorted(set(bad_refs)))
    )

print("[OK] trainer static audit passed")
PY

# Bash syntax checks.
bash -n "$ROOT/repaired_exp42_44_r2_first_scripts/common_repaired_exp42_44.sh"
bash -n "$ROOT/repaired_exp42_44_r2_first_scripts/smoke_repaired_exp42_44_r2_first.sh"
bash -n "$ROOT/repaired_exp42_44_r2_first_scripts/run_repaired_r2_first_then_exp42_44.sh"

# Check common script really carries Exp40 base objective pressure.
COMMON="$ROOT/repaired_exp42_44_r2_first_scripts/common_repaired_exp42_44.sh"
grep -q -- "--event-balanced-sampling" "$COMMON"
grep -q -- "--event-dynamics-weight" "$COMMON"
grep -q -- "--delta-loss-weight" "$COMMON"
grep -q -- "--inverse-dynamics-weight" "$COMMON"

echo "[OK] bash/common-script audit passed"
