#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?set REPO}"
: "${PY:?set PY}"
: "${CONFIG:?set CONFIG}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${SOURCE_RUN_META:?set SOURCE_RUN_META}"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
# Keep the CPU-only audit deterministic and prevent BLAS/OpenMP thread-pool
# oversubscription from stalling sequential PyTorch test processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

"$PY" -m py_compile \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/trainer.py \
  external/r2dreamer/hierarchical_options.py \
  external/r2dreamer/option_critic.py \
  external/r2dreamer/hierarchical_dreamer.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_option_critic_hierarchy.py \
  scripts/assert_option_critic_metrics.py \
  src/smacdreamer/validation_trainer.py

bash -n \
  scripts/static_audit_option_critic_hierarchy.sh \
  scripts/run_option_critic_2m.sh

git -C "$REPO" diff --check -- .

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" \
  timeout 90s "$PY" -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchy_migration.py

PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" \
  timeout 90s "$PY" - <<'PY_AUX'
from test_hierarchical_auxiliary import (
    test_hierarchical_auxiliary_loss_is_finite_and_routes_gradients,
    test_termination_head_is_frozen_during_fixed_hazard_warmup,
    test_hierarchy_control_counters_round_trip_through_training_state,
)

test_hierarchical_auxiliary_loss_is_finite_and_routes_gradients()
test_termination_head_is_frozen_during_fixed_hazard_warmup()
test_hierarchy_control_counters_round_trip_through_training_state()
print("3 auxiliary hierarchy integration tests passed")
PY_AUX

"$PY" scripts/audit_option_critic_hierarchy.py \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META" \
  --require-v1-2-source

echo "[OK] Option-Critic hierarchy v2 static audit passed"
