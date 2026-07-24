#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?set REPO}"
: "${PY:?set PY}"
: "${CONFIG:?set CONFIG}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${SOURCE_RUN_META:?set SOURCE_RUN_META}"

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
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
  external/r2dreamer/tools.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_option_critic_v5_stability.py \
  scripts/assert_option_critic_v5_metrics.py \
  src/smacdreamer/validation_trainer.py

bash -n \
  scripts/static_audit_option_critic_v5_stability.sh \
  scripts/run_option_critic_v5_stability_1m.sh \
  scripts/run_exp45_full_train_eval_resilient.sh \
  scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh

git -C "$REPO" diff --check -- .

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" \
  timeout 120s "$PY" -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchy_migration.py \
  tests/test_hierarchical_auxiliary.py

"$PY" scripts/audit_option_critic_v5_stability.py \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META" \
  --require-v1-2-source

echo "[OK] Option-Critic v5 stability static audit passed"
