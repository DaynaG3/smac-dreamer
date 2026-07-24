#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}"; : "${PY:?}"; : "${CONFIG:?}"; : "${CHECKPOINT:?}"; : "${SOURCE_RUN_META:?}"
cd "$REPO"; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
"$PY" -m py_compile external/r2dreamer/{dreamer.py,trainer.py,hierarchical_options.py,option_critic.py,hierarchical_dreamer.py,tools.py} scripts/{train_r2dreamer_smaclite_multimap.py,audit_option_critic_v6_progressive.py,assert_option_critic_v6_metrics.py} src/smacdreamer/validation_trainer.py
bash -n scripts/static_audit_option_critic_v6_progressive.sh scripts/run_option_critic_v6_progressive_1m.sh scripts/run_option_critic_v6_1m_then_exp45_pipeline.sh scripts/run_exp45_full_train_eval_resilient.sh
git -C "$REPO" diff --check -- .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" timeout 120s "$PY" -m pytest -q tests/test_hierarchical_options.py tests/test_option_critic_math.py tests/test_hierarchy_migration.py tests/test_hierarchical_auxiliary.py
"$PY" scripts/audit_option_critic_v6_progressive.py --repo "$REPO" --config "$CONFIG" --checkpoint "$CHECKPOINT" --source-run-meta "$SOURCE_RUN_META" --require-v1-2-source
echo "[OK] Option-Critic v6 progressive 8-slot static audit passed"
