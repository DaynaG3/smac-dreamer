#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${SOURCE_RUN_META:?SOURCE_RUN_META is required}"
cd "$REPO"

# Keep BLAS/OpenMP pools bounded so the complete audit cannot hang through CPU
# thread oversubscription on large accelerator hosts.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

"$PY" -m py_compile \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/trainer.py \
  external/r2dreamer/tools.py \
  external/r2dreamer/hierarchical_options.py \
  external/r2dreamer/hierarchical_dreamer.py \
  external/r2dreamer/option_critic.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_option_critic_v9_anchor_safe.py \
  scripts/assert_option_critic_v9_metrics.py \
  src/smacdreamer/validation_trainer.py

bash -n \
  scripts/static_audit_option_critic_v9_anchor_safe.sh \
  scripts/run_option_critic_v9_anchor_safe_800k.sh \
  scripts/run_exp45_full_train_eval_resilient.sh \
  scripts/run_forecast_first_then_option_critic_v9_800k.sh

# Check only the integration paths involved in this run. Unrelated historical
# work in the larger combined repository must not make this audit ambiguous.
git -C "$REPO" diff --check -- \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/trainer.py \
  external/r2dreamer/tools.py \
  external/r2dreamer/hierarchical_options.py \
  external/r2dreamer/hierarchical_dreamer.py \
  external/r2dreamer/option_critic.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_option_critic_v9_anchor_safe.py \
  scripts/assert_option_critic_v9_metrics.py \
  scripts/static_audit_option_critic_v9_anchor_safe.sh \
  scripts/run_option_critic_v9_anchor_safe_800k.sh \
  scripts/run_exp45_full_train_eval_resilient.sh \
  scripts/run_forecast_first_then_option_critic_v9_800k.sh \
  src/smacdreamer/validation_trainer.py \
  configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py

# Preserve every still-applicable v6 mathematical, migration, masking, BF16,
# replay-alignment, trust-region, and gradient-routing test. The deselected
# cases assert behavior intentionally removed in v9: progressive slot locks,
# learned/cumulative-hazard termination, and the old reselection curriculum.
LEGACY_EXCLUDE='not test_locked_child_delta_cannot_change_source_policy_before_unlock and not test_minimum_duration_is_never_violated and not test_preservation_phase_reselects_at_every_eligible_state and not test_eligible_probability_uses_fixed_hazard_without_preservation_override and not test_eval_uses_deterministic_cumulative_hazard and not test_initialized_bounded_termination_matches_fixed_hazard and not test_executed_termination_probability_has_warmup_gate_and_smooth_cap_gradients and not test_locked_slots_have_zero_probability_and_zero_pg_maturity and not test_child_learned_termination_waits_for_slot_maturity and not test_v1_2_migration_has_eight_capacity_but_only_two_active_anchors and not test_progressive_slot_unlock_and_specialization_are_continuous and not test_v1_2_migration_reselects_every_state_at_step_zero'

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" \
timeout 180s "$PY" -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchical_auxiliary.py \
  tests/test_hierarchy_migration.py \
  -k "$LEGACY_EXCLUDE"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$REPO/external/r2dreamer:$REPO/tests${PYTHONPATH:+:$PYTHONPATH}" \
timeout 180s "$PY" -m pytest -q \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py

"$PY" scripts/audit_option_critic_v9_anchor_safe.py \
  --repo "$REPO" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META" \
  --expected-checkpoint-sha256 "${EXPECTED_SOURCE_CHECKPOINT_SHA256:-74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e}"

echo '[OK] Option-Critic v9 anchor-safe static audit passed'
