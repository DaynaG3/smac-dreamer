#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BASE_REQUIRED = (
    "train/option/worker_policy_loss",
    "train/option/manager_policy_loss",
    "train/option/termination_loss",
    "train/option/critic_loss",
    "train/option/legacy_behavior_losses_disabled",
    "train/option/termination_blend",
    "train/option/manager_pg_blend",
    "train/option/worker_pg_blend",
    "train/option/commitment_reselect_probability",
    "train/option/world_model_grad_scale",
    "train/option/imag_horizon",
    "train/option/source_policy_kl_mean",
    "train/option/source_policy_kl_tail",
    "train/option/high_confidence_action_flip_rate",
    "train/option/source_manager_group_kl_mean",
    "train/option/source_manager_group_kl_tail",
    "train/option/source_manager_group_high_confidence_flip_rate",
    "train/option/source_manager_group_preservation_loss",
    "train/option/action_preservation_loss",
    "train/option/residual_to_base_ratio",
    "train/option/eligible_learned_beta_mean",
    "train/option/bounded_learned_beta_mean",
    "train/option/raw_to_bounded_beta_gap",
    "train/option/termination_execution_cap",
    "train/option/real_boundary_rate",
    "train/option/real_eligible_termination_rate",
    "train/option/real_mean_action_age",
    "train/option/imag_min_duration_violation_rate",
    "train/option/imag_max_duration_violation_rate",
    "train/option/imag_change_without_boundary_rate",
    "train/option/real_min_duration_violation_rate",
    "train/option/real_max_duration_violation_rate",
    "train/option/real_change_without_boundary_rate",
)
STRUCTURAL_ZERO = tuple(
    key for key in BASE_REQUIRED
    if "violation_rate" in key or "change_without_boundary" in key
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    path = args.run / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"[FAIL] missing {path}")

    latest: dict[str, float] = {}
    latest_step = -1
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_step = row.get("global_step", row.get("step", -1))
            if isinstance(row_step, (int, float)):
                latest_step = max(latest_step, int(row_step))
            latest.update({
                key: value for key, value in row.items()
                if isinstance(value, (int, float))
            })

    required = list(BASE_REQUIRED)
    for index in range(8):
        required.extend((
            f"train/option/usage_{index}",
            f"train/option/real_usage_{index}",
            f"train/option/imag_weight_{index}",
        ))
    for group in range(2):
        required.extend((
            f"train/option/source_manager_group_live_{group}",
            f"train/option/source_manager_group_reference_{group}",
        ))
    missing = [key for key in required if key not in latest]
    if missing:
        raise SystemExit(f"[FAIL] missing P1-final metrics: {missing}")
    bad = [key for key in required if not math.isfinite(float(latest[key]))]
    if bad:
        raise SystemExit(f"[FAIL] non-finite P1-final metrics: {bad}")

    for key in (
        "train/imag_post_mask_invalid_sample_rate",
        "train/real_post_mask_invalid_sample_rate",
    ):
        if float(latest.get(key, 0.0)) > 1.0e-6:
            raise SystemExit(f"[FAIL] {key} is nonzero: {latest[key]}")
    for key in STRUCTURAL_ZERO:
        if abs(float(latest[key])) > 1.0e-8:
            raise SystemExit(f"[FAIL] state-machine violation: {key}={latest[key]}")
    if float(latest["train/option/legacy_behavior_losses_disabled"]) != 1.0:
        raise SystemExit("[FAIL] competing legacy behavior losses are active")

    world_scale = float(latest["train/option/world_model_grad_scale"])
    if abs(world_scale) > 1.0e-8:
        raise SystemExit(f"[FAIL] source world model is not frozen: {world_scale}")

    action_mean_kl = float(latest["train/option/source_policy_kl_mean"])
    action_tail_kl = float(latest["train/option/source_policy_kl_tail"])
    action_flip = float(latest["train/option/high_confidence_action_flip_rate"])
    manager_mean_kl = float(latest["train/option/source_manager_group_kl_mean"])
    manager_tail_kl = float(latest["train/option/source_manager_group_kl_tail"])
    manager_flip = float(
        latest["train/option/source_manager_group_high_confidence_flip_rate"]
    )
    if action_mean_kl > 0.10:
        raise SystemExit(f"[FAIL] source-policy mean KL is unsafe: {action_mean_kl}")
    if action_tail_kl > 0.25:
        raise SystemExit(f"[FAIL] source-policy tail KL is unsafe: {action_tail_kl}")
    if action_flip > 0.20:
        raise SystemExit(
            f"[FAIL] high-confidence source action flip rate is unsafe: {action_flip}"
        )
    if manager_mean_kl > 0.08:
        raise SystemExit(
            f"[FAIL] source-manager group mean KL is unsafe: {manager_mean_kl}"
        )
    if manager_tail_kl > 0.25:
        raise SystemExit(
            f"[FAIL] source-manager group tail KL is unsafe: {manager_tail_kl}"
        )
    if manager_flip > 0.15:
        raise SystemExit(
            "[FAIL] high-confidence source-manager group flip rate is unsafe: "
            f"{manager_flip}"
        )

    horizon = int(round(float(latest["train/option/imag_horizon"])))
    if not 7 <= horizon <= 15:
        raise SystemExit(f"[FAIL] hierarchy imagination horizon out of range: {horizon}")

    manager_blend = float(latest["train/option/manager_pg_blend"])
    worker_blend = float(latest["train/option/worker_pg_blend"])
    term_blend = float(latest["train/option/termination_blend"])
    reselect = float(latest["train/option/commitment_reselect_probability"])

    if latest_step <= 20_000 and worker_blend > 1.0e-6:
        raise SystemExit("[FAIL] worker PG activated before 20k critic warm-up")
    if latest_step >= 100_000 and worker_blend < 0.999:
        raise SystemExit("[FAIL] worker PG did not finish ramping by 100k")
    if latest_step <= 100_000:
        if manager_blend > 1.0e-6:
            raise SystemExit("[FAIL] manager PG activated before 100k")
        if reselect < 0.999:
            raise SystemExit("[FAIL] source per-state routing not preserved through 100k")
    if latest_step >= 250_000:
        if manager_blend < 0.999:
            raise SystemExit("[FAIL] manager PG did not finish ramping by 250k")
        if reselect > 1.0e-3:
            raise SystemExit("[FAIL] temporal commitment did not finish by 250k")
    if latest_step <= 250_000 and term_blend > 1.0e-6:
        raise SystemExit("[FAIL] learned termination activated before 250k")
    if latest_step >= 400_000 and term_blend < 0.999:
        raise SystemExit("[FAIL] learned termination did not finish ramping by 400k")

    warnings = []
    if action_mean_kl > 0.01:
        warnings.append(f"mean source action KL exceeds target: {action_mean_kl:.5f}")
    if action_tail_kl > 0.03:
        warnings.append(f"tail source action KL exceeds target: {action_tail_kl:.5f}")
    if action_flip > 0.02:
        warnings.append(f"high-confidence source action flip elevated: {action_flip:.4f}")
    if manager_mean_kl > 0.005:
        warnings.append(f"source-manager group mean KL exceeds target: {manager_mean_kl:.5f}")
    if manager_tail_kl > 0.02:
        warnings.append(f"source-manager group tail KL exceeds target: {manager_tail_kl:.5f}")
    if manager_flip > 0.02:
        warnings.append(f"source-manager group flip elevated: {manager_flip:.4f}")
    if float(latest["train/option/residual_to_base_ratio"]) > 0.25:
        warnings.append("residual/source ratio exceeds target")

    print(f"[OK] Option-Critic P1-final metrics passed; latest_step={latest_step}")
    print(
        "[OK] frozen world model; structural violations zero; "
        "action and manager-routing trust regions finite"
    )
    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
