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
    "train/option/real_source_policy_kl_mean",
    "train/option/real_source_policy_kl_tail",
    "train/option/high_confidence_action_flip_rate",
    "train/option/real_source_high_confidence_action_flip_rate",
    "train/option/source_manager_group_kl_mean",
    "train/option/source_manager_group_kl_tail",
    "train/option/real_source_manager_group_kl_mean",
    "train/option/real_source_manager_group_kl_tail",
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
    for index in range(2):
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
        raise SystemExit(f"[FAIL] missing v5 stability metrics: {missing}")
    bad = [key for key in required if not math.isfinite(float(latest[key]))]
    if bad:
        raise SystemExit(f"[FAIL] non-finite v5 stability metrics: {bad}")

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
    if abs(float(latest["train/option/world_model_grad_scale"])) > 1.0e-8:
        raise SystemExit("[FAIL] source world model is not frozen")
    if abs(float(latest["train/option/termination_execution_cap"]) - 0.30) > 1.0e-5:
        raise SystemExit(
            "[FAIL] termination execution cap drifted from fixed 0.30: "
            f"{latest['train/option/termination_execution_cap']}"
        )

    trust_values = {
        "imag_action_mean": float(latest["train/option/source_policy_kl_mean"]),
        "imag_action_tail": float(latest["train/option/source_policy_kl_tail"]),
        "real_action_mean": float(latest["train/option/real_source_policy_kl_mean"]),
        "real_action_tail": float(latest["train/option/real_source_policy_kl_tail"]),
        "imag_manager_mean": float(latest["train/option/source_manager_group_kl_mean"]),
        "imag_manager_tail": float(latest["train/option/source_manager_group_kl_tail"]),
        "real_manager_mean": float(latest["train/option/real_source_manager_group_kl_mean"]),
        "real_manager_tail": float(latest["train/option/real_source_manager_group_kl_tail"]),
    }
    for name in ("imag_action_mean", "real_action_mean"):
        if trust_values[name] > 0.05:
            raise SystemExit(f"[FAIL] unsafe source action mean KL: {name}={trust_values[name]}")
    for name in ("imag_action_tail", "real_action_tail"):
        if trust_values[name] > 0.15:
            raise SystemExit(f"[FAIL] unsafe source action tail KL: {name}={trust_values[name]}")
    for name in ("imag_manager_mean", "real_manager_mean"):
        if trust_values[name] > 0.03:
            raise SystemExit(f"[FAIL] unsafe source manager mean KL: {name}={trust_values[name]}")
    for name in ("imag_manager_tail", "real_manager_tail"):
        if trust_values[name] > 0.10:
            raise SystemExit(f"[FAIL] unsafe source manager tail KL: {name}={trust_values[name]}")

    for key in (
        "train/option/high_confidence_action_flip_rate",
        "train/option/real_source_high_confidence_action_flip_rate",
        "train/option/source_manager_group_high_confidence_flip_rate",
    ):
        if float(latest[key]) > 0.10:
            raise SystemExit(f"[FAIL] unsafe high-confidence source flip rate: {key}={latest[key]}")

    horizon = int(round(float(latest["train/option/imag_horizon"])))
    if not 7 <= horizon <= 12:
        raise SystemExit(f"[FAIL] hierarchy imagination horizon out of range: {horizon}")

    manager_blend = float(latest["train/option/manager_pg_blend"])
    worker_blend = float(latest["train/option/worker_pg_blend"])
    term_blend = float(latest["train/option/termination_blend"])
    reselect = float(latest["train/option/commitment_reselect_probability"])
    if latest_step <= 20_000 and worker_blend > 1.0e-6:
        raise SystemExit("[FAIL] worker PG activated before 20k critic warm-up")
    if latest_step >= 150_000 and worker_blend < 0.999:
        raise SystemExit("[FAIL] worker PG did not finish ramping by 150k")
    if latest_step <= 100_000:
        if manager_blend > 1.0e-6:
            raise SystemExit("[FAIL] manager PG activated before 100k")
        if reselect < 0.999:
            raise SystemExit("[FAIL] exact source routing not preserved through 100k")
    if latest_step >= 500_000 and manager_blend < 0.999:
        raise SystemExit("[FAIL] manager PG did not finish ramping by 500k")
    if latest_step >= 600_000 and abs(reselect - 0.25) > 1.0e-3:
        raise SystemExit(f"[FAIL] reactive reselection floor is wrong: {reselect}")
    if reselect < 0.249 - 1.0e-6:
        raise SystemExit(f"[FAIL] reactive reselection fell below 0.25: {reselect}")
    if latest_step <= 350_000 and term_blend > 1.0e-6:
        raise SystemExit("[FAIL] learned termination activated before 350k")
    if latest_step >= 800_000 and term_blend < 0.999:
        raise SystemExit("[FAIL] learned termination did not finish ramping by 800k")

    warnings: list[str] = []
    targets = {
        "imag_action_mean": 0.002,
        "real_action_mean": 0.002,
        "imag_action_tail": 0.01,
        "real_action_tail": 0.01,
        "imag_manager_mean": 0.001,
        "real_manager_mean": 0.001,
        "imag_manager_tail": 0.005,
        "real_manager_tail": 0.005,
    }
    for name, target in targets.items():
        if trust_values[name] > target:
            warnings.append(f"{name} exceeds target: {trust_values[name]:.6f} > {target:.6f}")
    if float(latest["train/option/residual_to_base_ratio"]) > 0.25:
        warnings.append("residual/source ratio exceeds 0.25")

    print(f"[OK] Option-Critic v5 stability metrics passed; latest_step={latest_step}")
    print("[OK] structural invariants zero; real+imag source distillation finite; cap fixed")
    for warning in warnings:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
