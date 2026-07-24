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
    "train/option/termination_blend",
    "train/option/termination_active_upper_bound",
    "train/option/termination_execution_cap",
    "train/option/manager_pg_blend",
    "train/option/effective_count",
    "train/option/manager_mutual_information_normalized",
    "train/option/worker_entropy",
    "train/option/action_js_mean",
    "train/option/js_shortfall_fraction",
    "train/option/base_kl_mean",
    "train/option/residual_to_base_ratio",
    "train/option/boundary_rate",
    "train/option/eligible_learned_beta_mean",
    "train/option/termination_entropy",
    "train/option/termination_reliable_fraction",
    "train/option/termination_target_sign_agreement",
    "train/option/termination_target_disagreement",
    "train/option/manager_mi_shortfall_loss",
    "train/option/residual_duplicate_fraction",
    "train/option/same_option_reselection_rate",
    "train/option/real_boundary_rate",
    "train/option/real_termination_rate",
    "train/option/real_eligible_termination_rate",
    "train/option/real_mean_action_age",
    "train/option/real_mean_completed_dwell",
    "train/option/worker_advantage_rms",
    "train/option/manager_advantage_rms",
    "train/option/legacy_behavior_losses_disabled",
    "train/option/imag_min_duration_violation_rate",
    "train/option/imag_max_duration_violation_rate",
    "train/option/imag_change_without_boundary_rate",
    "train/option/real_min_duration_violation_rate",
    "train/option/real_max_duration_violation_rate",
    "train/option/real_change_without_boundary_rate",
)
STRUCTURAL_ZERO = (
    "train/option/imag_min_duration_violation_rate",
    "train/option/imag_max_duration_violation_rate",
    "train/option/imag_change_without_boundary_rate",
    "train/option/real_min_duration_violation_rate",
    "train/option/real_max_duration_violation_rate",
    "train/option/real_change_without_boundary_rate",
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
            latest.update(
                {key: value for key, value in row.items() if isinstance(value, (int, float))}
            )

    required = list(BASE_REQUIRED)
    for index in range(8):
        required.extend(
            (
                f"train/option/usage_{index}",
                f"train/option/real_usage_{index}",
                f"train/option/imag_weight_{index}",
                f"train/option/imag_return_{index}",
            )
        )
    missing = [key for key in required if key not in latest]
    if missing:
        raise SystemExit(f"[FAIL] missing option metrics: {missing}")
    bad = [key for key in required if not math.isfinite(float(latest[key]))]
    if bad:
        raise SystemExit(f"[FAIL] non-finite option metrics: {bad}")

    for key in (
        "train/imag_post_mask_invalid_sample_rate",
        "train/real_post_mask_invalid_sample_rate",
    ):
        if float(latest.get(key, 0.0)) > 1.0e-6:
            raise SystemExit(f"[FAIL] {key} is nonzero: {latest[key]}")
    for key in STRUCTURAL_ZERO:
        if abs(float(latest[key])) > 1.0e-8:
            raise SystemExit(f"[FAIL] hierarchy state-machine violation: {key}={latest[key]}")

    ratio = float(latest["train/option/residual_to_base_ratio"])
    if ratio > 0.50:
        raise SystemExit(f"[FAIL] option residual/base ratio is unsafe: {ratio}")
    base_kl = float(latest["train/option/base_kl_mean"])
    if base_kl > 0.10:
        raise SystemExit(f"[FAIL] option/base KL is unsafe: {base_kl}")
    beta = float(latest["train/option/eligible_learned_beta_mean"])
    if not 0.0 <= beta <= 1.0:
        raise SystemExit(f"[FAIL] invalid learned termination mean: {beta}")
    if float(latest["train/option/legacy_behavior_losses_disabled"]) != 1.0:
        raise SystemExit("[FAIL] ordinary Dreamer behavior losses are still active")
    real_term = float(latest["train/option/real_eligible_termination_rate"])
    if not 0.0 <= real_term <= 1.0:
        raise SystemExit(f"[FAIL] invalid replay termination rate: {real_term}")

    # Manager PG is full by 100k. Learned termination intentionally warms up
    # longer and reaches full blend at 300k, after the worker and critic mature.
    if latest_step >= 100_000 and float(
        latest["train/option/manager_pg_blend"]
    ) < 0.999:
        raise SystemExit("[FAIL] manager task-PG ramp did not finish by 100k")
    if latest_step <= 100_000 and float(
        latest["train/option/termination_blend"]
    ) > 1.0e-6:
        raise SystemExit("[FAIL] learned termination activated during warm-up")
    if latest_step >= 300_000 and float(
        latest["train/option/termination_blend"]
    ) < 0.999:
        raise SystemExit("[FAIL] learned-termination ramp did not finish by 300k")

    active_cap = float(latest["train/option/termination_execution_cap"])
    if latest_step <= 300_000:
        expected_cap = 0.30
    elif latest_step >= 500_000:
        expected_cap = 0.80
    else:
        expected_cap = 0.30 + (latest_step - 300_000) / 200_000 * 0.50
    if active_cap > expected_cap + 1.0e-3:
        raise SystemExit(
            f"[FAIL] termination execution cap relaxed too early: "
            f"{active_cap} > {expected_cap}"
        )

    warnings = []
    effective = float(latest["train/option/effective_count"])
    if effective < 1.5:
        warnings.append(f"manager effective option count is low: {effective:.3f}")
    if ratio > 0.25:
        warnings.append(f"residual/base ratio exceeds target: {ratio:.3f}")
    if base_kl > 0.02:
        warnings.append(f"option/base KL exceeds target: {base_kl:.4f}")
    if real_term > 0.90:
        warnings.append(f"eligible termination rate is very high: {real_term:.3f}")
    if beta < 0.01 or beta > 0.90:
        warnings.append(f"learned beta is near saturation: {beta:.3f}")
    covered = sum(float(latest[f"train/option/imag_weight_{i}"]) > 0.0 for i in range(8))
    if covered < 4:
        warnings.append(f"only {covered}/8 options received imagined task weight in latest batch")

    print(f"[OK] Option-Critic metrics are finite; latest_step={latest_step}")
    print("[OK] all real and imagined state-machine violation rates are zero")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for key in required:
        print(f"{key}={latest[key]}")


if __name__ == "__main__":
    main()
