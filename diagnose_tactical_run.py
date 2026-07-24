#!/usr/bin/env python3
"""
Read-only diagnosis for a running Tactical Mixture hardened run.

Usage:
  python diagnose_tactical_run.py \
    --run /path/to/run \
    --source-checkpoint /path/to/source/best_val_macro_winrate.pt
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:
    torch = None


GROUPS: dict[str, list[str]] = {
    "validation": [
        "val/macro_win_rate",
        "val/micro_win_rate",
        "val/macro_original_return",
        "val/micro_original_return",
        "val/macro_timeout_rate",
        "val/micro_timeout_rate",
        "val/macro_final_ally_ehp_frac",
        "val/micro_final_ally_ehp_frac",
        "val/macro_final_enemy_ehp_frac",
        "val/micro_final_enemy_ehp_frac",
    ],
    "selector": [
        "train/tactic/selector_logit_std",
        "train/tactic/selector_max_probability",
        "train/tactic/entropy",
        "train/tactic/entropy_normalized",
        "train/tactic/conditional_entropy",
        "train/tactic/marginal_entropy",
        "train/tactic/selector_mi",
        "train/tactic/selector_mi_normalized",
        "train/tactic/effective_count",
        "train/tactic/usage_max",
    ],
    "usage": [
        *[f"train/tactic/sampled_usage_{i}" for i in range(4)],
        *[f"train/tactic/argmax_usage_{i}" for i in range(4)],
        *[f"train/tactic/usage_{i}" for i in range(4)],
    ],
    "effect": [
        "train/tactic/effect_js",
        "train/tactic/effect_js_mean",
        "train/tactic/effect_js_min",
        "train/tactic/effect_js_max",
        "train/tactic/effect_loss",
        "train/tactic/residual_rms",
        "train/tactic/residual_to_base_ratio",
        *[f"train/tactic/residual_rms_{i}" for i in range(4)],
        "train/tactic/residual_guard_loss",
        "train/tactic/collapse_guard_loss",
        "train/tactic/policy_loss",
        "train/tactic/eval_gate_mean",
        "train/tactic/base_kl_mean",
        "train/tactic/base_kl_max",
        "train/tactic/action_flip_rate",
    ],
    "policy_and_model": [
        "train/action_entropy",
        "train/rew",
        "train/tar",
        "train/val",
        "train/slowval",
        "train/ret",
        "train/ret_005",
        "train/ret_095",
        "train/adv",
        "train/ret_replay_mean",
        "train/value_replay_mean",
        "train/slow_value_replay_mean",
        "train/loss/policy",
        "train/loss/value",
        "train/loss/repval",
        "train/loss/rew",
        "train/loss/con",
    ],
    "mask_and_jepa": [
        "train/real_pre_mask_invalid_mass",
        "train/real_post_mask_invalid_sample_rate",
        "train/imag_pre_mask_invalid_mass",
        "train/imag_pre_mask_invalid_sample_rate",
        "train/imag_post_mask_invalid_sample_rate",
        "train/imag_empty_mask_rate",
        "train/mask_precision",
        "train/mask_recall",
        "train/mask_fpr",
        "train/jepa/latent_norm",
        "train/jepa/latent_std",
        "train/jepa/memory_norm",
        "train/jepa/feature_norm",
        "train/jepa/presence_rate",
        "train/jepa/predicted_entity_count",
        "train/jepa/trainable_adapter_parameter_count",
    ],
}


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_series(path: Path) -> tuple[dict[str, list[tuple[int, float]]], int]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    lines = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            lines += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            step_raw = row.get("global_step", row.get("step", -1))
            try:
                step = int(step_raw)
            except Exception:
                step = -1

            for key, value in row.items():
                if finite_number(value):
                    series[key].append((step, float(value)))

    return series, lines


def summarize(values: list[tuple[int, float]]) -> str:
    if not values:
        return "MISSING"

    latest_step, latest = values[-1]
    last_20 = [value for _, value in values[-20:]]
    last_100 = [value for _, value in values[-100:]]

    return (
        f"latest={latest:.6g}@{latest_step}  "
        f"mean20={statistics.fmean(last_20):.6g}  "
        f"mean100={statistics.fmean(last_100):.6g}  "
        f"min100={min(last_100):.6g}  max100={max(last_100):.6g}"
    )


def latest(series: dict[str, list[tuple[int, float]]], key: str) -> float | None:
    values = series.get(key)
    return values[-1][1] if values else None


def latest_any(
    series: dict[str, list[tuple[int, float]]],
    keys: list[str],
) -> tuple[str | None, float | None]:
    for key in keys:
        value = latest(series, key)
        if value is not None:
            return key, value
    return None, None


def read_checkpoint(path: Path) -> dict[str, Any]:
    if torch is None or not path.is_file():
        return {}

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        return {"error": repr(exc)}

    state = checkpoint.get("agent_state_dict", {})
    tactical_keys = [
        key for key in state
        if "tactical" in key.lower()
    ]

    return {
        "step": checkpoint.get("step"),
        "val_macro_win_rate": checkpoint.get("val_macro_win_rate"),
        "val_macro_original_return": checkpoint.get(
            "val_macro_original_return"
        ),
        "has_training_state": bool(
            checkpoint.get("agent_training_state")
        ),
        "has_tactical_metadata": bool(
            checkpoint.get("tactical_mixture_metadata")
        ),
        "n_tactical_state_keys": len(tactical_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    args = parser.parse_args()

    run = args.run.resolve()
    metrics_path = run / "metrics.jsonl"

    if not metrics_path.is_file():
        raise SystemExit(f"[FAIL] missing metrics file: {metrics_path}")

    series, lines = load_series(metrics_path)
    print(f"RUN={run}")
    print(f"METRIC_LINES={lines}")
    print(f"METRIC_KEYS={len(series)}")

    for group, keys in GROUPS.items():
        print(f"\n=== {group.upper()} ===")
        found = False
        for key in keys:
            if key in series:
                found = True
                print(f"{key}: {summarize(series[key])}")
        if not found:
            print("(none of the expected keys were present)")

    validation = series.get("val/macro_win_rate", [])
    if validation:
        print("\n=== VALIDATION HISTORY ===")
        for step, value in validation:
            print(f"step={step} macro_win_rate={value:.6f}")

    source_win = None
    if args.source_checkpoint:
        source_info = read_checkpoint(args.source_checkpoint.resolve())
        print("\n=== SOURCE CHECKPOINT ===")
        print(json.dumps(source_info, indent=2, default=str))
        if finite_number(source_info.get("val_macro_win_rate")):
            source_win = float(source_info["val_macro_win_rate"])

    print("\n=== AUTOMATIC DIAGNOSIS ===")
    findings: list[str] = []
    severity = 0

    post_invalid_key, post_invalid = latest_any(
        series,
        [
            "train/imag_post_mask_invalid_sample_rate",
            "train/real_post_mask_invalid_sample_rate",
        ],
    )
    if post_invalid is not None and post_invalid > 1e-6:
        findings.append(
            f"STOP: {post_invalid_key}={post_invalid:.6g} is nonzero."
        )
        severity = max(severity, 3)

    empty_mask = latest(series, "train/imag_empty_mask_rate")
    if empty_mask is not None and empty_mask > 0.05:
        findings.append(
            f"STOP: imag_empty_mask_rate={empty_mask:.4f} is too high."
        )
        severity = max(severity, 3)

    effective = latest(series, "train/tactic/effective_count")
    usage_max = latest(series, "train/tactic/usage_max")
    if effective is not None and effective < 1.2:
        findings.append(
            f"STOP: effective tactic count collapsed to {effective:.3f}."
        )
        severity = max(severity, 3)
    if usage_max is not None and usage_max > 0.97:
        findings.append(
            f"STOP: one tactic dominates with usage_max={usage_max:.3f}."
        )
        severity = max(severity, 3)

    ratio = latest(series, "train/tactic/residual_to_base_ratio")
    if ratio is not None:
        if ratio > 1.5:
            findings.append(
                f"STOP: tactical residual/base ratio={ratio:.3f}."
            )
            severity = max(severity, 3)
        elif ratio > 0.75:
            findings.append(
                f"CAUTION: tactical residual/base ratio={ratio:.3f} is large."
            )
            severity = max(severity, 2)

    mi_key, mi = latest_any(
        series,
        [
            "train/tactic/selector_mi_normalized",
            "train/tactic/selector_mi",
        ],
    )
    effect_key, effect = latest_any(
        series,
        [
            "train/tactic/effect_js_mean",
            "train/tactic/effect_js",
        ],
    )
    selector_std = latest(series, "train/tactic/selector_logit_std")

    if selector_std is not None and selector_std < 0.02:
        findings.append(
            f"CAUTION: selector_logit_std={selector_std:.4g}; "
            "selector is nearly state-invariant."
        )
        severity = max(severity, 2)

    if mi is not None and mi < 0.01:
        findings.append(
            f"CAUTION: {mi_key}={mi:.4g}; tactics have little "
            "state-dependent information."
        )
        severity = max(severity, 2)

    if effect is not None and effect < 1e-6:
        findings.append(
            f"CAUTION: {effect_key}={effect:.4g}; tactics produce "
            "nearly identical action policies."
        )
        severity = max(severity, 2)

    if validation and source_win is not None:
        last_val = validation[-1][1]
        gap = last_val - source_win
        findings.append(
            f"Latest macro validation gap vs source: {gap:+.4f} "
            f"({last_val:.4f} vs {source_win:.4f})."
        )

        if len(validation) >= 2:
            last_two = [value for _, value in validation[-2:]]
            if all(value <= source_win - 0.05 for value in last_two):
                findings.append(
                    "STOP: the last two validations are at least "
                    "5 percentage points below the source."
                )
                severity = max(severity, 3)
            elif all(value < source_win for value in last_two):
                findings.append(
                    "CAUTION: the last two validations are below source."
                )
                severity = max(severity, 2)

    imagined_rew = latest(series, "train/rew")
    replay_ret = latest(series, "train/ret_replay_mean")
    if imagined_rew is not None and replay_ret is not None:
        findings.append(
            "Imagination/replay snapshot: "
            f"imagined_rew={imagined_rew:.4g}, "
            f"replay_return={replay_ret:.4g}."
        )

    if not findings:
        findings.append(
            "No explicit failure threshold was triggered from available metrics."
        )

    for finding in findings:
        print(f"- {finding}")

    verdict = {
        0: "SAFE_TO_CONTINUE",
        1: "SAFE_TO_CONTINUE",
        2: "CONTINUE_WITH_CAUTION",
        3: "STOP_AND_PATCH",
    }[severity]

    print(f"\nVERDICT={verdict}")

    latest_ckpt = run / "latest.pt"
    best_ckpt = run / "best_val_macro_winrate.pt"
    print("\n=== RUN CHECKPOINTS ===")
    print("latest.pt:", json.dumps(read_checkpoint(latest_ckpt), indent=2))
    print(
        "best_val_macro_winrate.pt:",
        json.dumps(read_checkpoint(best_ckpt), indent=2),
    )

    print("\n=== ALL TACTICAL KEYS PRESENT ===")
    for key in sorted(k for k in series if "tactic" in k):
        print(key)


if __name__ == "__main__":
    main()
