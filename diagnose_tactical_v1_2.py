#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


KEYS = [
    # Validation
    "val/macro_win_rate",
    "val/micro_win_rate",
    "val/macro_original_return",
    "val/micro_original_return",
    "val/n_episodes",
    "val/n_maps",
    "val/macro_timeout_rate",
    "val/macro_final_ally_ehp_frac",
    "val/macro_final_enemy_ehp_frac",
    # Selector
    "train/tactic/selector_max_probability",
    "train/tactic/selector_logit_std",
    "train/tactic/conditional_entropy",
    "train/tactic/marginal_entropy",
    "train/tactic/mutual_information",
    "train/tactic/mutual_information_normalized",
    "train/tactic/mi_shortfall",
    "train/tactic/effective_count",
    "train/tactic/usage_max",
    "train/tactic/sampled_usage_0",
    "train/tactic/sampled_usage_1",
    "train/tactic/argmax_usage_0",
    "train/tactic/argmax_usage_1",
    # Effect / trust region
    "train/tactic/effect_js",
    "train/tactic/effect_js_min",
    "train/tactic/effect_js_max",
    "train/tactic/effect_loss",
    "train/tactic/residual_rms",
    "train/tactic/residual_rms_0",
    "train/tactic/residual_rms_1",
    "train/tactic/residual_to_base_ratio",
    "train/tactic/residual_guard_loss",
    "train/tactic/base_kl_mean",
    "train/tactic/base_kl_max",
    "train/tactic/base_kl_loss",
    "train/tactic/action_flip_rate",
    "train/tactic/collapse_loss",
    "train/tactic/policy_loss",
    # RL/model
    "train/action_entropy",
    "train/rew",
    "train/tar",
    "train/val",
    "train/slowval",
    "train/ret_replay_mean",
    "train/value_replay_mean",
    "train/slow_value_replay_mean",
    # Masks / frozen adapter
    "train/real_pre_mask_invalid_mass",
    "train/real_post_mask_invalid_sample_rate",
    "train/imag_pre_mask_invalid_mass",
    "train/imag_post_mask_invalid_sample_rate",
    "train/imag_empty_mask_rate",
    "train/jepa/trainable_adapter_parameter_count",
]


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_metrics(path: Path):
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("global_step", row.get("step", -1))
            try:
                step = int(step)
            except Exception:
                step = -1
            for key, value in row.items():
                if finite(value):
                    series[key].append((step, float(value)))
    return series


def latest(series, key):
    vals = series.get(key, [])
    return vals[-1][1] if vals else None


def mean_tail(series, key, n=100):
    vals = [v for _, v in series.get(key, [])[-n:]]
    return statistics.fmean(vals) if vals else None


def fmt(x):
    return "MISSING" if x is None else f"{x:.8g}"


def wilson(p: float, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return None
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    ) / denom
    return center - radius, center + radius


def ckpt_info(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "step": ckpt.get("step"),
        "val_macro_win_rate": ckpt.get("val_macro_win_rate"),
        "val_macro_original_return": ckpt.get("val_macro_original_return"),
        "has_tactical_metadata": bool(ckpt.get("tactical_mixture_metadata")),
        "has_training_state": bool(ckpt.get("agent_training_state")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--eval-threshold", type=float, default=0.70)
    args = ap.parse_args()

    run = args.run.resolve()
    metrics = run / "metrics.jsonl"
    if not metrics.is_file():
        raise SystemExit(f"[FAIL] missing {metrics}")

    series = load_metrics(metrics)
    source = ckpt_info(args.source_checkpoint.resolve())

    print("RUN:", run)
    print("SOURCE:", args.source_checkpoint.resolve())
    print("SOURCE_INFO:", json.dumps(source, indent=2))

    print("\n=== VALIDATION HISTORY ===")
    val_steps = {}
    for key in (
        "val/macro_win_rate",
        "val/micro_win_rate",
        "val/macro_original_return",
        "val/n_episodes",
    ):
        for step, value in series.get(key, []):
            val_steps.setdefault(step, {})[key] = value
    for step in sorted(val_steps):
        print(step, json.dumps(val_steps[step], sort_keys=True))

    print("\n=== METRIC SUMMARY ===")
    for key in KEYS:
        if key in series:
            print(
                f"{key}: latest={fmt(latest(series, key))} "
                f"mean100={fmt(mean_tail(series, key, 100))}"
            )

    confidence = mean_tail(
        series, "train/tactic/selector_max_probability", 100
    )
    gate_est = None
    if confidence is not None:
        gate_est = max(
            0.0,
            min(1.0, (confidence - args.eval_threshold) /
                max(1.0 - args.eval_threshold, 1e-8)),
        )

    mi = mean_tail(
        series, "train/tactic/mutual_information_normalized", 100
    )
    effect = mean_tail(series, "train/tactic/effect_js", 100)
    ratio = mean_tail(
        series, "train/tactic/residual_to_base_ratio", 100
    )
    kl = mean_tail(series, "train/tactic/base_kl_mean", 100)
    flip = mean_tail(series, "train/tactic/action_flip_rate", 100)
    arg0 = mean_tail(series, "train/tactic/argmax_usage_0", 100)
    arg1 = mean_tail(series, "train/tactic/argmax_usage_1", 100)
    latest_macro = latest(series, "val/macro_win_rate")
    latest_micro = latest(series, "val/micro_win_rate")
    n_eps_raw = latest(series, "val/n_episodes")
    n_eps = int(n_eps_raw) if n_eps_raw is not None else 0
    source_win = source.get("val_macro_win_rate")

    print("\n=== INFERRED EVALUATION GATE ===")
    print("selector confidence mean100:", fmt(confidence))
    print("configured threshold:", args.eval_threshold)
    print("estimated gate on training-state distribution:", fmt(gate_est))
    print(
        "NOTE: this is an approximation from imagined training states; "
        "validation states may differ."
    )

    print("\n=== STATISTICAL CONTEXT ===")
    if latest_micro is not None and n_eps:
        interval = wilson(latest_micro, n_eps)
        print(
            f"latest micro={latest_micro:.6f}, n={n_eps}, "
            f"Wilson95=[{interval[0]:.6f}, {interval[1]:.6f}]"
        )
    else:
        print("Could not compute micro-win confidence interval.")
    if latest_macro is not None and finite(source_win):
        print(
            "latest macro gap vs source:",
            f"{latest_macro - float(source_win):+.6f}",
        )

    print("\n=== DIAGNOSIS ===")
    findings = []

    invalid = max(
        latest(series, "train/real_post_mask_invalid_sample_rate") or 0,
        latest(series, "train/imag_post_mask_invalid_sample_rate") or 0,
    )
    if invalid > 1e-6:
        findings.append(("BUG", f"post-mask invalid rate={invalid}"))

    if confidence is not None and confidence < args.eval_threshold:
        findings.append((
            "GATE_CLOSED",
            "selector confidence remains below 0.70, so deterministic "
            "validation likely uses the inherited actor exactly",
        ))

    if mi is not None and mi < 0.01:
        findings.append((
            "SELECTOR_WEAK",
            f"normalized selector MI is only {mi:.6g}",
        ))

    if effect is not None and effect < 1e-5:
        findings.append((
            "EFFECT_WEAK",
            f"pairwise action JS is only {effect:.6g}",
        ))

    if ratio is not None and ratio > 0.25:
        findings.append((
            "RESIDUAL_TOO_LARGE",
            f"residual/base ratio is {ratio:.4f}",
        ))

    if kl is not None and kl > 0.04:
        findings.append((
            "TRUST_REGION_TOO_LOOSE",
            f"base KL is {kl:.6g}",
        ))

    if arg0 is not None and arg1 is not None:
        if min(arg0, arg1) < 0.02:
            findings.append((
                "ARGMAX_COLLAPSE",
                f"argmax usages are {arg0:.4f}/{arg1:.4f}",
            ))

    if (
        confidence is not None
        and confidence >= args.eval_threshold
        and mi is not None and mi >= 0.03
        and effect is not None and effect >= 1e-4
        and flip is not None and flip > 0.01
        and latest_macro is not None and finite(source_win)
        and latest_macro <= float(source_win) + 0.005
    ):
        findings.append((
            "METHOD_PLATEAU_LIKELY",
            "tactical policy is active and state-dependent but has not "
            "improved held-out macro win rate",
        ))

    if not findings:
        findings.append((
            "NO_CLEAR_FAILURE",
            "no direct bug or conceptual bottleneck was established",
        ))

    for label, message in findings:
        print(f"{label}: {message}")

    labels = {label for label, _ in findings}
    if "BUG" in labels:
        verdict = "IMPLEMENTATION_BUG"
    elif "GATE_CLOSED" in labels:
        verdict = "NOT_A_METHOD_LIMIT_EVAL_GATE_IS_CLOSED"
    elif "METHOD_PLATEAU_LIKELY" in labels:
        verdict = "LIKELY_METHOD_LIMIT_OR_ONE_STEP_TACTIC_LIMIT"
    elif {"SELECTOR_WEAK", "EFFECT_WEAK", "ARGMAX_COLLAPSE"} & labels:
        verdict = "TACTICAL_MECHANISM_NOT_ACTIVATED"
    else:
        verdict = "INCONCLUSIVE"

    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
