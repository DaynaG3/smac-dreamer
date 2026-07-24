#!/usr/bin/env python3
"""Inspect hardened tactic metrics and flag mechanism failures.

The script deliberately distinguishes marginal usage from state-dependent
specialisation. Four usage curves near 0.25 are not treated as evidence that
four learned tactics exist.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib


REQUIRED = {
    "train/tactic/effective_count",
    "train/tactic/usage_max",
    "train/tactic/conditional_entropy",
    "train/tactic/marginal_entropy",
    "train/tactic/mutual_information",
    "train/tactic/mutual_information_normalized",
    "train/tactic/selector_max_probability",
    "train/tactic/selector_logit_std",
    "train/tactic/effect_js",
    "train/tactic/effect_js_min",
    "train/tactic/effect_js_max",
    "train/tactic/residual_rms",
    "train/tactic/residual_to_base_ratio",
    "train/tactic/residual_guard_loss",
}
for index in range(4):
    REQUIRED.update(
        {
            f"train/tactic/usage_{index}",
            f"train/tactic/sampled_usage_{index}",
            f"train/tactic/argmax_usage_{index}",
            f"train/tactic/residual_rms_{index}",
        }
    )


def numeric_values(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) != len(values):
        raise SystemExit(f"[FAIL] non-finite metric: {key}")
    return finite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tail", type=int, default=200)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    path = pathlib.Path(args.run_dir) / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"[FAIL] missing metrics file: {path}")

    tactic_rows = []
    val_rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(key.startswith("train/tactic/") for key in row):
                tactic_rows.append(row)
            if "val/macro_win_rate" in row or "val/micro_win_rate" in row:
                val_rows.append(row)
    tactic_rows = tactic_rows[-args.tail :]
    if not tactic_rows:
        raise SystemExit("[FAIL] no train/tactic/* metrics found")

    present = {key for row in tactic_rows for key in row}
    missing = sorted(REQUIRED - present)
    if missing:
        raise SystemExit(f"[FAIL] hardened tactic metrics missing: {missing}")

    for key in sorted(key for key in present if key.startswith("train/tactic/")):
        values = numeric_values(tactic_rows, key)
        if not values:
            continue
        print(
            f"{key}: latest={values[-1]:.6g} "
            f"mean={sum(values)/len(values):.6g}"
        )

    latest = tactic_rows[-1]
    warnings = []
    effective = float(latest["train/tactic/effective_count"])
    usage_max = float(latest["train/tactic/usage_max"])
    mi = float(latest["train/tactic/mutual_information_normalized"])
    effect = float(latest["train/tactic/effect_js"])
    effect_min = float(latest["train/tactic/effect_js_min"])
    ratio = float(latest["train/tactic/residual_to_base_ratio"])
    selector_max = float(latest["train/tactic/selector_max_probability"])
    selector_std = float(latest["train/tactic/selector_logit_std"])

    if effective < 1.5 or usage_max > 0.90:
        warnings.append("selector appears globally collapsed")
    if mi < 0.02:
        warnings.append(
            "selector shows little state-dependent specialization; balanced "
            "usage alone does not demonstrate learned tactics"
        )
    if selector_max < 0.30 and selector_std < 0.05:
        warnings.append("selector remains close to the initial uniform policy")
    if effect < 1.0e-5 or effect_min < 1.0e-7:
        warnings.append("one or more tactics have nearly identical action distributions")
    if ratio > 1.0:
        warnings.append("tactical residual is larger than inherited base logits")

    argmax_usage = [
        float(latest[f"train/tactic/argmax_usage_{index}"])
        for index in range(4)
    ]
    if max(argmax_usage) > 0.95:
        warnings.append(
            "deterministic argmax selector is effectively one tactic even though "
            "marginal probabilities may look balanced"
        )

    if val_rows:
        print("validation checkpoints:")
        for row in val_rows[-5:]:
            step = row.get("global_step", row.get("step", "?"))
            macro = row.get("val/macro_win_rate")
            micro = row.get("val/micro_win_rate")
            print(f"  step={step} macro={macro} micro={micro}")
        micros = [
            float(row["val/micro_win_rate"])
            for row in val_rows
            if isinstance(row.get("val/micro_win_rate"), (int, float))
        ]
        if len(micros) >= 3 and micros[-1] < micros[0] - 0.03:
            warnings.append("held-out micro win rate has dropped by more than 3 points")

    if warnings:
        print("[WARN] mechanism/policy diagnostics:")
        for warning in warnings:
            print(f"  - {warning}")
        if args.fail_on_warning:
            raise SystemExit(1)
    else:
        print("[OK] hardened tactical mechanism metrics are active and finite")


if __name__ == "__main__":
    main()
