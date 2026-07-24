#!/usr/bin/env python3
"""Fail unless a smoke run activated both priority mechanisms."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(item, dict):
                out.update(flatten(item, name))
            else:
                out[name] = item
    return out


def as_finite_float(value: Any, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[FAIL] metric {key!r} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"[FAIL] metric {key!r} is non-finite: {number!r}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--require-nonuniform", action="store_true")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                flat = flatten(obj)
                if any(str(key).startswith("priority/") for key in flat):
                    rows.append(flat)

    if not rows:
        raise SystemExit(
            f"[FAIL] no priority metrics found in JSONL files under {run_dir}"
        )

    latest: dict[str, Any] = {}
    seen: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if not key.startswith("priority/"):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                raise SystemExit(f"[FAIL] non-finite priority metric {key}={number}")
            latest[key] = number
            seen.setdefault(key, []).append(number)

    required = [
        "priority/critic_error_mean",
        "priority/sequence_priority_mean",
        "priority/sequence_priority_max",
        "priority/sequence_raw_mean",
        "priority/is_weight_mean",
        "priority/is_weight_min",
        "priority/effective_sample_size",
        "priority/map_entropy",
        "priority/map_probability_min",
        "priority/map_probability_max",
        "priority/maps_with_feedback",
    ]
    missing = [key for key in required if key not in latest]
    if missing:
        raise SystemExit(f"[FAIL] missing priority metrics: {missing}")

    for key in required:
        as_finite_float(latest[key], key)

    if max(seen["priority/critic_error_mean"]) <= 0:
        raise SystemExit("[FAIL] critic-error priority signal never became positive")
    if max(seen["priority/sequence_priority_max"]) <= 0:
        raise SystemExit("[FAIL] sequence priorities never became positive")
    if latest["priority/maps_with_feedback"] <= 0:
        raise SystemExit("[FAIL] no training map received critic feedback")
    if latest["priority/effective_sample_size"] <= 0:
        raise SystemExit("[FAIL] invalid sequence-PER effective sample size")
    if not (0 < latest["priority/is_weight_min"] <= 1.0 + 1e-6):
        raise SystemExit("[FAIL] importance weights are outside (0, 1]")
    if not (0 < latest["priority/is_weight_mean"] <= 1.0 + 1e-6):
        raise SystemExit("[FAIL] mean importance weight is outside (0, 1]")
    if latest["priority/map_probability_min"] < 0:
        raise SystemExit("[FAIL] negative map probability")
    if latest["priority/map_probability_max"] > 1.0 + 1e-9:
        raise SystemExit("[FAIL] map probability exceeds one")

    nonuniform_sequence = min(seen["priority/is_weight_min"]) < 0.999999
    nonuniform_map = any(
        hi - lo > 1e-12
        for hi, lo in zip(
            seen["priority/map_probability_max"],
            seen["priority/map_probability_min"],
        )
    )
    if args.require_nonuniform and not nonuniform_sequence:
        raise SystemExit(
            "[FAIL] sequence PER never produced non-uniform importance weights"
        )
    if args.require_nonuniform and not nonuniform_map:
        raise SystemExit("[FAIL] adaptive map probabilities never became non-uniform")

    report = {
        "run_dir": str(run_dir),
        "priority_metric_rows": len(rows),
        "nonuniform_sequence_per_observed": nonuniform_sequence,
        "nonuniform_map_distribution_observed": nonuniform_map,
        "latest": {key: latest[key] for key in required},
    }
    print(json.dumps(report, indent=2))
    print("[OK] both unified priority mechanisms emitted finite active metrics")


if __name__ == "__main__":
    main()
