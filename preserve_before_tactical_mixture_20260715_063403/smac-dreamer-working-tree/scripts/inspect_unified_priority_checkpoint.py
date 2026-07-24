#!/usr/bin/env python3
"""Inspect adaptive-priority checkpoint state and fail on invalid values."""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import torch


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--require-adaptive", action="store_true")
    parser.add_argument("--min-step", type=int, default=1)
    args = parser.parse_args()

    path = pathlib.Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    step = int(checkpoint.get("step", 0))
    if step < args.min_step:
        fail(f"checkpoint step {step} < required {args.min_step}")

    state = checkpoint.get("adaptive_priority_state")
    if state is None:
        if args.require_adaptive:
            fail("checkpoint has no adaptive_priority_state")
        print(json.dumps({"checkpoint": str(path), "step": step, "adaptive": False}, indent=2))
        return

    probabilities = torch.as_tensor(state.get("probabilities"), dtype=torch.float64)
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        fail("saved map probabilities are missing or malformed")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        fail("saved map probabilities contain invalid values")
    if abs(float(probabilities.sum()) - 1.0) > 1e-8:
        fail(f"saved map probabilities sum to {float(probabilities.sum())}, not 1")

    feedback = torch.as_tensor(state.get("feedback_count"), dtype=torch.int64)
    collections = torch.as_tensor(state.get("collection_count"), dtype=torch.int64)
    errors = torch.as_tensor(state.get("error_ema"), dtype=torch.float64)
    if not (feedback.shape == collections.shape == errors.shape == probabilities.shape):
        fail("adaptive map-state tensor shapes do not match")
    if not torch.isfinite(errors).all():
        fail("critic-error EMA contains non-finite values")

    entropy = float(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    report = {
        "checkpoint": str(path),
        "step": step,
        "adaptive": True,
        "schema": checkpoint.get("adaptive_priority_schema"),
        "map_count": int(probabilities.numel()),
        "probability_sum": float(probabilities.sum()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_entropy": entropy,
        "uniform_entropy": math.log(int(probabilities.numel())),
        "maps_with_feedback": int((feedback > 0).sum()),
        "maps_collected": int((collections > 0).sum()),
        "feedback_timesteps": int(feedback.sum()),
        "collection_episodes": int(collections.sum()),
        "critic_error_mean": float(errors.mean()),
        "critic_error_max": float(errors.max()),
    }
    print(json.dumps(report, indent=2))
    print("[OK] adaptive checkpoint state is finite and internally consistent")


if __name__ == "__main__":
    main()
