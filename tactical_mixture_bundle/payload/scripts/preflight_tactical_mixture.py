#!/usr/bin/env python3
"""Read-only preflight for Tactical Mixture Actor v1.

Checks source markers, tactical config isolation, legacy/tactical checkpoint
shape, and compatibility with the old run metadata when supplied.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
from typing import Any

import torch
from omegaconf import OmegaConf


REQUIRED = {
    "external/r2dreamer/dreamer.py": [
        "TACTICAL_MIXTURE_V1",
        "load_tactical_compatible_state_dict",
        "tactic/entropy",
        "_frozen_tactical_policy",
        "primitive_policy_loss",
    ],
    "external/r2dreamer/tactical_policy.py": [
        "class TacticalMixturePolicy",
        "assert_legacy_equivalence_ready",
        "effect_js",
    ],
    "scripts/train_r2dreamer_smaclite_multimap.py": [
        "TACTICAL_MIXTURE_V1",
        "tactical_mixture_metadata",
        "load_tactical_compatible_state_dict",
        "using original uniform SliceSampler",
    ],
}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(obj: Any, *keys: str, default=None):
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-run-meta")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    result = {
        "repo": str(repo),
        "checks": {},
        "checkpoint": None,
        "source_compatibility": None,
    }
    for relative, markers in REQUIRED.items():
        path = repo / relative
        if not path.is_file():
            fail(f"missing file: {relative}")
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        missing = [marker for marker in markers if marker not in text]
        if missing:
            fail(f"{relative} missing markers: {missing}")
        result["checks"][relative] = "ok"

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    cfg = OmegaConf.load(config_path)
    tactical = cfg.get("tactical_mixture")
    if tactical is None or not bool(tactical.get("enabled", False)):
        fail("tactical_mixture.enabled is not true")
    if int(tactical.get("duration", -1)) != 1:
        fail("Tactical v1 requires duration=1")
    if int(tactical.get("num_tactics", 0)) < 2:
        fail("num_tactics must be >= 2")

    adaptive = cfg.get("adaptive_priority") or {}
    if bool(adaptive.get("enabled", False)):
        fail("adaptive_priority.enabled must be false for the first tactical ablation")
    if bool((adaptive.get("map") or {}).get("enabled", False)):
        fail("adaptive map priority is still enabled")
    if bool((adaptive.get("sequence") or {}).get("enabled", False)):
        fail("sequence PER is still enabled")
    if str(cfg.get("sampling_mode", "")) == "adaptive_priority":
        fail("sampling_mode is still adaptive_priority")

    validation = cfg.get("validation") or {}
    if bool(validation.get("run_at_start", True)):
        fail("validation.run_at_start must be false")
    if int(validation.get("every", 0)) != 200000:
        fail("validation.every must be 200000")
    if str(nested_get(cfg, "world_model", "backend", default="")) != "jepa":
        fail("Tactical Mixture v1 is validated only for the JEPA backend")
    if int(cfg.get("imag_horizon", -1)) != 5:
        fail("first tactical experiment must preserve imag_horizon=5")
    result["checks"]["config"] = "ok"

    if args.source_run_meta:
        meta_path = pathlib.Path(args.source_run_meta).resolve()
        if not meta_path.is_file():
            fail(f"source run metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        comparisons = {
            "reward_name": (str(nested_get(cfg, "reward", "name", default="")), str(meta.get("reward_name", ""))),
            "imag_horizon": (int(cfg.get("imag_horizon", -1)), int(meta.get("imag_horizon", -2))),
            "max_episode_steps": (int(cfg.get("max_episode_steps", -1)), int(meta.get("max_episode_steps", -2))),
            "gamma": (float(cfg.get("gamma", float("nan"))), float(meta.get("gamma", float("nan")))),
            "world_model_backend": (str(nested_get(cfg, "world_model", "backend", default="")), str(meta.get("world_model_backend", ""))),
        }
        mismatches = {
            key: {"tactical_config": left, "source_run": right}
            for key, (left, right) in comparisons.items()
            if left != right
        }
        if mismatches:
            fail(
                "tactical config differs from source run in critical fields: "
                + json.dumps(mismatches, sort_keys=True)
            )
        jepa_path = pathlib.Path(
            str(nested_get(cfg, "world_model", "jepa", "checkpoint", default=""))
        ).expanduser()
        if not jepa_path.is_absolute():
            jepa_path = (repo / jepa_path).resolve()
        expected_hash = meta.get("jepa_checkpoint_sha256")
        if expected_hash:
            if not jepa_path.is_file():
                fail(f"JEPA checkpoint missing: {jepa_path}")
            actual_hash = sha256_file(jepa_path)
            if actual_hash != expected_hash:
                fail(
                    "JEPA checkpoint hash differs from source run: "
                    f"{actual_hash} != {expected_hash}"
                )
        result["source_compatibility"] = {
            "source_run_meta": str(meta_path),
            "critical_fields": "match",
            "jepa_checkpoint": str(jepa_path),
            "jepa_checkpoint_sha256": expected_hash,
        }

    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint).resolve()
        if not checkpoint_path.is_file():
            fail(f"checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if "agent_state_dict" not in checkpoint:
            fail("checkpoint lacks agent_state_dict")
        state = checkpoint["agent_state_dict"]
        has_tactical_keys = any(
            key.startswith("tactical_policy.") for key in state
        )
        metadata = checkpoint.get("tactical_mixture_metadata")
        if metadata is None and has_tactical_keys:
            fail("checkpoint has tactical keys but no tactical metadata")
        if (
            metadata is not None
            and metadata.get("architecture") != "tactical_mixture_v1"
        ):
            fail("unsupported tactical checkpoint architecture")
        result["checkpoint"] = {
            "path": str(checkpoint_path),
            "stored_step": int(checkpoint.get("step", 0)),
            "has_tactical_keys": has_tactical_keys,
            "has_tactical_metadata": metadata is not None,
            "has_training_state": bool(checkpoint.get("agent_training_state")),
            "has_rng_state": bool(checkpoint.get("rng_state")),
        }
        if metadata is None:
            print(
                "[OK] legacy checkpoint detected; tactical modules will be "
                "zero-initialized"
            )
            if checkpoint.get("agent_training_state"):
                print(
                    "[WARN] legacy optimizer state will be skipped because "
                    "parameter groups changed"
                )
        else:
            print(
                "[OK] tactical checkpoint detected; strict model and metadata "
                "compatibility required"
            )

    print("[OK] tactical-mixture preflight passed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
