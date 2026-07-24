#!/usr/bin/env python3
"""Read-only integration, config-compatibility, and checkpoint preflight."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import sys
from typing import Any

import torch


REQUIRED_MARKERS = {
    "src/smacdreamer/adaptive_priority.py": [
        "class AdaptivePriorityController",
        "shared_probabilities",
        "record_critic_feedback",
    ],
    "external/r2dreamer/adaptive_buffer.py": [
        "class AdaptiveBuffer",
        "candidate_multiplier",
        "importance_weights",
        "sequence_uids",
    ],
    "src/smacdreamer/envs/map_sampler.py": [
        "'adaptive_priority'",
        "shared_probabilities",
        "_adaptive_weights",
    ],
    "src/smacdreamer/r2dreamer_factory.py": [
        "shared_map_probabilities",
        "shared_map_version",
    ],
    "external/r2dreamer/trainer.py": [
        "start_step",
        "current_step",
        "record_collection",
        "set_env_step",
        "replay_buffer.count() // envs.env_num",
    ],
    "external/r2dreamer/dreamer.py": [
        "importance_weights",
        "_priority_sequence_priorities",
        "_priority_value = value_dist.mode()",
        "record_critic_feedback",
        "update_priorities",
    ],
    "src/smacdreamer/checkpointing.py": ["extra_state_fn"],
    "scripts/train_r2dreamer_smaclite_multimap.py": [
        "AdaptivePriorityController",
        "AdaptiveBuffer",
        "adaptive_priority_state",
        "start_step",
        "resume_start_step",
        "rng_state",
    ],
}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    node = cfg
    for key in dotted.split("."):
        if node is None:
            return default
        try:
            if key not in node:
                return default
            node = node[key]
        except (TypeError, KeyError):
            return default
    return node


def check_resume_compatibility(
    repo: pathlib.Path,
    config_path: pathlib.Path,
    run_meta_path: pathlib.Path,
    *,
    expected_validation_every: int,
) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        fail(f"OmegaConf is required for config compatibility checks: {exc}")

    if not config_path.exists():
        fail(f"resume config does not exist: {config_path}")
    if not run_meta_path.exists():
        fail(f"source run_meta.json does not exist: {run_meta_path}")

    cfg = OmegaConf.load(config_path)
    meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if str(_cfg_get(cfg, "sampling_mode", "")) != "adaptive_priority":
        fail("resume config sampling_mode must be adaptive_priority")
    if not bool(_cfg_get(cfg, "adaptive_priority.map.enabled", False)):
        fail("resume config must enable adaptive_priority.map")
    if not bool(_cfg_get(cfg, "adaptive_priority.sequence.enabled", False)):
        fail("resume config must enable adaptive_priority.sequence")

    validation_every = int(_cfg_get(cfg, "validation.every", -1))
    if validation_every != int(expected_validation_every):
        fail(
            "validation.every must be "
            f"{int(expected_validation_every):,}, got {validation_every:,}"
        )

    comparisons = {
        "obs_mode": (_cfg_get(cfg, "observation.mode"), meta.get("obs_mode")),
        "units": (_cfg_get(cfg, "units"), meta.get("units")),
        "deter": (_cfg_get(cfg, "deter"), meta.get("deter")),
        "batch_size": (_cfg_get(cfg, "batch_size"), meta.get("batch_size")),
        "batch_length": (_cfg_get(cfg, "batch_length"), meta.get("batch_length")),
        "imag_horizon": (_cfg_get(cfg, "imag_horizon"), meta.get("imag_horizon")),
        "max_episode_steps": (
            _cfg_get(cfg, "max_episode_steps"),
            meta.get("max_episode_steps"),
        ),
        "gamma": (_cfg_get(cfg, "gamma"), meta.get("gamma")),
        "reward_name": (_cfg_get(cfg, "reward.name"), meta.get("reward_name")),
        "world_model_backend": (
            _cfg_get(cfg, "world_model.backend"),
            meta.get("world_model_backend"),
        ),
    }
    mismatches = {}
    for name, (new, old) in comparisons.items():
        if old is None:
            continue
        if isinstance(old, float) or isinstance(new, float):
            try:
                equal = abs(float(new) - float(old)) <= 1e-12
            except (TypeError, ValueError):
                equal = new == old
        else:
            equal = new == old
        if not equal:
            mismatches[name] = {"resume_config": new, "source_run": old}
    if mismatches:
        fail(
            "resume config differs from the trained run in model/training-critical "
            f"fields: {json.dumps(mismatches, sort_keys=True)}"
        )

    expected_hash = meta.get("jepa_checkpoint_sha256")
    configured_jepa = _cfg_get(cfg, "world_model.jepa.checkpoint")
    actual_hash = None
    configured_path = None
    if configured_jepa:
        configured_path = pathlib.Path(str(configured_jepa)).expanduser()
        if not configured_path.is_absolute():
            configured_path = repo / configured_path
        if not configured_path.exists():
            fail(f"configured JEPA checkpoint does not exist: {configured_path}")
        actual_hash = sha256_file(configured_path)
        if expected_hash and actual_hash != expected_hash:
            fail(
                "configured JEPA checkpoint hash does not match the source run: "
                f"{actual_hash} != {expected_hash}"
            )

    return {
        "config": str(config_path),
        "source_run_meta": str(run_meta_path),
        "critical_fields": "match",
        "validation_every": validation_every,
        "jepa_checkpoint": str(configured_path) if configured_path else None,
        "jepa_checkpoint_sha256": actual_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--source-run-meta")
    parser.add_argument("--json-out")
    parser.add_argument("--allow-weights-only", action="store_true")
    parser.add_argument("--expected-validation-every", type=int, default=200000)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    if not (repo / "external/r2dreamer/dreamer.py").exists():
        fail(f"not an smac-dreamer repo root: {repo}")

    result: dict[str, Any] = {
        "repo": str(repo),
        "checks": {},
        "checkpoint": None,
        "resume_compatibility": None,
    }

    for relative, markers in REQUIRED_MARKERS.items():
        path = repo / relative
        if not path.exists():
            fail(f"missing integrated file: {relative}")
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        missing = [marker for marker in markers if marker not in text]
        if missing:
            fail(f"{relative} missing markers: {missing}")
        result["checks"][relative] = "ok"

    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "external/r2dreamer"))
    from smacdreamer.adaptive_priority import AdaptivePriorityController
    from adaptive_buffer import AdaptiveBuffer  # noqa: F401; import is a runtime dependency check

    class Entry:
        def __init__(self, mid, name):
            self.map_id = mid
            self.name = name

    controller_cfg = {
        "map": {
            "enabled": True,
            "error_ema_decay": 0.9,
            "uniform_floor": 0.1,
            "staleness_mix": 0.2,
            "update_every_feedbacks": 1,
            "minimum_feedback": 1,
        }
    }
    controller = AdaptivePriorityController.from_entries(
        [Entry(10, "a"), Entry(20, "b"), Entry(30, "c")], controller_cfg
    )
    controller.record_collection(
        torch.tensor([10, 20, 30]), torch.ones(3), env_step=100
    )
    controller.record_critic_feedback(
        torch.tensor([[10, 10], [20, 30]]),
        torch.tensor([[1.0, 1.0], [3.0, 0.2]]),
        torch.ones(2, 2),
        env_step=100,
    )
    probabilities = controller.shared_probabilities
    if (
        not torch.isfinite(probabilities).all()
        or abs(float(probabilities.sum()) - 1.0) > 1e-9
    ):
        fail("adaptive map probabilities are invalid")
    state = controller.state_dict()
    clone = AdaptivePriorityController.from_entries(
        [Entry(10, "a"), Entry(20, "b"), Entry(30, "c")], controller_cfg
    )
    clone.load_state_dict(state)
    if not torch.allclose(
        controller.shared_probabilities, clone.shared_probabilities
    ):
        fail("adaptive map state round-trip failed")
    result["checks"]["controller_runtime"] = "ok"

    checkpoint = None
    if args.checkpoint:
        checkpoint = pathlib.Path(args.checkpoint).resolve()
        if not checkpoint.exists():
            fail(f"checkpoint does not exist: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            fail("checkpoint is not a dict")
        if "agent_state_dict" not in ckpt:
            fail("checkpoint lacks agent_state_dict")
        step = int(ckpt.get("step", 0))
        has_training = bool(ckpt.get("agent_training_state"))
        has_optimizer_payload = bool(ckpt.get("optims_state_dict"))
        has_rng_state = bool(ckpt.get("rng_state"))
        result["checkpoint"] = {
            "path": str(checkpoint),
            "step": step,
            "has_training_state": has_training,
            "has_optimizer_payload": has_optimizer_payload,
            "has_rng_state": has_rng_state,
            "has_adaptive_state": "adaptive_priority_state" in ckpt,
        }
        print(
            f"[OK] checkpoint step={step:,} training_state={has_training} "
            f"adaptive_state={'adaptive_priority_state' in ckpt}"
        )
        if step <= 0:
            fail("checkpoint has no positive absolute 'step'")
        if not has_training and not args.allow_weights_only:
            fail(
                "checkpoint has no agent_training_state; this is not a faithful "
                "optimizer/scheduler/scaler continuation. Pass --allow-weights-only "
                "only when intentionally starting a new optimization run."
            )
        if not has_training:
            print("[WARN] weights-only continuation explicitly allowed")
        if not has_rng_state and not args.allow_weights_only:
            fail(
                "checkpoint has no rng_state; faithful stochastic continuation is "
                "not possible. Pass --allow-weights-only only when intentional."
            )
        if not has_rng_state:
            print("[WARN] missing RNG state explicitly allowed")

    if bool(args.config) != bool(args.source_run_meta):
        fail("pass --config and --source-run-meta together")
    if args.config and args.source_run_meta:
        config_path = pathlib.Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = repo / config_path
        run_meta_path = pathlib.Path(args.source_run_meta).expanduser().resolve()
        result["resume_compatibility"] = check_resume_compatibility(
            repo, config_path.resolve(), run_meta_path,
            expected_validation_every=args.expected_validation_every,
        )
        print("[OK] resume config matches the trained run's critical metadata")

    print("[OK] unified adaptive-priority static + runtime preflight passed")
    print(json.dumps(result, indent=2))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
