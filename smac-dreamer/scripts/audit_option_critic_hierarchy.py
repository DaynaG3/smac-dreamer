#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_tokens(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        fail(f"{label} missing contracts: {missing}")


def method_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        fail(f"expected exactly one {name}() method, found {len(nodes)}")
    lines = text.splitlines(keepends=True)
    node = nodes[0]
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-meta", type=Path, required=True)
    parser.add_argument("--require-v1-2-source", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo / config_path).resolve()
    checkpoint_path = args.checkpoint.resolve()
    meta_path = args.source_run_meta.resolve()
    for path in (config_path, checkpoint_path, meta_path):
        if not path.is_file():
            fail(f"missing required path: {path}")

    cfg = OmegaConf.load(config_path)
    h = cfg.hierarchical_options
    checks = {
        "world_model_jepa": str(cfg.world_model.backend) == "jepa",
        "dense_reward_unchanged": str(cfg.reward.name) == "dense_v3",
        "hierarchy_enabled": bool(h.enabled),
        "eight_option_capacity": int(h.num_options) == 8,
        "trajectory_preserving_duration": (
            int(h.min_duration) == 1 and int(h.max_duration) == 20
        ),
        "commitment_schedule": (
            int(h.commitment_warmup_steps) == 100_000
            and int(h.commitment_full_steps) == 300_000
            and float(h.commitment_reselect_initial) == 1.0
            and float(h.commitment_reselect_final) == 0.0
        ),
        "source_preserving_manager": (
            float(h.manager_unimix_initial) == 0.0
            and float(h.manager_unimix_final) == 0.02
            and int(h.manager_pg_warmup_steps) == 100_000
            and int(h.manager_pg_full_steps) == 300_000
        ),
        "source_preserving_worker": (
            float(h.worker_scale_initial) == 0.25
            and float(h.worker_scale_max) == 0.25
            and int(h.worker_pg_warmup_steps) == 100_000
            and int(h.worker_pg_full_steps) == 300_000
        ),
        "smooth_termination": (
            int(h.termination_warmup_steps) == 100_000
            and int(h.termination_full_steps) == 300_000
            and float(h.termination_unimix) == 0.02
            and float(h.termination_collapse_scale) == 0.0
        ),
        "task_agnostic_regularizers_disabled": (
            float(h.manager_collapse_scale) == 0.0
            and float(h.manager_mi_scale) == 0.0
            and float(h.action_diversity_scale) == 0.0
            and float(h.residual_cosine_scale) == 0.0
        ),
        "full_source_trust_region": (
            float(h.base_kl_target) == 0.01
            and float(h.base_kl_tail_target) == 0.03
            and float(h.base_kl_tail_fraction) == 0.10
            and float(h.action_preservation_confidence) == 0.80
            and float(h.action_preservation_scale) > 0.0
        ),
        "world_model_exactly_frozen": (
            float(h.world_model_grad_scale_initial) == 0.0
            and float(h.world_model_grad_scale_final) == 0.0
        ),
        "variable_horizon": (
            int(h.imag_horizon_initial_max) == 8
            and int(h.imag_horizon_final_max) == 10
            and int(h.imag_horizon_window) == 4
        ),
        "startup_validation": (
            bool(cfg.validation.run_at_start)
            and int(cfg.validation.every) == 100_000
        ),
        "fresh_uniform_replay": (
            str(cfg.sampling_mode) == "shuffled_round_robin"
            and not bool(cfg.adaptive_priority.enabled)
            and not bool(cfg.adaptive_priority.map.enabled)
            and not bool(cfg.adaptive_priority.sequence.enabled)
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail(f"configuration contracts failed: {failed}")

    paths = {
        "options": repo / "external/r2dreamer/hierarchical_options.py",
        "hierarchy": repo / "external/r2dreamer/hierarchical_dreamer.py",
        "dreamer": repo / "external/r2dreamer/dreamer.py",
        "trainer": repo / "external/r2dreamer/trainer.py",
        "runner": repo / "scripts/train_r2dreamer_smaclite_multimap.py",
        "validation": repo / "src/smacdreamer/validation_trainer.py",
        "tools": repo / "external/r2dreamer/tools.py",
    }
    for path in paths.values():
        if not path.is_file():
            fail(f"missing installed source: {path}")
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    require_tokens(text["options"], (
        'ARCHITECTURE = "dreamer_option_critic_v3_p0p1"',
        "commitment_reselect_probability",
        "bounded_learned_termination_probability",
        "next_imagination_horizon",
        "base_kl_tail",
        "action_preservation_loss",
    ), "hierarchical_options")
    if ".clamp(\n            max=self.termination_probability_cap" in text["options"]:
        fail("hard termination clamp remains in execution path")
    require_tokens(text["hierarchy"], (
        "apply_hierarchy_gradient_guards",
        "_source_hierarchical_options",
        "start_valid = (1.0 - raw_data[\"is_last\"].float())",
        "next_imagination_horizon()",
        "four_exact_copies_per_v1_2_mode",
        "temperature = 1.0",
        's.action_preservation_scale * behavior["action_preservation_loss"]',
    ), "hierarchical_dreamer")
    require_tokens(text["dreamer"], (
        "# OPTION_CRITIC_P0P1_HOTFIX_V3",
        "apply_hierarchy_gradient_guards",
        "apply_hierarchy_gradient_guards(self)",
    ), "dreamer integration")
    update = method_source(text["dreamer"], "update")
    guard_pos = update.find("apply_hierarchy_gradient_guards(self)")
    unscale_positions = []
    offset = 0
    for line in update.splitlines(keepends=True):
        stripped = line.split("#", 1)[0].strip()
        if (
            stripped.startswith("self._scaler.unscale_(")
            and stripped.endswith(")")
        ):
            unscale_positions.append(offset)
        offset += len(line)

    if len(unscale_positions) != 1:
        fail(
            "expected exactly one scaler unscale call in update(), found "
            f"{len(unscale_positions)}"
        )

    unscale_pos = unscale_positions[0]

    if guard_pos < 0 or guard_pos > unscale_pos:
        fail("world-model optimizer guard is not before optimizer unscale/step")

    grad_method = method_source(text["dreamer"], "_cal_grad_jepa")
    legacy = grad_method.find("losses.pop(legacy_key, None)")
    total = grad_method.find("total_loss = sum([v * self._loss_scales[k]")
    if legacy < 0 or total < 0 or legacy > total:
        fail("legacy behavior losses are not disabled before total-loss construction")

    require_tokens(text["trainer"], (
        'trans[option_key] = agent_state[option_key]',
        "agent.set_hierarchy_training_step(step)",
        "agent.set_hierarchy_training_step(train_step)",
    ), "trainer")
    require_tokens(text["runner"], (
        "config.model.hierarchical_options",
        "load_hierarchical_compatible_state_dict",
        '"hierarchical_options_metadata"',
    ), "runner")
    require_tokens(text["validation"], (
        'best_payload["hierarchical_options_metadata"]',
        "agent.set_hierarchy_training_step(train_step)",
    ), "validation")
    require_tokens(text["tools"], (
        "torch.bfloat16",
        "x = x.float()",
        "return x.cpu().numpy()",
    ), "BF16 logging guard")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("agent_state_dict")
    if not isinstance(state, dict):
        fail("source checkpoint has no agent_state_dict")
    tactical_meta = ckpt.get("tactical_mixture_metadata") or {}
    if args.require_v1_2_source:
        if tactical_meta.get("architecture") != "tactical_mixture_v1_2":
            fail(f"source architecture is {tactical_meta.get('architecture')!r}")
        if int(tactical_meta.get("num_tactics", -1)) != 2:
            fail("source checkpoint is not Tactical Mixture v1.2 with two modes")
        if not any(key.startswith("tactical_policy.") for key in state):
            fail("source lacks tactical_policy keys")
        if any(key.startswith("hierarchical_options.") for key in state):
            fail("source already contains hierarchy keys")
        source_win = float(ckpt.get("val_macro_win_rate", -1.0))
        if source_win < 0.3749:
            fail(f"source macro win rate {source_win} is below 0.375")
    if checkpoint_path.parent != meta_path.parent:
        fail("checkpoint and run_meta are not from the same run")
    json.loads(meta_path.read_text(encoding="utf-8"))

    report = {
        "repo": str(repo),
        "config": str(config_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "step": ckpt.get("step"),
            "val_macro_win_rate": ckpt.get("val_macro_win_rate"),
            "val_macro_original_return": ckpt.get("val_macro_original_return"),
        },
        "checks": {**checks, "source_ast": True, "migration_lineage": True},
    }
    print("[OK] Option-Critic P0/P1 source/config audit passed")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
