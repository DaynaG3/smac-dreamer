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


def close(value: object, expected: float, atol: float = 1e-12) -> bool:
    try:
        return abs(float(value) - expected) <= atol
    except Exception:
        return False


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
        "two_source_options_only": int(h.num_options) == 2,
        "short_bounded_duration": int(h.min_duration) == 1 and int(h.max_duration) == 8,
        "reactive_reselection_floor": (
            int(h.commitment_warmup_steps) == 100_000
            and int(h.commitment_full_steps) == 600_000
            and close(h.commitment_reselect_initial, 1.0)
            and close(h.commitment_reselect_final, 0.25)
        ),
        "staged_policy_learning": (
            int(h.worker_pg_warmup_steps) == 20_000
            and int(h.worker_pg_full_steps) == 150_000
            and int(h.manager_pg_warmup_steps) == 100_000
            and int(h.manager_pg_full_steps) == 500_000
        ),
        "late_low_scale_termination": (
            int(h.termination_warmup_steps) == 350_000
            and int(h.termination_full_steps) == 800_000
            and close(h.termination_loss_scale, 0.02)
            and close(h.termination_max_probability_during_ramp, 0.30)
            and close(h.termination_max_probability_final, 0.30)
        ),
        "forward_action_distillation": (
            close(h.base_kl_target, 0.002)
            and close(h.base_kl_tail_target, 0.01)
            and close(h.base_kl_scale, 0.50)
            and close(h.action_preservation_scale, 0.50)
            and int(h.max_diversity_states) == 2048
        ),
        "source_manager_distillation": (
            int(h.source_manager_group_count) == 2
            and close(h.manager_group_kl_target, 0.001)
            and close(h.manager_group_kl_tail_target, 0.005)
            and close(h.manager_group_kl_scale, 0.50)
            and close(h.manager_group_preservation_scale, 0.50)
        ),
        "task_agnostic_regularizers_disabled": (
            close(h.manager_collapse_scale, 0.0)
            and close(h.manager_mi_scale, 0.0)
            and close(h.action_diversity_scale, 0.0)
            and close(h.residual_cosine_scale, 0.0)
        ),
        "world_model_exactly_frozen": (
            close(h.world_model_grad_scale_initial, 0.0)
            and close(h.world_model_grad_scale_final, 0.0)
        ),
        "conservative_variable_horizon": (
            int(h.imag_horizon_initial_max) == 10
            and int(h.imag_horizon_final_max) == 12
            and int(h.imag_horizon_window) == 4
            and int(h.imag_horizon_ramp_steps) == 600_000
        ),
        "validation_start_and_every_200k": (
            bool(cfg.validation.run_at_start)
            and int(cfg.validation.every) == 200_000
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
        "critic": repo / "external/r2dreamer/option_critic.py",
        "dreamer": repo / "external/r2dreamer/dreamer.py",
        "trainer": repo / "external/r2dreamer/trainer.py",
        "runner": repo / "scripts/train_r2dreamer_smaclite_multimap.py",
        "validation": repo / "src/smacdreamer/validation_trainer.py",
        "tools": repo / "external/r2dreamer/tools.py",
        "launcher": repo / "scripts/run_option_critic_v5_stability_1m.sh",
        "pipeline": repo / "scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh",
        "forecast": repo / "scripts/run_exp45_full_train_eval_resilient.sh",
    }
    for path in paths.values():
        if not path.is_file():
            fail(f"missing installed source: {path}")
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    require_tokens(text["options"], (
        'ARCHITECTURE = "dreamer_option_critic_v5_stability"',
        "active_imagination_horizon_range",
        "source.clamp_min(eps)",
        "base_kl_distillation",
        "kl_distillation",
        "commitment_reselect_final: float = 0.25",
        "termination_max_probability_final: float = 0.30",
    ), "hierarchical_options")
    if "live.clamp_min(eps) * (live.clamp_min(eps).log() - source" in text["options"]:
        fail("reverse action KL remains in v5 source-preservation path")
    require_tokens(text["hierarchy"], (
        "real_source_logits",
        "real_source_manager_probs",
        "source_policy_kl_loss = 0.5",
        "source_manager_kl_loss = 0.5",
        '"option/real_source_policy_kl_mean"',
        '"option/real_source_manager_group_kl_mean"',
        "start_valid = (1.0 - raw_data[\"is_last\"].float())",
        "next_imagination_horizon()",
        "temperature = 1.0",
    ), "hierarchical_dreamer")
    require_tokens(text["critic"], (
        "call_and_return_bootstrap",
        "torch.lerp",
    ), "option_critic")
    require_tokens(text["dreamer"], (
        "# OPTION_CRITIC_P0P1_HOTFIX_V3",
        "apply_hierarchy_gradient_guards",
        "apply_hierarchy_gradient_guards(self)",
    ), "dreamer integration")
    update = method_source(text["dreamer"], "update")
    guard_pos = update.find("apply_hierarchy_gradient_guards(self)")
    unscale_positions: list[int] = []
    offset = 0
    for line in update.splitlines(keepends=True):
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("self._scaler.unscale_(") and stripped.endswith(")"):
            unscale_positions.append(offset)
        offset += len(line)
    if len(unscale_positions) != 1:
        fail(f"expected one optimizer unscale call, found {len(unscale_positions)}")
    if guard_pos < 0 or guard_pos > unscale_positions[0]:
        fail("world-model gradient guard is not before optimizer unscale")

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
        "torch.bfloat16", "x = x.float()", "return x.cpu().numpy()",
    ), "BF16 logging guard")
    require_tokens(text["launcher"], (
        'FINAL_STEP="${FINAL_STEP:-1000000}"',
        "CURRENT_OPTION_CRITIC_V5_STABILITY_1M_RUN.txt",
        "static_audit_option_critic_v5_stability.sh",
    ), "1M launcher")
    require_tokens(text["pipeline"], (
        "CONTINUE_ON_FAILURE",
        "run_option_critic_v5_stability_1m.sh",
        "run_exp45_full_train_eval_resilient.sh",
        "CURRENT_OPTION_CRITIC_V5_AND_EXP45_PIPELINE.txt",
    ), "combined pipeline")
    require_tokens(text["forecast"], (
        "run_stage", "EVAL_PARTIAL_ON_TRAIN_FAILURE", "ordinary_eval", "hidden_eval",
    ), "resilient forecast pipeline")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("agent_state_dict")
    meta = ckpt.get("tactical_mixture_metadata") or {}
    if not isinstance(state, dict):
        fail("source checkpoint lacks agent_state_dict")
    if args.require_v1_2_source:
        if meta.get("architecture") != "tactical_mixture_v1_2":
            fail(f"wrong source architecture: {meta.get('architecture')!r}")
        if int(meta.get("num_tactics", -1)) != 2:
            fail("source checkpoint is not two-tactic v1.2")
        if any(key.startswith("hierarchical_options.") for key in state):
            fail("source checkpoint already contains hierarchy parameters")
        if float(ckpt.get("val_macro_win_rate", -1.0)) < 0.3749:
            fail("source checkpoint macro validation win rate is below 0.375")
    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(run_meta, dict):
        fail("source run metadata is not a mapping")

    print("[OK] Option-Critic v5 stability source/config audit passed")
    print(json.dumps({
        "repo": str(repo),
        "config": str(config_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "step": ckpt.get("step"),
            "val_macro_win_rate": ckpt.get("val_macro_win_rate"),
            "val_macro_original_return": ckpt.get("val_macro_original_return"),
        },
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
