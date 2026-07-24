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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_tokens(text: str, tokens: list[str], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        fail(f"{label} missing contracts: {missing}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-run-meta", type=Path, required=True)
    p.add_argument("--require-v1-2-source", action="store_true")
    args = p.parse_args()

    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    checkpoint_path = args.checkpoint.resolve()
    meta_path = args.source_run_meta.resolve()
    for path in (config_path, checkpoint_path, meta_path):
        if not path.is_file():
            fail(f"missing required path: {path}")

    cfg = OmegaConf.load(config_path)
    h = cfg.hierarchical_options
    checks = {
        "world_model": str(cfg.world_model.backend) == "jepa",
        "reward": str(cfg.reward.name) == "dense_v3",
        "horizon": int(cfg.imag_horizon) == 5,
        "hierarchy_enabled": bool(h.enabled),
        "eight_options": int(h.num_options) == 8,
        "duration": int(h.min_duration) == 3 and int(h.max_duration) == 20,
        "learned_termination": (
            float(h.initial_termination_probability) == 0.10
            and int(h.termination_warmup_steps) == 100000
            and int(h.termination_full_steps) == 300000
            and not bool(h.eval_sample_termination)
            and float(h.eval_termination_hazard_threshold) == 1.0
            and float(h.termination_max_probability_during_ramp) == 0.30
            and float(h.termination_max_probability_final) == 0.80
            and int(h.termination_cap_full_steps) == 500000
            and float(h.termination_unimix) == 0.02
            and float(h.termination_entropy_scale) == 0.0
            and float(h.termination_collapse_scale) == 0.05
            and float(h.termination_mean_max) == 0.60
            and float(h.termination_min_advantage_magnitude) == 0.01
            and float(h.termination_max_target_disagreement) == 0.25
            and float(h.manager_unimix_initial) == 0.20
            and int(h.manager_unimix_decay_steps) == 300000
            and int(h.manager_pg_warmup_steps) == 25000
            and int(h.manager_pg_full_steps) == 100000
        ),
        "conservative_worker": (
            float(h.worker_scale_max) == 0.25
            and float(h.max_residual_to_base) == 0.25
            and float(h.base_kl_target) == 0.02
        ),
        "non_forced_capacity": float(h.min_effective_options) == 3.0,
        "tactical_disabled": not bool(cfg.tactical_mixture.enabled),
        "adaptive_disabled": (
            not bool(cfg.adaptive_priority.enabled)
            and not bool(cfg.adaptive_priority.map.enabled)
            and not bool(cfg.adaptive_priority.sequence.enabled)
        ),
        "uniform_replay": str(cfg.sampling_mode) != "adaptive_priority",
        "fresh_replay": str(cfg.buffer.scratch_dir) == "replay",
        "startup_validation_disabled": cfg.validation.run_at_start is False,
        "validation_interval": int(cfg.validation.every) == 200000,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        fail(f"config contract failed: {failed}")

    dreamer = (repo / "external/r2dreamer/dreamer.py").read_text(encoding="utf-8")
    trainer = (repo / "external/r2dreamer/trainer.py").read_text(encoding="utf-8")
    runner = (repo / "scripts/train_r2dreamer_smaclite_multimap.py").read_text(encoding="utf-8")
    validation = (repo / "src/smacdreamer/validation_trainer.py").read_text(encoding="utf-8")
    hierarchy_helper = (repo / "external/r2dreamer/hierarchical_dreamer.py").read_text(
        encoding="utf-8"
    )
    hierarchy_policy = (repo / "external/r2dreamer/hierarchical_options.py").read_text(
        encoding="utf-8"
    )
    option_critic_source = (repo / "external/r2dreamer/option_critic.py").read_text(
        encoding="utf-8"
    )
    ast.parse(hierarchy_policy, filename="hierarchical_options")
    ast.parse(option_critic_source, filename="option_critic")
    require_tokens(
        hierarchy_policy,
        [
            'ARCHITECTURE = "dreamer_option_critic_v2"',
            "termination_option_embedding",
            "pairwise_js_shortfall",
            "candidate_hazard >= self.settings.eval_termination_hazard_threshold",
            "def termination_probability_cap(",
            "forced_continue = age < s.min_duration",
            "forced_terminate = age >= s.max_duration",
        ],
        "hierarchical policy",
    )
    require_tokens(
        option_critic_source,
        [
            "def call_and_return_bootstrap(",
            "def option_lambda_return(",
            "def normalized_advantage(",
            "boundary_mask",
            "Q_continue - V_switch + margin",
        ],
        "option critic math",
    )
    for name, text in (
        ("dreamer", dreamer),
        ("trainer", trainer),
        ("runner", runner),
        ("validation", validation),
    ):
        ast.parse(text, filename=name)
        if "OPTION_CRITIC_HIERARCHY_V2" not in text:
            fail(f"{name} lacks hierarchy source marker")
        if "OPTION_CRITIC_HIERARCHY_V1" in text:
            fail(f"{name} still contains the superseded v1 marker")

    require_tokens(
        dreamer,
        [
            "build_hierarchical_modules(self, config)",
            "def set_hierarchy_training_step(self, step)",
            'modules["hierarchical_options"] = self.hierarchical_options',
            'modules["option_critic"] = self.option_critic',
            'modules.pop("actor", None)',
            'modules.pop("jepa_feature_adapter", None)',
            "clone_and_freeze_hierarchy(self)",
            "hierarchical_act_logits(",
            "hierarchy_state_dict_fields(",
            "hierarchical_auxiliary_loss(",
            "self._scaler.scale(hierarchy_loss).backward()",
            "update_slow_option_critic(self)",
            "def load_hierarchical_compatible_state_dict",
            "load_hierarchical_compatible_state(",
            'losses.pop(legacy_key, None)',
            'option/legacy_behavior_losses_disabled',
        ],
        "Dreamer",
    )
    require_tokens(
        hierarchy_helper,
        [
            "Option-Critic migration requires a Tactical Mixture v1.2",
            "zero-sum codes break the worker's otherwise invariant",
            "temperature * source_out_w[source_index]",
            "code_epsilon = 1.0e-3",
            "normalized_advantage",
            "manager_pg_blend",
            'raw_data["option_before_id"]',
            'raw_data["option_before_age"]',
            'raw_data["option_before_has"]',
            'call_and_return_bootstrap(',
            'option_lambda_return(',
            'execution_beta, online_termination_eligible',
            'reliable_termination',
            'termination_target_sign_agreement',
            'option/real_boundary_rate',
            'option/real_usage_',
            'option/real_min_duration_violation_rate',
            'option/imag_change_without_boundary_rate',
            'option/imag_return_',
            'option/js_shortfall_fraction',
            'option/termination_execution_cap',
            '- blend * s.termination_entropy_scale',
            '+ blend * s.termination_collapse_scale',
        ],
        "hierarchical helper",
    )
    if dreamer.index('modules["hierarchical_options"]') > dreamer.index("self._named_params = OrderedDict()"):
        fail("hierarchy modules are registered after optimizer parameter collection")
    if dreamer.index("hierarchical_auxiliary_loss(") > dreamer.index("self._scaler.unscale_(self._optimizer)"):
        fail("hierarchy backward occurs after gradient unscale")
    dreamer_tree = ast.parse(dreamer, filename="dreamer")
    jepa_grad_methods = [
        node
        for node in ast.walk(dreamer_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_cal_grad_jepa"
    ]
    if len(jepa_grad_methods) != 1:
        fail(
            "expected exactly one _cal_grad_jepa method, found "
            f"{len(jepa_grad_methods)}"
        )

    jepa_grad_node = jepa_grad_methods[0]
    dreamer_lines = dreamer.splitlines(keepends=True)
    jepa_grad = "".join(
        dreamer_lines[
            jepa_grad_node.lineno - 1 : jepa_grad_node.end_lineno
        ]
    )

    legacy_disable_token = "losses.pop(legacy_key, None)"
    total_loss_token = "total_loss = sum([v * self._loss_scales[k]"

    if legacy_disable_token not in jepa_grad:
        fail("legacy behavior loss disabling is missing from _cal_grad_jepa")
    if total_loss_token not in jepa_grad:
        fail("JEPA total-loss construction is missing from _cal_grad_jepa")
    if jepa_grad.index(legacy_disable_token) > jepa_grad.index(
        total_loss_token
    ):
        fail(
            "legacy behavior losses are disabled after total-loss construction"
        )

    require_tokens(
        trainer,
        [
            'trans[option_key] = agent_state[option_key]',
            'agent.set_hierarchy_training_step(step)',
            'agent.set_hierarchy_training_step(train_step)',
            '"option_id", "option_age", "option_has"',
            '"option_before_id", "option_before_age"',
            '"option_before_has", "option_action_age"',
            '"option_termination_prob"',
        ],
        "trainer replay",
    )
    require_tokens(
        runner,
        [
            "config.model.hierarchical_options",
            "load_hierarchical_compatible_state_dict",
            '"hierarchical_options_metadata"',
        ],
        "runner",
    )
    require_tokens(
        validation,
        [
            'best_payload["hierarchical_options_metadata"]',
            'agent.set_hierarchy_training_step(train_step)',
        ],
        "validation checkpoint",
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("agent_state_dict")
    if not isinstance(state, dict):
        fail("source checkpoint has no agent_state_dict")
    has_tactical = any(key.startswith("tactical_policy.") for key in state)
    has_hierarchy = any(key.startswith("hierarchical_options.") for key in state)
    tactical_meta = ckpt.get("tactical_mixture_metadata") or {}
    if args.require_v1_2_source:
        if tactical_meta.get("architecture") != "tactical_mixture_v1_2":
            fail(f"source architecture is {tactical_meta.get('architecture')!r}, not v1.2")
        if int(tactical_meta.get("num_tactics", -1)) != 2:
            fail("source checkpoint is not the two-tactic v1.2 model")
        if not has_tactical or has_hierarchy:
            fail("source checkpoint tactical/hierarchy key lineage is invalid")
        source_win = float(ckpt.get("val_macro_win_rate", -1.0))
        if source_win < 0.3749:
            fail(f"source checkpoint win rate {source_win} is below the v1.2 best 0.375")

    run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if checkpoint_path.parent != meta_path.parent:
        fail("checkpoint and source run_meta are not from the same run directory")

    report = {
        "repo": str(repo),
        "config": str(config_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "step": ckpt.get("step"),
            "val_macro_win_rate": ckpt.get("val_macro_win_rate"),
            "val_macro_original_return": ckpt.get("val_macro_original_return"),
            "tactical_architecture": tactical_meta.get("architecture"),
            "has_training_state": bool(ckpt.get("agent_training_state")),
        },
        "checks": {**checks, "source_ast": True, "migration_lineage": True},
    }
    print("[OK] Option-Critic hierarchy source/config audit passed")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
