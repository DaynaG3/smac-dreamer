#!/usr/bin/env python3
"""Read-only source/config/checkpoint audit for Tactical Mixture hardening v1.1."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
from typing import Any

import torch
from omegaconf import OmegaConf


MARKER = "TACTICAL_MIXTURE_HARDENING_V1_1"


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


def check_set_env_step_guards(text: str) -> None:
    lines = text.splitlines()
    occurrences = 0
    for index, line in enumerate(lines):
        if "replay_buffer.set_env_step(" not in line:
            continue
        occurrences += 1
        indent = len(line) - len(line.lstrip())
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0:
            fail("set_env_step call has no guard")
        prev = lines[previous].strip()
        prev_indent = len(lines[previous]) - len(lines[previous].lstrip())
        if "hasattr(replay_buffer" not in prev or "set_env_step" not in prev:
            fail(f"unguarded replay_buffer.set_env_step at line {index + 1}")
        if prev_indent != indent - 4:
            fail(f"mis-indented set_env_step guard at line {index + 1}")
    if occurrences == 0:
        fail("runner contains no replay_buffer.set_env_step calls to audit")



def method_source(text: str, method_name: str) -> str:
    tree = ast.parse(text)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(matches) != 1:
        fail(f"method {method_name!r}: expected one AST match, found {len(matches)}")
    source = ast.get_source_segment(text, matches[0])
    if not source:
        fail(f"method {method_name!r}: source extraction failed")
    return source


def check_tactical_action_contracts(dreamer_text: str) -> None:
    act = method_source(dreamer_text, "act")
    tactic_positions = [
        position
        for marker in ("eval_combined_logits(", "combine_logits(")
        if (position := act.find(marker)) >= 0
    ]
    if not tactic_positions:
        fail("act() has no tactical logit composition")
    mask_position = act.find("MaskedMultiOneHotDist(")
    if mask_position < 0 or min(tactic_positions) > mask_position:
        fail("real action mask is not applied after tactical logit composition")

    imagine = method_source(dreamer_text, "_imagine")
    for marker in (
        "_frozen_tactical_policy.select_tactic(",
        "_frozen_tactical_policy.combine_logits(",
        "MaskedMultiOneHotDist(",
        "img_step(",
    ):
        if marker not in imagine:
            fail(f"_imagine() lacks required marker: {marker}")
    if imagine.find("combine_logits(") > imagine.find("MaskedMultiOneHotDist("):
        fail("imagined action mask is not applied after tactical logit composition")

    tree = ast.parse(dreamer_text)
    img_step_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "img_step":
            img_step_calls.append(node)
    if not img_step_calls:
        fail("Dreamer source contains no img_step call")
    for call in img_step_calls:
        source = ast.get_source_segment(dreamer_text, call) or ""
        if "tactic" in source:
            fail("tactic tensor is incorrectly passed into the world-model img_step")

    objective = method_source(dreamer_text, "_cal_grad_jepa")
    required = (
        "base_policy_logits",
        "tactic_logits",
        "imag_tactic",
        "tactic_dist.log_prob(imag_tactic)",
        "self.tactical_policy.combine_logits(",
    )
    missing = [marker for marker in required if marker not in objective]
    if missing:
        fail(f"JEPA tactical objective contract is incomplete: {missing}")


def check_optimizer_registration_order(dreamer_text: str) -> None:
    tree = ast.parse(dreamer_text)

    dreamer_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Dreamer"
    ]
    if len(dreamer_classes) != 1:
        fail(
            "expected exactly one Dreamer class, found "
            f"{len(dreamer_classes)}"
        )

    init_methods = [
        node
        for node in dreamer_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__init__"
    ]
    if len(init_methods) != 1:
        fail(
            "expected exactly one Dreamer.__init__, found "
            f"{len(init_methods)}"
        )

    init_node = init_methods[0]
    tactical_assignments = []
    named_param_assignments = []

    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "modules"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "tactical_policy"
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "tactical_policy"
            ):
                tactical_assignments.append(node)

            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_named_params"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "OrderedDict"
            ):
                named_param_assignments.append(node)

    if len(tactical_assignments) != 1:
        fail(
            "expected exactly one semantic optimizer registration "
            "modules[tactical_policy] = self.tactical_policy, found "
            f"{len(tactical_assignments)}"
        )

    if len(named_param_assignments) != 1:
        fail(
            "expected exactly one self._named_params = OrderedDict() "
            f"assignment, found {len(named_param_assignments)}"
        )

    tactical_line = tactical_assignments[0].lineno
    named_line = named_param_assignments[0].lineno

    if tactical_line > named_line:
        fail(
            "tactical policy is registered after optimizer "
            "parameter collection"
        )

    init_source = ast.get_source_segment(dreamer_text, init_node) or ""
    named_position = init_source.find(
        "self._named_params = OrderedDict()"
    )

    if named_position < 0:
        fail(
            "could not locate optimizer parameter collection "
            "in Dreamer.__init__"
        )

    for marker, label in (
        ("inherited base actor frozen", "base actor"),
        ("inherited JEPA adapter frozen", "JEPA adapter"),
    ):
        marker_position = init_source.find(marker)

        if marker_position < 0:
            fail(f"{label} freeze marker missing")

        if marker_position > named_position:
            fail(
                f"{label} is frozen after optimizer "
                "parameter collection"
            )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--config",
        default="configs/r2_2100_jepa_tactical_mixture_hardened.yaml",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--source-run-meta")
    parser.add_argument("--require-legacy-source", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).resolve()
    files = {
        "dreamer": repo / "external/r2dreamer/dreamer.py",
        "policy": repo / "external/r2dreamer/tactical_policy.py",
        "runner": repo / "scripts/train_r2dreamer_smaclite_multimap.py",
        "validation": repo / "src/smacdreamer/validation_trainer.py",
    }
    texts = {}
    for name, path in files.items():
        if not path.is_file():
            fail(f"missing {name} source: {path}")
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        texts[name] = text

    launcher_path = repo / "scripts/run_tactical_hardened_2m.sh"
    if not launcher_path.is_file():
        fail(f"missing hardened launch script: {launcher_path}")
    launcher = launcher_path.read_text(encoding="utf-8")
    for marker in (
        "CURRENT_UNIFIED_PRIORITY_RUN.txt",
        "best_val_macro_winrate.pt",
        "source checkpoint already contains tactical parameters",
        "refusing to reuse non-empty RUN_DIR",
        "source run metadata is not inside the selected adaptive run",
    ):
        if marker not in launcher:
            fail(f"hardened launcher safety marker missing: {marker}")
    if '${CHECKPOINT:-' in launcher or 'CHECKPOINT="${CHECKPOINT' in launcher:
        fail(
            "hardened launcher still accepts the ambiguous generic "
            "CHECKPOINT variable"
        )

    for name in ("dreamer", "runner", "validation"):
        if MARKER not in texts[name]:
            fail(f"{name} lacks hardening marker")

    required_dreamer = [
        "usage_statistics(",
        "effect_statistics(",
        "base_policy_logits[:, :-1].detach()",
        'metrics["tactic/mutual_information"]',
        'f"tactic/sampled_usage_',
        "optimizer parameter registry contains duplicates",
        "metadata-less tactical checkpoint",
        "eval_combined_logits(",
        "inherited base actor frozen",
        "inherited JEPA adapter frozen",
        'metrics["tactic/residual_guard_loss"]',
        'metrics["jepa/adapter_total_parameter_count"]',
        "if p.requires_grad",
        "metadata_is_legacy",
    ]
    missing = [item for item in required_dreamer if item not in texts["dreamer"]]
    if missing:
        fail(f"dreamer hardening markers missing: {missing}")

    required_policy = [
        'ARCHITECTURE = "tactical_mixture_v1_1"',
        "collapse_loss_scale",
        "symmetry_break_std",
        "mutual_information_normalized",
        "base_logits.detach()",
        "_repair_empty_masks",
        "eval_confidence_threshold",
        "freeze_base_actor",
        "freeze_feature_adapter",
        "max_residual_to_base",
        "max_abs_residual_logit",
        "cap * torch.tanh(raw / cap)",
        "continuous confidence gate",
        "state_weight = active_sel.any(-1).float()",
    ]
    missing = [item for item in required_policy if item not in texts["policy"]]
    if missing:
        fail(f"tactical policy hardening markers missing: {missing}")
    check_tactical_action_contracts(texts["dreamer"])
    check_optimizer_registration_order(texts["dreamer"])
    if "torch.nonzero" in texts["policy"]:
        fail("tactical effect path still uses dynamic torch.nonzero")
    if "empty.any()" in texts["policy"]:
        fail("tactical mask repair still has a Tensor-to-Python branch")

    check_set_env_step_guards(texts["runner"])
    for marker in (
        "config.model.tactical_mixture",
        "tactical_mixture_metadata",
        "adaptive priority disabled; using original uniform SliceSampler",
    ):
        if marker not in texts["runner"]:
            fail(f"runner tactical/config isolation marker missing: {marker}")
    if "adaptive priority disabled; source priority state skipped" not in texts["runner"]:
        fail("runner does not explicitly skip adaptive state when disabled")
    if "if _adaptive_any:" not in texts["runner"]:
        fail("runner does not condition adaptive checkpoint state")
    if "tactical modules are zero-init" in texts["runner"]:
        fail("runner still claims exact zero tactical initialization")

    if "tactical_mixture_metadata" not in texts["validation"]:
        fail("best validation checkpoint still omits tactical metadata")
    if "best_payload" not in texts["validation"]:
        fail("best validation checkpoint payload was not hardened")

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    if not config_path.is_file():
        fail(f"missing hardened config: {config_path}")
    cfg = OmegaConf.load(config_path)
    tactical = cfg.get("tactical_mixture") or {}
    expected = {
        "enabled": True,
        "duration": 1,
        "num_tactics": 4,
    }
    for key, value in expected.items():
        if tactical.get(key) != value:
            fail(f"config tactical_mixture.{key}={tactical.get(key)!r}, expected {value!r}")
    for key in (
        "collapse_loss_scale",
        "max_usage_target",
        "min_effective_tactics",
        "symmetry_break_std",
        "eval_confidence_threshold",
        "freeze_base_actor",
        "freeze_feature_adapter",
        "residual_guard_scale",
        "max_residual_to_base",
        "max_abs_residual_logit",
    ):
        if tactical.get(key) is None:
            fail(f"hardened config missing tactical_mixture.{key}")
    if not bool(tactical.get("freeze_base_actor")):
        fail("recommended hardened config must freeze the inherited base actor")
    if not bool(tactical.get("freeze_feature_adapter")):
        fail("recommended hardened config must freeze the inherited JEPA adapter")
    threshold = float(tactical.get("eval_confidence_threshold"))
    if not 0.25 <= threshold <= 1.0:
        fail("eval_confidence_threshold must be in [0.25, 1]")
    if not math.isclose(threshold, 0.55, rel_tol=0.0, abs_tol=1e-12):
        fail("recommended hardened eval confidence threshold must be 0.55")
    symmetry = float(tactical.get("symmetry_break_std"))
    if not math.isclose(symmetry, 1.0e-2, rel_tol=0.0, abs_tol=1e-12):
        fail("recommended hardened symmetry_break_std must be 1e-2")
    if float(tactical.get("max_usage_target")) != 0.80:
        fail("recommended max_usage_target must be 0.80")
    if float(tactical.get("min_effective_tactics")) != 2.0:
        fail("recommended min_effective_tactics must be 2.0")
    if float(tactical.get("max_residual_to_base")) != 1.0:
        fail("recommended max_residual_to_base must be 1.0")
    if float(tactical.get("max_abs_residual_logit")) != 4.0:
        fail("recommended max_abs_residual_logit must be 4.0")
    if bool((cfg.get("adaptive_priority") or {}).get("enabled", False)):
        fail("adaptive priority must be disabled for the isolated tactical run")
    if bool(((cfg.get("adaptive_priority") or {}).get("map") or {}).get("enabled", False)):
        fail("adaptive map priority is enabled")
    if bool(((cfg.get("adaptive_priority") or {}).get("sequence") or {}).get("enabled", False)):
        fail("adaptive sequence PER is enabled")
    if str(cfg.get("sampling_mode", "")) == "adaptive_priority":
        fail("sampling_mode remains adaptive_priority")
    if bool((cfg.get("validation") or {}).get("run_at_start", True)):
        fail("startup validation must be disabled")
    if int((cfg.get("validation") or {}).get("every", 0)) != 200000:
        fail("validation interval must remain 200000")
    if int(cfg.get("imag_horizon", -1)) != 5:
        fail("imagination horizon changed from 5")
    if str(nested_get(cfg, "world_model", "backend", default="")) != "jepa":
        fail("world model backend is not JEPA")
    scratch_dir = str(nested_get(cfg, "buffer", "scratch_dir", default=""))
    if scratch_dir != "replay":
        fail(
            "hardened config must use relative buffer.scratch_dir='replay' "
            "to prevent stale memmap reuse"
        )

    source_compatibility = None
    if args.source_run_meta:
        meta_path = pathlib.Path(args.source_run_meta).resolve()
        if not meta_path.is_file():
            fail(f"source run metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        comparisons = {
            "reward_name": (
                str(nested_get(cfg, "reward", "name", default="")),
                str(meta.get("reward_name", "")),
            ),
            "imag_horizon": (
                int(cfg.get("imag_horizon", -1)),
                int(meta.get("imag_horizon", -2)),
            ),
            "max_episode_steps": (
                int(cfg.get("max_episode_steps", -1)),
                int(meta.get("max_episode_steps", -2)),
            ),
            "gamma": (
                float(cfg.get("gamma", float("nan"))),
                float(meta.get("gamma", float("nan"))),
            ),
            "world_model_backend": (
                str(nested_get(cfg, "world_model", "backend", default="")),
                str(meta.get("world_model_backend", "")),
            ),
        }
        mismatches = {
            key: {"hardened_config": left, "source_run": right}
            for key, (left, right) in comparisons.items()
            if left != right
        }
        if mismatches:
            fail(
                "hardened tactical config differs from source run in critical "
                f"fields: {json.dumps(mismatches, sort_keys=True)}"
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
        source_compatibility = {
            "source_run_meta": str(meta_path),
            "critical_fields": "match",
            "jepa_checkpoint": str(jepa_path),
            "jepa_checkpoint_sha256": expected_hash,
        }

    checkpoint_result = None
    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint).resolve()
        if not checkpoint_path.is_file():
            fail(f"checkpoint missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("agent_state_dict")
        if not isinstance(state, dict):
            fail("checkpoint lacks agent_state_dict")
        tactical_keys = [key for key in state if key.startswith("tactical_policy.")]
        metadata = checkpoint.get("tactical_mixture_metadata")
        if tactical_keys and metadata is not None:
            architecture = metadata.get("architecture")
            if architecture not in (
                "tactical_mixture_v1",
                "tactical_mixture_v1_1",
            ):
                fail(f"unsupported tactical checkpoint architecture {architecture!r}")
        if args.require_legacy_source and tactical_keys:
            fail(
                "source checkpoint already contains tactical parameters; use the "
                "adaptive-PER best checkpoint rather than the declining v1 "
                "tactical run"
            )
        checkpoint_result = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": int(checkpoint.get("step", 0)),
            "val_macro_win_rate": checkpoint.get("val_macro_win_rate"),
            "val_macro_original_return": checkpoint.get(
                "val_macro_original_return"
            ),
            "has_tactical_keys": bool(tactical_keys),
            "has_tactical_metadata": metadata is not None,
            "has_training_state": bool(checkpoint.get("agent_training_state")),
        }

    result = {
        "repo": str(repo),
        "config": str(config_path),
        "checkpoint": checkpoint_result,
        "source_compatibility": source_compatibility,
        "checks": {
            "source_ast": "ok",
            "buffer_compatibility": "ok",
            "effect_gradient_isolation": "ok",
            "best_checkpoint_metadata": "ok",
            "smooth_conservative_eval_gate": "ok",
            "inherited_policy_freeze": "ok",
            "adapter_trainable_count_metric": "ok",
            "compile_friendly_effect_path": "ok",
            "bounded_residual_logits": "ok",
            "action_mask_ordering": "ok",
            "world_model_action_contract": "ok",
            "optimizer_registration_order": "ok",
            "config_plumbing": "ok",
            "legacy_metadata_compatibility": "ok",
            "config_isolation": "ok",
            "fresh_replay_path": "ok",
            "launcher_source_lineage": "ok",
            "nonempty_run_dir_guard": "ok",
        },
    }
    print("[OK] tactical hardening source/config audit passed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
