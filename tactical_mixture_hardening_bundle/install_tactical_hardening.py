#!/usr/bin/env python3
"""Fail-closed hardening installer for an existing Tactical Mixture v1 repo.

This installer expects the initial tactical bundle and unified-priority bundle to
already be installed. It creates a complete backup before modifying anything.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from typing import Callable

from omegaconf import OmegaConf


BUNDLE = pathlib.Path(__file__).resolve().parent
PAYLOAD = BUNDLE / "payload"
HARDENING_MARKER = "# TACTICAL_MIXTURE_HARDENING_V1_1"


def die(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_manifest() -> None:
    manifest_path = BUNDLE / "MANIFEST.sha256.json"
    if not manifest_path.is_file():
        die(f"bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        die("unsupported bundle manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        die("bundle manifest has no file entries")
    for relative_text, expected in entries.items():
        relative = pathlib.Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            die(f"unsafe bundle manifest path: {relative}")
        path = BUNDLE / relative
        if not path.is_file():
            die(f"bundle file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            die(
                f"bundle file hash mismatch for {relative}: "
                f"{actual} != {expected}"
            )
    print(f"[OK] verified {len(entries)} bundle file hashes")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0 and new in text:
        return text
    if count != 1:
        die(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def method_segment(text: str, method_name: str) -> tuple[int, int, str]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(matches) != 1:
        die(f"method {method_name!r}: expected one AST match, found {len(matches)}")
    node = matches[0]
    if getattr(node, "end_lineno", None) is None:
        die(f"method {method_name!r}: missing end_lineno")
    start_line = node.lineno - 1
    end_line = node.end_lineno
    start = sum(len(line) for line in lines[:start_line])
    end = sum(len(line) for line in lines[:end_line])
    return start, end, "".join(lines[start_line:end_line])


def transform_method(
    text: str,
    method_name: str,
    transform: Callable[[str], str],
) -> str:
    start, end, segment = method_segment(text, method_name)
    changed = transform(segment)
    if changed == segment:
        die(f"method {method_name!r} was not changed")
    return text[:start] + changed + text[end:]


OLD_OBJECTIVE = '''        if self.tactical_enabled:
            tactic_logpi = tactic_dist.log_prob(imag_tactic)[
                :, :-1
            ].unsqueeze(-1)
            tactic_entropy = tactic_dist.entropy()[
                :, :-1
            ].unsqueeze(-1)
            tactical = self.tactical_policy.settings
            tactic_policy_loss = _priority_weighted_mean(
                weight[:, :-1].detach()
                * -(
                    tactical.tactic_pg_scale
                    * tactic_logpi
                    * adv.detach()
                    + tactical.tactic_entropy_scale * tactic_entropy
                )
            )
            start_is = _seq_is[:, None].expand(B, T).reshape(
                B * T, 1
            )
            tactic_aux_weight = (
                weight[:, :-1, 0].detach() * start_is
            )
            balance_loss, tactic_usage = (
                self.tactical_policy.balance_loss(
                    tactic_logits[:, :-1], tactic_aux_weight
                )
            )
            effect_js = self.tactical_policy.effect_js(
                imag_feat[:, :-1],
                base_policy_logits[:, :-1],
                _amask[:, :-1],
                _aactive[:, :-1],
                self._actor_shape,
                tactic_aux_weight,
            )
            effect_loss = torch.relu(
                torch.as_tensor(
                    tactical.effect_target,
                    device=effect_js.device,
                    dtype=effect_js.dtype,
                )
                - effect_js
            )
            losses["policy"] = (
                primitive_policy_loss
                + tactic_policy_loss
                + tactical.balance_loss_scale * balance_loss
                + tactical.effect_loss_scale * effect_loss
            )
            metrics["tactic/policy_loss"] = tactic_policy_loss
            metrics["tactic/entropy"] = tactic_entropy.mean()
            metrics["tactic/entropy_normalized"] = (
                tactic_entropy.mean()
                / math.log(self.tactical_policy.num_tactics)
            )
            metrics["tactic/selector_max_probability"] = (
                tactic_dist.probs.max(-1).values.mean()
            )
            metrics["tactic/balance_loss"] = balance_loss
            metrics["tactic/effect_loss"] = effect_loss
            metrics["tactic/effect_js"] = effect_js
            residual = policy_logits - base_policy_logits
            residual_rms = residual.float().square().mean().sqrt()
            base_rms = (
                base_policy_logits.float().square().mean().sqrt()
            )
            metrics["tactic/residual_rms"] = residual_rms
            metrics["tactic/residual_to_base_ratio"] = (
                residual_rms / base_rms.clamp_min(1e-6)
            )
            metrics["tactic/usage_max"] = tactic_usage.max()
            usage_entropy = -(
                tactic_usage.clamp_min(1e-8)
                * tactic_usage.clamp_min(1e-8).log()
            ).sum()
            metrics["tactic/effective_count"] = usage_entropy.exp()
            for tactic_index, tactic_probability in enumerate(
                tactic_usage
            ):
                metrics[
                    f"tactic/usage_{tactic_index}"
                ] = tactic_probability
        else:
            losses["policy"] = primitive_policy_loss
'''

NEW_OBJECTIVE = '''        if self.tactical_enabled:
            tactic_logpi = tactic_dist.log_prob(imag_tactic)[
                :, :-1
            ].unsqueeze(-1)
            tactic_entropy = tactic_dist.entropy()[
                :, :-1
            ].unsqueeze(-1)
            tactical = self.tactical_policy.settings
            tactic_policy_loss = _priority_weighted_mean(
                weight[:, :-1].detach()
                * -(
                    tactical.tactic_pg_scale
                    * tactic_logpi
                    * adv.detach()
                    + tactical.tactic_entropy_scale * tactic_entropy
                )
            )
            start_is = _seq_is[:, None].expand(B, T).reshape(
                B * T, 1
            )
            tactic_aux_weight = (
                weight[:, :-1, 0].detach() * start_is
            )
            tactic_stats = self.tactical_policy.usage_statistics(
                tactic_logits[:, :-1],
                sampled_tactic=imag_tactic[:, :-1],
                state_weights=tactic_aux_weight,
            )
            collapse_loss = tactic_stats["collapse_loss"]
            effect_stats = self.tactical_policy.effect_statistics(
                imag_feat[:, :-1].detach(),
                base_policy_logits[:, :-1].detach(),
                _amask[:, :-1].detach(),
                _aactive[:, :-1].detach(),
                self._actor_shape,
                tactic_aux_weight,
            )
            effect_js = effect_stats["js_mean"]
            effect_loss = torch.relu(
                torch.as_tensor(
                    tactical.effect_target,
                    device=effect_js.device,
                    dtype=effect_js.dtype,
                )
                - effect_js
            )
            residual_ratio = (
                effect_stats["residual_rms"]
                / effect_stats["base_rms"].clamp_min(1e-6)
            )
            residual_guard_loss = torch.relu(
                residual_ratio
                - torch.as_tensor(
                    tactical.max_residual_to_base,
                    device=residual_ratio.device,
                    dtype=residual_ratio.dtype,
                )
            ).square()
            losses["policy"] = (
                primitive_policy_loss
                + tactic_policy_loss
                + tactical.collapse_loss_scale * collapse_loss
                + tactical.effect_loss_scale * effect_loss
                + tactical.residual_guard_scale * residual_guard_loss
            )
            metrics["tactic/policy_loss"] = tactic_policy_loss
            metrics["tactic/entropy"] = tactic_entropy.mean()
            metrics["tactic/entropy_normalized"] = (
                tactic_entropy.mean()
                / math.log(self.tactical_policy.num_tactics)
            )
            metrics["tactic/conditional_entropy"] = tactic_stats[
                "conditional_entropy"
            ]
            metrics["tactic/marginal_entropy"] = tactic_stats[
                "marginal_entropy"
            ]
            metrics["tactic/mutual_information"] = tactic_stats[
                "mutual_information"
            ]
            metrics["tactic/mutual_information_normalized"] = tactic_stats[
                "mutual_information_normalized"
            ]
            metrics["tactic/selector_max_probability"] = tactic_stats[
                "selector_max_probability"
            ]
            metrics["tactic/selector_logit_std"] = tactic_stats[
                "selector_logit_std"
            ]
            metrics["tactic/collapse_loss"] = collapse_loss
            # Compatibility panel name; semantics are now collapse-only.
            metrics["tactic/balance_loss"] = collapse_loss
            metrics["tactic/effect_loss"] = effect_loss
            metrics["tactic/effect_js"] = effect_js
            metrics["tactic/effect_js_min"] = effect_stats["js_min"]
            metrics["tactic/effect_js_max"] = effect_stats["js_max"]
            metrics["tactic/residual_rms"] = effect_stats["residual_rms"]
            metrics["tactic/residual_to_base_ratio"] = residual_ratio
            metrics["tactic/residual_guard_loss"] = residual_guard_loss
            metrics["tactic/usage_max"] = tactic_stats["usage_max"]
            metrics["tactic/effective_count"] = tactic_stats[
                "effective_count"
            ]
            for tactic_index in range(self.tactical_policy.num_tactics):
                metrics[f"tactic/usage_{tactic_index}"] = tactic_stats[
                    "marginal"
                ][tactic_index]
                metrics[
                    f"tactic/sampled_usage_{tactic_index}"
                ] = tactic_stats["sampled_usage"][tactic_index]
                metrics[
                    f"tactic/argmax_usage_{tactic_index}"
                ] = tactic_stats["argmax_usage"][tactic_index]
                metrics[
                    f"tactic/residual_rms_{tactic_index}"
                ] = effect_stats[f"residual_rms_{tactic_index}"]
        else:
            losses["policy"] = primitive_policy_loss
'''


OLD_MIGRATION_START = "    def tactical_metadata(self):\n"
OLD_MIGRATION_END = "    def _update_slow_target(self):\n"

OLD_ACT_TACTIC = '''            if self.tactical_enabled:
                tactic = self._frozen_tactical_policy.select_tactic(
                    feat, deterministic=eval
                )
                raw_logits = self._frozen_tactical_policy.combine_logits(
                    raw_logits, feat, tactic
                )
'''

NEW_ACT_TACTIC = '''            if self.tactical_enabled:
                if eval:
                    (
                        raw_logits,
                        tactic,
                        tactic_confidence,
                        tactic_applied,
                    ) = self._frozen_tactical_policy.eval_combined_logits(
                        raw_logits, feat
                    )
                else:
                    tactic = self._frozen_tactical_policy.select_tactic(
                        feat, deterministic=False
                    )
                    raw_logits = (
                        self._frozen_tactical_policy.combine_logits(
                            raw_logits, feat, tactic
                        )
                    )
'''

NEW_MIGRATION = '''    def tactical_metadata(self):
        if not self.tactical_enabled:
            return {
                "schema_version": 2,
                "architecture": "legacy",
                "enabled": False,
            }
        metadata = self.tactical_policy.metadata()
        metadata["enabled"] = True
        return metadata

    def load_tactical_compatible_state_dict(
        self,
        state_dict,
        checkpoint_metadata=None,
    ):
        """Strict tactical resume or allowlisted migration from legacy.

        Metadata-less tactical best checkpoints from v1 are accepted only when
        their live tactical keys load shape-strictly. This repairs the original
        best-checkpoint metadata omission without relaxing legacy migration.
        """
        if not self.tactical_enabled:
            self.load_state_dict(state_dict, strict=True)
            self.clone_and_freeze()
            return {"migrated_legacy": False, "strict": True}

        state_keys = tuple(state_dict.keys())
        has_live_tactical = any(
            key.startswith("tactical_policy.") for key in state_keys
        )

        metadata_is_legacy = bool(
            checkpoint_metadata is not None
            and (
                checkpoint_metadata.get("enabled") is False
                or checkpoint_metadata.get("architecture") == "legacy"
            )
        )
        if metadata_is_legacy and has_live_tactical:
            raise RuntimeError(
                "checkpoint metadata declares a legacy policy but tactical "
                "parameter keys are present"
            )

        if checkpoint_metadata is not None and not metadata_is_legacy:
            architecture = checkpoint_metadata.get("architecture")
            if architecture not in (
                "tactical_mixture_v1",
                "tactical_mixture_v1_1",
            ):
                raise RuntimeError(
                    f"unsupported tactical checkpoint architecture: {architecture!r}"
                )
            expected = self.tactical_metadata()
            for key in (
                "num_tactics",
                "embedding_dim",
                "hidden_dim",
                "duration",
                "feature_dim",
                "action_logit_dim",
                "eval_confidence_threshold",
                "freeze_base_actor",
                "freeze_feature_adapter",
                "max_residual_to_base",
                "max_abs_residual_logit",
            ):
                if key in checkpoint_metadata and (
                    checkpoint_metadata.get(key) != expected.get(key)
                ):
                    raise RuntimeError(
                        f"tactical metadata mismatch for {key}: "
                        f"{checkpoint_metadata.get(key)!r} "
                        f"!= {expected.get(key)!r}"
                    )
            self.load_state_dict(state_dict, strict=True)
            self.clone_and_freeze()
            return {
                "migrated_legacy": False,
                "strict": True,
                "checkpoint_architecture": architecture,
            }

        if has_live_tactical:
            incompatible = self.load_state_dict(state_dict, strict=False)
            illegal_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("_frozen_tactical_policy.")
            ]
            if illegal_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "metadata-less tactical checkpoint is incompatible: "
                    f"illegal_missing={illegal_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}"
                )
            self.clone_and_freeze()
            return {
                "migrated_legacy": False,
                "strict": not bool(incompatible.missing_keys),
                "metadata_inferred": True,
            }

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_prefixes = (
            "tactical_policy.",
            "_frozen_tactical_policy.",
        )
        illegal_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "legacy tactical migration found incompatible keys: "
                f"illegal_missing={illegal_missing}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        if not incompatible.missing_keys:
            raise RuntimeError(
                "checkpoint has no tactical metadata and no missing tactical keys"
            )
        self.tactical_policy.assert_legacy_equivalence_ready()
        self.clone_and_freeze()
        return {
            "migrated_legacy": True,
            "strict": False,
            "missing_keys": list(incompatible.missing_keys),
        }

'''



def patch_adapter_parameter_metrics(text: str) -> str:
    """Make the JEPA adapter count metric reflect requires_grad accurately."""
    if 'metrics["jepa/adapter_total_parameter_count"]' in text:
        if "if p.requires_grad" not in text:
            die("adapter parameter metric appears partially hardened")
        return text

    tree = ast.parse(text)
    matches: list[ast.Assign] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "metrics":
                continue
            key = target.slice
            if isinstance(key, ast.Constant) and (
                key.value == "jepa/trainable_adapter_parameter_count"
            ):
                matches.append(node)
    if len(matches) != 1:
        die(
            "JEPA adapter parameter metric: expected one AST assignment, "
            f"found {len(matches)}"
        )
    node = matches[0]
    if node.end_lineno is None:
        die("JEPA adapter parameter metric lacks end_lineno")
    lines = text.splitlines(keepends=True)
    original = lines[node.lineno - 1]
    indent = original[: len(original) - len(original.lstrip())]
    i1 = indent + "    "
    replacement = (
        f'{indent}metrics["jepa/adapter_total_parameter_count"] = torch.tensor(\n'
        f'{i1}float(sum(\n'
        f'{i1}    p.numel()\n'
        f'{i1}    for p in self.jepa_world_model.feature_adapter.parameters()\n'
        f'{i1})),\n'
        f'{i1}device=feat.device,\n'
        f'{indent})\n'
        f'{indent}metrics["jepa/trainable_adapter_parameter_count"] = torch.tensor(\n'
        f'{i1}float(sum(\n'
        f'{i1}    p.numel()\n'
        f'{i1}    for p in self.jepa_world_model.feature_adapter.parameters()\n'
        f'{i1}    if p.requires_grad\n'
        f'{i1})),\n'
        f'{i1}device=feat.device,\n'
        f'{indent})\n'
    )
    return (
        "".join(lines[: node.lineno - 1])
        + replacement
        + "".join(lines[node.end_lineno :])
    )

def patch_dreamer(text: str) -> str:
    if HARDENING_MARKER in text:
        return text
    if "# TACTICAL_MIXTURE_V1" not in text:
        die("dreamer.py does not contain the original tactical integration")
    if "def load_tactical_compatible_state_dict" not in text:
        die("dreamer.py lacks tactical checkpoint migration")

    text = replace_once(
        text,
        "# TACTICAL_MIXTURE_V1\n",
        "# TACTICAL_MIXTURE_V1\n" + HARDENING_MARKER + "\n",
        "hardening marker",
    )

    def patch_objective(segment: str) -> str:
        return replace_once(
            segment,
            OLD_OBJECTIVE,
            NEW_OBJECTIVE,
            "hardened tactical objective",
        )

    text = transform_method(text, "_cal_grad_jepa", patch_objective)
    text = patch_adapter_parameter_metrics(text)

    start_count = text.count(OLD_MIGRATION_START)
    end_count = text.count(OLD_MIGRATION_END)
    if start_count != 1 or end_count != 1:
        die(
            "checkpoint method anchors are ambiguous: "
            f"start={start_count}, end={end_count}"
        )
    start = text.index(OLD_MIGRATION_START)
    end = text.index(OLD_MIGRATION_END, start)
    text = text[:start] + NEW_MIGRATION + text[end:]

    def patch_init(segment: str) -> str:
        freeze_anchor = "        # count number of parameters in each module\n"
        freeze_addition = (
            "        if self.tactical_enabled:\n"
            "            _tactical_settings = self.tactical_policy.settings\n"
            "            if _tactical_settings.freeze_base_actor:\n"
            "                for _param in self.actor.parameters():\n"
            "                    _param.requires_grad_(False)\n"
            "                modules.pop('actor', None)\n"
            "                print(' tactical safety: inherited base actor frozen')\n"
            "            if _tactical_settings.freeze_feature_adapter:\n"
            "                if self.world_model_backend != 'jepa':\n"
            "                    raise RuntimeError(\n"
            "                        'freeze_feature_adapter requires JEPA backend'\n"
            "                    )\n"
            "                for _param in (\n"
            "                    self.jepa_world_model.feature_adapter.parameters()\n"
            "                ):\n"
            "                    _param.requires_grad_(False)\n"
            "                modules.pop('jepa_feature_adapter', None)\n"
            "                print(' tactical safety: inherited JEPA adapter frozen')\n"
            "        # count number of parameters in each module\n"
        )
        segment = replace_once(
            segment,
            freeze_anchor,
            freeze_addition,
            "tactical inherited-policy freeze",
        )

        anchor = (
            '        print(f"Optimizer has: '
            '{sum(p.numel() for p in self._named_params.values())} parameters.")\n'
        )
        addition = anchor + (
            "        _optimizer_param_ids = [\n"
            "            id(param) for param in self._named_params.values()\n"
            "        ]\n"
            "        if len(_optimizer_param_ids) != len(set(_optimizer_param_ids)):\n"
            "            raise RuntimeError(\n"
            "                'optimizer parameter registry contains duplicates'\n"
            "            )\n"
            "        if self.tactical_enabled:\n"
            "            _tactical_param_ids = {\n"
            "                id(param) for param in self.tactical_policy.parameters()\n"
            "            }\n"
            "            _registered_tactical_ids = {\n"
            "                id(param)\n"
            "                for name, param in self._named_params.items()\n"
            "                if name.startswith('tactical_policy.')\n"
            "            }\n"
            "            if _registered_tactical_ids != _tactical_param_ids:\n"
            "                raise RuntimeError(\n"
            "                    'tactical parameters are not registered exactly once'\n"
            "                )\n"
        )
        return replace_once(
            segment,
            anchor,
            addition,
            "optimizer uniqueness assertions",
        )

    text = transform_method(text, "__init__", patch_init)

    def patch_act(segment: str) -> str:
        return replace_once(
            segment,
            OLD_ACT_TACTIC,
            NEW_ACT_TACTIC,
            "confidence-gated deterministic tactic evaluation",
        )

    text = transform_method(text, "act", patch_act)
    ast.parse(text)
    return text


def _guard_set_env_step(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    guarded = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("replay_buffer.set_env_step("):
            indent = line[: len(line) - len(stripped)]
            previous = output[-1].strip() if output else ""
            if previous == 'if hasattr(replay_buffer, "set_env_step"):' or previous == "if hasattr(replay_buffer, 'set_env_step'):":
                output.append(line)
            else:
                output.append(
                    f'{indent}if hasattr(replay_buffer, "set_env_step"):\n'
                )
                output.append("    " + line)
                guarded += 1
        else:
            output.append(line)
    patched = "".join(output)
    if "replay_buffer.set_env_step(" not in patched:
        die("runner has no replay_buffer.set_env_step call to audit")
    return patched


def patch_runner(text: str) -> str:
    if HARDENING_MARKER not in text:
        if "# TACTICAL_MIXTURE_V1" not in text:
            die("runner lacks original tactical marker")
        text = replace_once(
            text,
            "# TACTICAL_MIXTURE_V1\n",
            "# TACTICAL_MIXTURE_V1\n" + HARDENING_MARKER + "\n",
            "runner hardening marker",
        )

    text = _guard_set_env_step(text)

    text = text.replace(
        " [resume] migrated legacy weights; tactical modules are zero-init",
        " [resume] migrated legacy weights; tactical modules use bounded symmetry-break init",
    )

    resume_pattern = re.compile(
        r"(?P<indent>^[ \t]*)if ckpt\.get\('adaptive_priority_state'\) is not None:\n"
        r"(?P<body>(?:(?P=indent)[ \t]+.*\n)+?)"
        r"(?P=indent)else:\n"
        r"(?P<elsebody>(?:(?P=indent)[ \t]+.*\n)+?)",
        re.MULTILINE,
    )
    resume_matches = list(resume_pattern.finditer(text))
    already_hardened = "if _adaptive_any and ckpt.get('adaptive_priority_state')" in text
    if not already_hardened:
        candidates = [
            match
            for match in resume_matches
            if "priority_controller.load_state_dict" in match.group(0)
            and "old checkpoint has no adaptive state" in match.group(0)
        ]
        if len(candidates) != 1:
            die(
                "adaptive-state resume block: expected one semantic match, "
                f"found {len(candidates)}"
            )
        match = candidates[0]
        indent = match.group("indent")
        replacement = (
            f"{indent}if _adaptive_any and ckpt.get('adaptive_priority_state') is not None:\n"
            f"{indent}    priority_controller.load_state_dict(\n"
            f"{indent}        ckpt['adaptive_priority_state'], strict=True\n"
            f"{indent}    )\n"
            f"{indent}    print(' [resume] restored adaptive map-priority state')\n"
            f"{indent}elif _adaptive_any:\n"
            f"{indent}    print(' [resume] old checkpoint has no adaptive state; maps start uniform')\n"
            f"{indent}else:\n"
            f"{indent}    print(' [resume] adaptive priority disabled; source priority state skipped')\n"
        )
        text = text[: match.start()] + replacement + text[match.end() :]

    if "def _extra_checkpoint_state():" not in text:
        die("runner lacks _extra_checkpoint_state")
    if "state = {\n                'tactical_mixture_metadata'" not in text:
        extra_pattern = re.compile(
            r"(?P<indent>^[ \t]*)def _extra_checkpoint_state\(\):\n"
            r"(?P<body>(?:(?P=indent)[ \t]+.*\n)+?)"
            r"(?=(?P=indent)checkpointer\s*=)",
            re.MULTILINE,
        )
        matches = list(extra_pattern.finditer(text))
        candidates = [
            match
            for match in matches
            if "adaptive_priority_state" in match.group(0)
            and "tactical_mixture_metadata" in match.group(0)
        ]
        if len(candidates) != 1:
            die(
                "extra checkpoint state block: expected one semantic match, "
                f"found {len(candidates)}"
            )
        match = candidates[0]
        indent = match.group("indent")
        i1 = indent + "    "
        i2 = i1 + "    "
        replacement = (
            f"{indent}def _extra_checkpoint_state():\n"
            f"{i1}state = {{\n"
            f"{i2}'tactical_mixture_metadata': agent.tactical_metadata(),\n"
            f"{i1}}}\n"
            f"{i1}if _adaptive_any:\n"
            f"{i2}state.update({{\n"
            f"{i2}    'adaptive_priority_schema': 1,\n"
            f"{i2}    'adaptive_priority_state': priority_controller.state_dict(),\n"
            f"{i2}}})\n"
            f"{i1}return state\n"
        )
        text = text[: match.start()] + replacement + text[match.end() :]

    ast.parse(text)
    return text

def patch_validation_trainer(text: str) -> str:
    if HARDENING_MARKER in text:
        return text
    if "best_val_macro_winrate.pt" not in text:
        die("validation trainer lacks best checkpoint save")

    tree = ast.parse(text)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "save":
            continue
        source = ast.get_source_segment(text, node) or ""
        if "best_val_macro_winrate.pt" in source and "agent_state_dict" in source:
            matches.append(node)
    if len(matches) != 1:
        die(
            "best checkpoint save: expected one AST match, "
            f"found {len(matches)}"
        )

    current = matches[0]
    while current is not None and not isinstance(current, ast.stmt):
        current = parents.get(current)
    if current is None or current.end_lineno is None:
        die("best checkpoint save: could not locate enclosing statement")

    lines = text.splitlines(keepends=True)
    start_line = current.lineno - 1
    end_line = current.end_lineno
    original = lines[start_line]
    indent = original[: len(original) - len(original.lstrip())]
    i1 = indent + "    "
    replacement = (
        f"{indent}best_payload = {{\n"
        f'{i1}"agent_state_dict": agent.state_dict(),\n'
        f'{i1}"val_macro_win_rate": wr,\n'
        f'{i1}"val_macro_original_return": ret,\n'
        f'{i1}"step": int(train_step),\n'
        f'{i1}"obs_mode": self._val_obs_mode,\n'
        f"{indent}}}\n"
        f'{indent}if hasattr(agent, "tactical_metadata"):\n'
        f'{i1}best_payload["tactical_mixture_metadata"] = (\n'
        f"{i1}    agent.tactical_metadata()\n"
        f"{i1})\n"
        f"{indent}torch.save(\n"
        f"{i1}best_payload,\n"
        f'{i1}self._logdir / "best_val_macro_winrate.pt",\n'
        f"{indent})\n"
    )
    text = "".join(lines[:start_line]) + replacement + "".join(lines[end_line:])

    # Insert the source marker after the import block without assuming a
    # particular import (the original bundle previously assumed ``import
    # math`` existed).
    marker_tree = ast.parse(text)
    imports = [
        node
        for node in marker_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not imports:
        die("validation trainer has no top-level import block for marker insertion")
    last_import = max(imports, key=lambda node: node.end_lineno or node.lineno)
    marker_lines = text.splitlines(keepends=True)
    marker_lines.insert(last_import.end_lineno, "\n" + HARDENING_MARKER + "\n")
    text = "".join(marker_lines)
    if HARDENING_MARKER not in text:
        die("failed to insert validation hardening marker")
    ast.parse(text)
    return text


PATCHERS = {
    "external/r2dreamer/dreamer.py": patch_dreamer,
    "scripts/train_r2dreamer_smaclite_multimap.py": patch_runner,
    "src/smacdreamer/validation_trainer.py": patch_validation_trainer,
}


def build_config(
    repo: pathlib.Path,
    source_config: str,
) -> tuple[pathlib.Path, object]:
    """Build and validate the hardened config without modifying the repo.

    Configuration used to be generated only after source files were replaced.
    A malformed source YAML could therefore leave a partial installation.  The
    installer now resolves and validates the complete output during dry-run and
    before creating the backup or writing any target file.
    """
    source = pathlib.Path(source_config).expanduser()
    if not source.is_absolute():
        source = repo / source
    source = source.resolve()
    if not source.is_file():
        die(f"source tactical config missing: {source}")
    try:
        source.relative_to(repo.resolve())
    except ValueError:
        die("source tactical config must live inside the target repo")

    try:
        cfg = OmegaConf.load(source)
    except Exception as exc:
        die(f"could not parse source tactical config {source}: {exc}")

    if str((cfg.get("world_model") or {}).get("backend", "")) != "jepa":
        die("source tactical config must use world_model.backend=jepa")
    if str((cfg.get("reward") or {}).get("name", "")) != "dense_v3":
        die("source tactical config must use reward.name=dense_v3")
    if int(cfg.get("imag_horizon", -1)) != 5:
        die("source tactical config must retain imag_horizon=5")
    source_tactical = cfg.get("tactical_mixture") or OmegaConf.create({})
    if not bool(source_tactical.get("enabled", False)):
        die("source config is not an enabled Tactical Mixture v1 config")
    if int(source_tactical.get("duration", -1)) != 1:
        die("source tactical config must have duration=1")

    target = repo / "configs/r2_2100_jepa_tactical_mixture_hardened.yaml"
    tactical = source_tactical
    tactical.enabled = True
    tactical.num_tactics = int(tactical.get("num_tactics", 4))
    tactical.embedding_dim = int(tactical.get("embedding_dim", 16))
    tactical.hidden_dim = int(tactical.get("hidden_dim", 128))
    tactical.duration = 1
    tactical.tactic_pg_scale = float(tactical.get("tactic_pg_scale", 1.0))
    tactical.tactic_entropy_scale = 1.0e-4
    tactical.collapse_loss_scale = 1.0e-3
    tactical.balance_loss_scale = 0.0
    tactical.max_usage_target = 0.80
    tactical.min_effective_tactics = 2.0
    tactical.symmetry_break_std = 1.0e-2
    tactical.eval_confidence_threshold = 0.55
    tactical.freeze_base_actor = True
    tactical.freeze_feature_adapter = True
    tactical.effect_loss_scale = float(tactical.get("effect_loss_scale", 1.0e-3))
    tactical.effect_target = float(tactical.get("effect_target", 0.02))
    tactical.residual_guard_scale = 1.0e-3
    tactical.max_residual_to_base = 1.0
    tactical.max_abs_residual_logit = 4.0
    tactical.max_effect_states = int(tactical.get("max_effect_states", 256))
    cfg.tactical_mixture = tactical

    adaptive = cfg.get("adaptive_priority") or OmegaConf.create({})
    adaptive.enabled = False
    if adaptive.get("map") is None:
        adaptive.map = OmegaConf.create({})
    if adaptive.get("sequence") is None:
        adaptive.sequence = OmegaConf.create({})
    adaptive.map.enabled = False
    adaptive.sequence.enabled = False
    cfg.adaptive_priority = adaptive
    if str(cfg.get("sampling_mode", "")) == "adaptive_priority":
        cfg.sampling_mode = "shuffled_round_robin"

    # Force replay storage to be relative to the newly created log directory.
    # An absolute scratch_dir can silently reopen stale TorchRL memmap state.
    if cfg.get("buffer") is None:
        cfg.buffer = OmegaConf.create({})
    cfg.buffer.scratch_dir = "replay"

    if cfg.get("validation") is None:
        cfg.validation = OmegaConf.create({})
    cfg.validation.run_at_start = False
    cfg.validation.every = 200000
    if cfg.get("wandb") is None:
        cfg.wandb = OmegaConf.create({})
    cfg.wandb.run_name = "adaptive_best_tactical_mixture_hardened_2m"

    # Force resolution now so interpolation/schema errors fail before writes.
    try:
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except Exception as exc:
        die(f"hardened config cannot be fully resolved: {exc}")
    return target, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--source-config",
        default="configs/r2_2100_jepa_tactical_mixture.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verify_bundle_manifest()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    required = [repo / relative for relative in PATCHERS]
    required.append(repo / "external/r2dreamer/tactical_policy.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        die(f"required installed tactical files missing: {missing}")

    try:
        git_root = pathlib.Path(
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        )
    except Exception:
        die(f"{repo} is not inside a Git worktree")

    print(f"[INFO] Git root: {git_root}")
    print(f"[INFO] Target subtree: {repo}")

    patched: dict[str, str] = {}
    for relative, patcher in PATCHERS.items():
        source = repo / relative
        patched[relative] = patcher(source.read_text(encoding="utf-8"))
        ast.parse(patched[relative], filename=str(source))

    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(PAYLOAD)
        if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if source.suffix == ".py":
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        elif source.suffix == ".sh":
            subprocess.run(["bash", "-n", str(source)], check=True)

    target, hardened_cfg = build_config(repo, args.source_config)

    if args.dry_run:
        print("[OK] hardening dry-run matched all source anchors, parsed all ASTs, and resolved the output config")
        return

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo.parent / f"{repo.name}_tactical_hardening_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    source_config_path = pathlib.Path(args.source_config).expanduser()
    if not source_config_path.is_absolute():
        source_config_path = repo / source_config_path
    source_config_path = source_config_path.resolve()
    try:
        source_config_relative = source_config_path.relative_to(repo)
    except ValueError:
        die("source tactical config escaped the target repo after validation")

    backup_entries: list[tuple[pathlib.Path, pathlib.Path]] = [
        (repo / relative, pathlib.Path(relative)) for relative in PATCHERS
    ]
    backup_entries.extend(
        [
            (
                repo / "external/r2dreamer/tactical_policy.py",
                pathlib.Path("external/r2dreamer/tactical_policy.py"),
            ),
            (source_config_path, source_config_relative),
        ]
    )
    existing_target = repo / "configs/r2_2100_jepa_tactical_mixture_hardened.yaml"
    if existing_target.is_file() and existing_target != source_config_path:
        backup_entries.append(
            (
                existing_target,
                pathlib.Path("configs/r2_2100_jepa_tactical_mixture_hardened.yaml"),
            )
        )
    # Preserve every payload destination that already exists. This makes a
    # repeated/partial installation reversible instead of silently destroying
    # an earlier version of a hardening script or test.
    payload_relatives = [
        source.relative_to(PAYLOAD)
        for source in PAYLOAD.rglob("*")
        if source.is_file()
        and not any(part in {"__pycache__", ".pytest_cache"} for part in source.relative_to(PAYLOAD).parts)
        and source.suffix not in {".pyc", ".pyo"}
    ]
    for relative in payload_relatives:
        destination = repo / relative
        if destination.is_file():
            backup_entries.append((destination, relative))

    copied_backup_destinations: set[pathlib.Path] = set()
    for source, relative in backup_entries:
        if not source.is_file() or relative in copied_backup_destinations:
            continue
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_backup_destinations.add(relative)

    introduced = [
        str(relative)
        for relative in payload_relatives
        if not (backup / relative).is_file()
    ]
    hardened_config_relative = pathlib.Path(
        "configs/r2_2100_jepa_tactical_mixture_hardened.yaml"
    )
    if not (backup / hardened_config_relative).is_file():
        introduced.append(str(hardened_config_relative))
    manifest = {
        "schema_version": 1,
        "repo": str(repo),
        "introduced_files": introduced,
        "backed_up_files": sorted(str(path) for path in copied_backup_destinations),
    }
    (backup / "hardening_backup_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    try:
        for relative, content in patched.items():
            (repo / relative).write_text(content, encoding="utf-8")

        for source in PAYLOAD.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(PAYLOAD)
            if any(
                part in {"__pycache__", ".pytest_cache"}
                for part in relative.parts
            ):
                continue
            if source.suffix in {".pyc", ".pyo"}:
                continue
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if destination.suffix == ".sh" or (
                destination.suffix == ".py"
                and destination.parent.name == "scripts"
            ):
                destination.chmod(destination.stat().st_mode | 0o111)

        OmegaConf.save(hardened_cfg, target)
    except Exception as exc:
        # Best-effort transaction rollback. The untouched backup remains even
        # if rollback itself encounters a secondary filesystem error.
        rollback_errors: list[str] = []
        for relative_text in manifest["backed_up_files"]:
            relative = pathlib.Path(relative_text)
            source = backup / relative
            destination = repo / relative
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"restore {relative}: {rollback_exc}"
                )
        for relative_text in manifest["introduced_files"]:
            destination = repo / relative_text
            try:
                if destination.is_file() or destination.is_symlink():
                    destination.unlink()
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"remove {relative_text}: {rollback_exc}"
                )
        detail = (
            "; rollback errors=" + " | ".join(rollback_errors)
            if rollback_errors
            else "; rollback completed"
        )
        die(f"installation write failed: {exc}{detail}")

    print("[OK] installed Tactical Mixture hardening v1.1")
    print(f"[OK] backup: {backup}")
    print(f"[OK] generated config: {target}")
    print("[NEXT] run scripts/static_audit_tactical_hardening.sh")


if __name__ == "__main__":
    main()
