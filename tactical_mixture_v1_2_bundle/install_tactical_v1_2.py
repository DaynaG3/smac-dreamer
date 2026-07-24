#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

MARKER = "# TACTICAL_MIXTURE_V1_2_CENTERED_TRUST_REGION"


def die(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        die(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def patch_dreamer(text: str) -> str:
    if MARKER in text:
        die("dreamer.py already contains Tactical Mixture v1.2")
    if "tactical_mixture_v1_1" not in text:
        die("dreamer.py is not the audited v1.1 hardened source")

    old_arch = '''            if architecture not in (
                "tactical_mixture_v1",
                "tactical_mixture_v1_1",
            ):
'''
    new_arch = '''            if architecture not in (
                "tactical_mixture_v1",
                "tactical_mixture_v1_1",
                "tactical_mixture_v1_2",
            ):
'''
    text = replace_once(text, old_arch, new_arch, "checkpoint architecture")

    old_keys = '''                "max_residual_to_base",
                "max_abs_residual_logit",
            ):
'''
    new_keys = '''                "max_residual_to_base",
                "max_abs_residual_logit",
                "selector_symmetry_break_std",
                "residual_scale",
                "min_selector_mi_normalized",
                "base_kl_target",
                "base_kl_scale",
            ):
'''
    text = replace_once(text, old_keys, new_keys, "checkpoint metadata keys")

    old_loss = '''            residual_guard_loss = torch.relu(
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
'''
    new_loss = '''            residual_guard_loss = torch.relu(
                residual_ratio
                - torch.as_tensor(
                    tactical.max_residual_to_base,
                    device=residual_ratio.device,
                    dtype=residual_ratio.dtype,
                )
            ).square()
            base_kl_loss = effect_stats["base_kl_loss"]
            losses["policy"] = (
                primitive_policy_loss
                + tactic_policy_loss
                + tactical.collapse_loss_scale * collapse_loss
                + tactical.effect_loss_scale * effect_loss
                + tactical.residual_guard_scale * residual_guard_loss
                + tactical.base_kl_scale * base_kl_loss
            )
'''
    text = replace_once(text, old_loss, new_loss, "trust-region policy loss")

    old_metrics = '''            metrics["tactic/residual_guard_loss"] = residual_guard_loss
            metrics["tactic/usage_max"] = tactic_stats["usage_max"]
'''
    new_metrics = '''            metrics["tactic/residual_guard_loss"] = residual_guard_loss
            metrics["tactic/base_kl_loss"] = base_kl_loss
            metrics["tactic/base_kl_mean"] = effect_stats["base_kl_mean"]
            metrics["tactic/base_kl_max"] = effect_stats["base_kl_max"]
            metrics["tactic/action_flip_rate"] = effect_stats[
                "action_flip_rate"
            ]
            metrics["tactic/mi_shortfall"] = tactic_stats["mi_shortfall"]
            metrics["tactic/usage_max"] = tactic_stats["usage_max"]
'''
    text = replace_once(text, old_metrics, new_metrics, "trust-region metrics")

    # Marker near the top keeps installation idempotence explicit.
    lines = text.splitlines(keepends=True)
    insert_at = 0
    while insert_at < len(lines) and (
        lines[insert_at].startswith("#!")
        or lines[insert_at].startswith("# -*-")
    ):
        insert_at += 1
    lines.insert(insert_at, MARKER + "\n")
    patched = "".join(lines)
    ast.parse(patched)
    return patched


def build_config(repo: Path, source_rel: str):
    source = (repo / source_rel).resolve()
    if not source.is_file():
        die(f"source hardened config missing: {source}")
    try:
        source.relative_to(repo.resolve())
    except ValueError:
        die("source config escaped repository")
    cfg = OmegaConf.load(source)
    if str(cfg.world_model.backend) != "jepa":
        die("world_model.backend must be jepa")
    if str(cfg.reward.name) != "dense_v3":
        die("reward.name must be dense_v3")
    if int(cfg.imag_horizon) != 5:
        die("imag_horizon must remain 5")

    tactical = cfg.tactical_mixture
    tactical.enabled = True
    tactical.num_tactics = 2
    tactical.embedding_dim = 16
    tactical.hidden_dim = 128
    tactical.duration = 1
    tactical.tactic_pg_scale = 1.0
    tactical.tactic_entropy_scale = 0.0
    tactical.collapse_loss_scale = 0.10
    tactical.balance_loss_scale = 0.0
    tactical.effect_loss_scale = 0.10
    tactical.effect_target = 0.002
    tactical.residual_guard_scale = 0.05
    tactical.max_residual_to_base = 0.25
    tactical.max_abs_residual_logit = 2.0
    tactical.max_effect_states = 256
    tactical.symmetry_break_std = 1.0e-2
    tactical.selector_symmetry_break_std = 1.0e-3
    tactical.residual_scale = 0.25
    tactical.min_selector_mi_normalized = 0.05
    tactical.base_kl_target = 0.02
    tactical.base_kl_scale = 0.10
    tactical.max_usage_target = 0.75
    tactical.min_effective_tactics = 1.60
    tactical.eval_confidence_threshold = 0.70
    tactical.freeze_base_actor = True
    tactical.freeze_feature_adapter = True

    cfg.sampling_mode = "shuffled_round_robin"
    cfg.adaptive_priority.enabled = False
    cfg.adaptive_priority.map.enabled = False
    cfg.adaptive_priority.sequence.enabled = False
    cfg.buffer.scratch_dir = "replay"
    cfg.validation.run_at_start = False
    cfg.validation.every = 200000
    if "wandb" in cfg:
        cfg.wandb.run_name = "adaptive_best_tactical_v1_2_centered_2m"

    target = repo / "configs/r2_2100_jepa_tactical_mixture_v1_2.yaml"
    # Resolve before any write.
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return target, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--source-config",
        default="configs/r2_2100_jepa_tactical_mixture_hardened.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    bundle = Path(__file__).resolve().parent
    payload = bundle / "payload"
    required = [
        repo / "external/r2dreamer/dreamer.py",
        repo / "external/r2dreamer/tactical_policy.py",
        repo / "scripts/train_r2dreamer_smaclite_multimap.py",
        payload / "external/r2dreamer/tactical_policy.py",
        payload / "scripts/audit_tactical_v1_2.py",
        payload / "scripts/static_audit_tactical_v1_2.sh",
        payload / "scripts/run_tactical_v1_2_2m.sh",
        payload / "scripts/assert_tactical_v1_2_metrics.py",
        payload / "tests/test_tactical_policy_v1_2.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        die(f"missing required files: {missing}")

    dreamer_path = repo / "external/r2dreamer/dreamer.py"
    patched_dreamer = patch_dreamer(dreamer_path.read_text(encoding="utf-8"))
    compile(patched_dreamer, str(dreamer_path), "exec")

    policy_text = (payload / "external/r2dreamer/tactical_policy.py").read_text(
        encoding="utf-8"
    )
    compile(policy_text, "tactical_policy.py", "exec")
    target_config, cfg = build_config(repo, args.source_config)

    if args.dry_run:
        print("[OK] v1.2 dry-run matched source, parsed patched ASTs, and resolved config")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo.parent / f"{repo.name}_tactical_v1_2_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    backed_up = [
        "external/r2dreamer/dreamer.py",
        "external/r2dreamer/tactical_policy.py",
    ]
    introduced = [
        "configs/r2_2100_jepa_tactical_mixture_v1_2.yaml",
        "scripts/audit_tactical_v1_2.py",
        "scripts/static_audit_tactical_v1_2.sh",
        "scripts/run_tactical_v1_2_2m.sh",
        "scripts/assert_tactical_v1_2_metrics.py",
        "tests/test_tactical_policy_v1_2.py",
    ]

    try:
        for rel in backed_up:
            src = repo / rel
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        dreamer_path.write_text(patched_dreamer, encoding="utf-8")
        shutil.copy2(
            payload / "external/r2dreamer/tactical_policy.py",
            repo / "external/r2dreamer/tactical_policy.py",
        )
        OmegaConf.save(cfg, target_config)
        for rel in introduced[1:]:
            src = payload / rel
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.suffix == ".sh":
                dst.chmod(0o755)

        for path in (
            dreamer_path,
            repo / "external/r2dreamer/tactical_policy.py",
            repo / "scripts/audit_tactical_v1_2.py",
            repo / "scripts/assert_tactical_v1_2_metrics.py",
            repo / "tests/test_tactical_policy_v1_2.py",
        ):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

        manifest = {
            "schema_version": 1,
            "repo": str(repo),
            "backed_up_files": backed_up,
            "introduced_files": introduced,
        }
        (backup / "v1_2_backup_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        for rel in backed_up:
            src = backup / rel
            if src.exists():
                dst = repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for rel in introduced:
            path = repo / rel
            if path.exists():
                path.unlink()
        raise

    print("[OK] installed Tactical Mixture v1.2 centered trust-region")
    print(f"[OK] backup: {backup}")
    print(f"[OK] config: {target_config}")
    print("[NEXT] run scripts/static_audit_tactical_v1_2.sh")


if __name__ == "__main__":
    main()
