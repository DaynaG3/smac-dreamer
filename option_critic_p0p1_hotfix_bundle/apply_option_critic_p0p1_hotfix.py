#!/usr/bin/env python3
"""Fail-closed post-install hotfix for the already integrated Option-Critic v2."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess

from omegaconf import OmegaConf

BUNDLE = pathlib.Path(__file__).resolve().parent
PAYLOAD = BUNDLE / "payload"
MANIFEST = BUNDLE / "MANIFEST.sha256.json"
V2_MARKER = "# OPTION_CRITIC_HIERARCHY_V2"
V3_MARKER = "# OPTION_CRITIC_P0P1_HOTFIX_V3"

REPLACED = (
    "external/r2dreamer/hierarchical_options.py",
    "external/r2dreamer/hierarchical_dreamer.py",
    "external/r2dreamer/option_critic.py",
    "scripts/audit_option_critic_hierarchy.py",
    "scripts/static_audit_option_critic_hierarchy.sh",
    "scripts/assert_option_critic_metrics.py",
    "scripts/run_option_critic_2m.sh",
    "tests/test_hierarchical_options.py",
    "tests/test_option_critic_math.py",
    "tests/test_hierarchical_auxiliary.py",
    "tests/test_hierarchy_migration.py",
    "external/r2dreamer/dreamer.py",
    "external/r2dreamer/tools.py",
)

PAYLOAD_MAP = {
    "external/r2dreamer/hierarchical_options.py": "external/r2dreamer/hierarchical_options.py",
    "external/r2dreamer/hierarchical_dreamer.py": "external/r2dreamer/hierarchical_dreamer.py",
    "external/r2dreamer/option_critic.py": "external/r2dreamer/option_critic.py",
    "scripts/audit_option_critic_hierarchy.py": "scripts/audit_option_critic_p0p1.py",
    "scripts/static_audit_option_critic_hierarchy.sh": "scripts/static_audit_option_critic_p0p1.sh",
    "scripts/assert_option_critic_metrics.py": "scripts/assert_option_critic_p0p1_metrics.py",
    "scripts/run_option_critic_2m.sh": "scripts/run_option_critic_p0p1_2m.sh",
    "tests/test_hierarchical_options.py": "tests/test_hierarchical_options.py",
    "tests/test_option_critic_math.py": "tests/test_option_critic_math.py",
    "tests/test_hierarchical_auxiliary.py": "tests/test_hierarchical_auxiliary.py",
    "tests/test_hierarchy_migration.py": "tests/test_hierarchy_migration.py",
}

INTRODUCED = (
    "configs/r2_2100_jepa_option_critic_8_v3_p0p1.yaml",
    "scripts/check_option_critic_win_guard.py",
)


def die(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        die("bundle manifest missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(entries, dict):
        die("invalid bundle manifest")
    for rel, expected in entries.items():
        path = BUNDLE / rel
        if not path.is_file():
            die(f"bundle file missing: {rel}")
        actual = sha256(path)
        if actual != expected:
            die(f"bundle hash mismatch: {rel}: {actual} != {expected}")
    print(f"[OK] verified {len(entries)} hotfix bundle hashes")


def method_segment(text: str, name: str) -> tuple[int, int, str]:
    tree = ast.parse(text)
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        die(f"expected one {name}() method, found {len(nodes)}")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return start, end, "".join(lines[node.lineno - 1 : node.end_lineno])


def patch_dreamer(text: str) -> str:
    if V3_MARKER in text:
        die("dreamer.py already contains the P0/P1 hotfix")
    if V2_MARKER not in text:
        die("dreamer.py is not the integrated Option-Critic v2 source")

    import_anchor = "from hierarchical_dreamer import (\n"
    if text.count(import_anchor) != 1:
        die("hierarchical_dreamer import block is ambiguous")
    first_item = "    build_hierarchical_modules, clone_and_freeze_hierarchy,\n"
    if text.count(first_item) != 1:
        die("hierarchy import item anchor is missing")
    text = text.replace(
        first_item,
        "    apply_hierarchy_gradient_guards,\n" + first_item,
        1,
    )

    start, end, update = method_segment(text, "update")
    update_lines = update.splitlines(keepends=True)
    matches = [
        index for index, line in enumerate(update_lines)
        if line.split("#", 1)[0].strip().startswith(
            "self._scaler.unscale_("
        )
        and line.split("#", 1)[0].strip().endswith(")")
    ]
    if len(matches) != 1:
        candidates = [
            line.strip() for line in update_lines
            if "unscale_" in line or ".step(" in line
        ]
        die(
            "optimizer unscale anchor count="
            f"{len(matches)}; candidates={candidates}"
        )
    index = matches[0]
    line = update_lines[index]
    indent = line[: len(line) - len(line.lstrip())]
    guard = (
        f"{indent}if self.hierarchical_enabled:\n"
        f"{indent}    apply_hierarchy_gradient_guards(self)\n"
    )
    update_lines.insert(index, guard)
    update = "".join(update_lines)
    text = text[:start] + update + text[end:]
    text = text.replace(V2_MARKER, V2_MARKER + "\n" + V3_MARKER, 1)
    ast.parse(text)
    return text


def patch_tools(text: str) -> str:
    corrected = """    x = x.detach()\n    if x.dtype == torch.bfloat16:\n        x = x.float()\n    return x.cpu().numpy()\n"""
    if corrected in text:
        return text
    old = "    return x.detach().cpu().numpy()\n"
    if text.count(old) != 1:
        die("tools.to_np BF16 anchor is missing or ambiguous")
    changed = text.replace(old, corrected, 1)
    ast.parse(changed)
    return changed


def build_config(repo: pathlib.Path) -> tuple[pathlib.Path, object]:
    source = repo / "configs/r2_2100_jepa_option_critic_8_v2.yaml"
    if not source.is_file():
        die(f"integrated v2 config missing: {source}")
    cfg = OmegaConf.load(source)
    if str(cfg.world_model.backend) != "jepa":
        die("world_model.backend must remain jepa")
    if str(cfg.reward.name) != "dense_v3":
        die("reward.name changed unexpectedly")
    if not bool(cfg.hierarchical_options.enabled):
        die("source config is not Option-Critic enabled")

    cfg.hierarchical_options = OmegaConf.create({
        "enabled": True,
        "num_options": 8,
        "option_embedding_dim": 16,
        "age_embedding_dim": 8,
        "hidden_dim": 128,
        "min_duration": 1,
        "max_duration": 20,
        "commitment_warmup_steps": 100000,
        "commitment_full_steps": 300000,
        "commitment_reselect_initial": 1.0,
        "commitment_reselect_final": 0.0,
        "initial_termination_probability": 0.10,
        "termination_warmup_steps": 100000,
        "termination_full_steps": 300000,
        "termination_max_probability_during_ramp": 0.30,
        "termination_max_probability_final": 0.80,
        "termination_cap_full_steps": 500000,
        "termination_margin_normalized": 0.02,
        "termination_loss_scale": 0.05,
        "termination_entropy_scale": 0.0,
        "termination_collapse_scale": 0.0,
        "termination_mean_min": 0.02,
        "termination_mean_max": 0.60,
        "termination_advantage_clip": 1.0,
        "termination_min_advantage_magnitude": 0.01,
        "termination_max_target_disagreement": 0.25,
        "termination_unimix": 0.02,
        "eval_sample_termination": False,
        "eval_termination_hazard_threshold": 1.0,
        "manager_unimix_initial": 0.0,
        "manager_unimix_final": 0.02,
        "manager_unimix_decay_steps": 300000,
        "manager_pg_scale": 1.0,
        "manager_pg_warmup_steps": 100000,
        "manager_pg_full_steps": 300000,
        "manager_entropy_scale": 0.0,
        "manager_collapse_scale": 0.0,
        "manager_mi_target_normalized": 0.10,
        "manager_mi_scale": 0.0,
        "max_usage_target": 0.75,
        "min_effective_options": 1.0,
        "worker_pg_scale": 1.0,
        "worker_pg_warmup_steps": 100000,
        "worker_pg_full_steps": 300000,
        "worker_entropy_scale": 0.0,
        "worker_scale_initial": 0.25,
        "worker_scale_warmup_steps": 0,
        "worker_scale_full_steps": 1,
        "worker_scale_max": 0.25,
        "max_abs_residual_logit": 2.0,
        "max_residual_to_base": 0.25,
        "residual_guard_scale": 0.05,
        "base_kl_target": 0.01,
        "base_kl_tail_target": 0.03,
        "base_kl_tail_fraction": 0.10,
        "base_kl_tail_relative_scale": 1.0,
        "base_kl_scale": 0.25,
        "action_preservation_confidence": 0.80,
        "action_preservation_margin": 0.05,
        "action_preservation_scale": 0.25,
        "action_diversity_target": 0.002,
        "action_diversity_scale": 0.0,
        "residual_cosine_target": 0.95,
        "residual_cosine_scale": 0.0,
        "max_diversity_states": 128,
        "max_diversity_pairs": 12,
        "option_critic_scale": 1.0,
        "hierarchy_value_scale": 0.5,
        "slow_target_update": 1,
        "slow_target_fraction": 0.005,
        "freeze_base_actor": True,
        "freeze_feature_adapter": True,
        "world_model_grad_scale_initial": 0.0,
        "world_model_grad_scale_final": 0.0,
        "world_model_grad_warmup_steps": 200000,
        "world_model_grad_full_steps": 500000,
        "imag_horizon_initial_max": 8,
        "imag_horizon_final_max": 10,
        "imag_horizon_window": 4,
        "imag_horizon_ramp_steps": 500000,
    })
    cfg.tactical_mixture.enabled = False
    cfg.sampling_mode = "shuffled_round_robin"
    cfg.adaptive_priority.enabled = False
    cfg.adaptive_priority.map.enabled = False
    cfg.adaptive_priority.sequence.enabled = False
    cfg.buffer.scratch_dir = "replay"
    cfg.validation.run_at_start = True
    cfg.validation.every = 100000
    if "compile" in cfg:
        cfg.compile = False
    if "model" in cfg and "compile" in cfg.model:
        cfg.model.compile = False
    if "wandb" in cfg:
        cfg.wandb.run_name = "tactical_v12_option_critic_v3_p0p1_2m"
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return repo / "configs/r2_2100_jepa_option_critic_8_v3_p0p1.yaml", cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verify_manifest()
    repo = args.repo.expanduser().resolve()
    try:
        git_root = pathlib.Path(subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip())
    except Exception:
        die(f"{repo} is not inside a Git worktree")
    print(f"[INFO] Git root: {git_root}")
    print(f"[INFO] Target subtree: {repo}")

    for rel in REPLACED:
        if not (repo / rel).is_file():
            die(f"required installed file missing: {rel}")
    for destination, source in PAYLOAD_MAP.items():
        path = PAYLOAD / source
        if not path.is_file():
            die(f"payload file missing: {source}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    dreamer_changed = patch_dreamer(
        (repo / "external/r2dreamer/dreamer.py").read_text(encoding="utf-8")
    )
    tools_changed = patch_tools(
        (repo / "external/r2dreamer/tools.py").read_text(encoding="utf-8")
    )
    target_config, config = build_config(repo)
    introduced_paths = [repo / rel for rel in INTRODUCED]
    collisions = [str(path.relative_to(repo)) for path in introduced_paths if path.exists()]
    if collisions:
        die(f"refusing to overwrite existing hotfix-introduced files: {collisions}")

    if args.dry_run:
        print("[OK] P0/P1 hotfix dry-run matched integrated v2, parsed all Python, and resolved v3 config")
        return

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo.parent / f"{repo.name}_option_critic_p0p1_hotfix_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    try:
        for rel in REPLACED:
            src = repo / rel
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        (repo / "external/r2dreamer/dreamer.py").write_text(
            dreamer_changed, encoding="utf-8"
        )
        (repo / "external/r2dreamer/tools.py").write_text(
            tools_changed, encoding="utf-8"
        )
        for destination, source in PAYLOAD_MAP.items():
            src = PAYLOAD / source
            dst = repo / destination
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.suffix == ".sh":
                dst.chmod(0o755)

        OmegaConf.save(config, target_config)
        guard_src = PAYLOAD / "scripts/check_option_critic_win_guard.py"
        guard_dst = repo / "scripts/check_option_critic_win_guard.py"
        shutil.copy2(guard_src, guard_dst)
        guard_dst.chmod(0o755)

        manifest = {
            "schema_version": 1,
            "repo": str(repo),
            "backed_up_files": list(REPLACED),
            "backed_up_sha256": {
                rel: sha256(backup / rel) for rel in REPLACED
            },
            "introduced_files": list(INTRODUCED),
        }
        (backup / "option_critic_p0p1_hotfix_backup_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        for rel in REPLACED:
            src = backup / rel
            if src.exists():
                dst = repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for rel in INTRODUCED:
            path = repo / rel
            if path.exists():
                path.unlink()
        raise

    print("[OK] installed Option-Critic P0/P1 hotfix v3")
    print(f"[OK] backup: {backup}")
    print(f"[OK] config: {target_config}")
    print("[NEXT] run scripts/static_audit_option_critic_hierarchy.sh with the v3 config")


if __name__ == "__main__":
    main()
