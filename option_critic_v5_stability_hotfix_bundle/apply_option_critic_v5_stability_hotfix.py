#!/usr/bin/env python3
"""Fail-closed post-install v5 stability patch for integrated Option-Critic v4."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess

from omegaconf import OmegaConf

BUNDLE = pathlib.Path(__file__).resolve().parent
PAYLOAD = BUNDLE / "payload"
MANIFEST = BUNDLE / "MANIFEST.sha256.json"
V4_ARCH = 'ARCHITECTURE = "dreamer_option_critic_v4_p1_final"'
V5_ARCH = 'ARCHITECTURE = "dreamer_option_critic_v5_stability"'

REPLACED = (
    "external/r2dreamer/hierarchical_options.py",
    "external/r2dreamer/hierarchical_dreamer.py",
    "external/r2dreamer/option_critic.py",
    "tests/test_hierarchical_options.py",
    "tests/test_option_critic_math.py",
    "tests/test_hierarchical_auxiliary.py",
    "tests/test_hierarchy_migration.py",
    "scripts/run_exp45_full_train_eval_resilient.sh",
)

PAYLOAD_MAP = {
    "external/r2dreamer/hierarchical_options.py": "external/r2dreamer/hierarchical_options.py",
    "external/r2dreamer/hierarchical_dreamer.py": "external/r2dreamer/hierarchical_dreamer.py",
    "external/r2dreamer/option_critic.py": "external/r2dreamer/option_critic.py",
    "tests/test_hierarchical_options.py": "tests/test_hierarchical_options.py",
    "tests/test_option_critic_math.py": "tests/test_option_critic_math.py",
    "tests/test_hierarchical_auxiliary.py": "tests/test_hierarchical_auxiliary.py",
    "tests/test_hierarchy_migration.py": "tests/test_hierarchy_migration.py",
    "scripts/run_exp45_full_train_eval_resilient.sh": "scripts/run_exp45_full_train_eval_resilient.sh",
}

INTRODUCED_MAP = {
    "configs/r2_2100_jepa_option_critic_2_v5_stability_1m.yaml": None,
    "scripts/audit_option_critic_v5_stability.py": "scripts/audit_option_critic_v5_stability.py",
    "scripts/static_audit_option_critic_v5_stability.sh": "scripts/static_audit_option_critic_v5_stability.sh",
    "scripts/assert_option_critic_v5_metrics.py": "scripts/assert_option_critic_v5_metrics.py",
    "scripts/run_option_critic_v5_stability_1m.sh": "scripts/run_option_critic_v5_stability_1m.sh",
    "scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh": "scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh",
}


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
    print(f"[OK] verified {len(entries)} v5 stability bundle hashes")


def build_config(repo: pathlib.Path):
    source = repo / "configs/r2_2100_jepa_option_critic_8_v4_p1_final_1m.yaml"
    if not source.is_file():
        die(f"integrated v4 config missing: {source}")
    cfg = OmegaConf.load(source)
    if str(cfg.world_model.backend) != "jepa":
        die("world_model.backend must remain jepa")
    if str(cfg.reward.name) != "dense_v3":
        die("reward.name changed unexpectedly")
    if not bool(cfg.hierarchical_options.enabled):
        die("source config is not Option-Critic enabled")

    h = OmegaConf.to_container(cfg.hierarchical_options, resolve=True)
    if not isinstance(h, dict):
        die("hierarchical_options config is not a mapping")
    h.update({
        "enabled": True,
        "num_options": 2,
        "min_duration": 1,
        "max_duration": 8,
        "commitment_warmup_steps": 100000,
        "commitment_full_steps": 600000,
        "commitment_reselect_initial": 1.0,
        "commitment_reselect_final": 0.25,
        "initial_termination_probability": 0.10,
        "termination_warmup_steps": 350000,
        "termination_full_steps": 800000,
        "termination_max_probability_during_ramp": 0.30,
        "termination_max_probability_final": 0.30,
        "termination_cap_full_steps": 900000,
        "termination_soft_cap_temperature": 0.03,
        "termination_loss_scale": 0.02,
        "manager_unimix_initial": 0.0,
        "manager_unimix_final": 0.01,
        "manager_unimix_decay_steps": 600000,
        "manager_pg_warmup_steps": 100000,
        "manager_pg_full_steps": 500000,
        "worker_pg_warmup_steps": 20000,
        "worker_pg_full_steps": 150000,
        "worker_scale_initial": 0.25,
        "worker_scale_max": 0.25,
        "max_usage_target": 0.95,
        "min_effective_options": 1.0,
        "source_manager_group_count": 2,
        "base_kl_target": 0.002,
        "base_kl_tail_target": 0.01,
        "base_kl_tail_fraction": 0.10,
        "base_kl_tail_relative_scale": 1.0,
        "base_kl_scale": 0.50,
        "action_preservation_confidence": 0.80,
        "action_preservation_margin": 0.05,
        "action_preservation_scale": 0.50,
        "manager_group_kl_target": 0.001,
        "manager_group_kl_tail_target": 0.005,
        "manager_group_kl_tail_fraction": 0.10,
        "manager_group_kl_tail_relative_scale": 1.0,
        "manager_group_kl_scale": 0.50,
        "manager_group_preservation_confidence": 0.80,
        "manager_group_preservation_margin": 0.05,
        "manager_group_preservation_scale": 0.50,
        "manager_collapse_scale": 0.0,
        "manager_mi_scale": 0.0,
        "action_diversity_scale": 0.0,
        "residual_cosine_scale": 0.0,
        "max_diversity_states": 2048,
        "world_model_grad_scale_initial": 0.0,
        "world_model_grad_scale_final": 0.0,
        "imag_horizon_initial_max": 10,
        "imag_horizon_final_max": 12,
        "imag_horizon_window": 4,
        "imag_horizon_ramp_steps": 600000,
    })
    cfg.hierarchical_options = OmegaConf.create(h)
    cfg.tactical_mixture.enabled = False
    cfg.sampling_mode = "shuffled_round_robin"
    cfg.adaptive_priority.enabled = False
    cfg.adaptive_priority.map.enabled = False
    cfg.adaptive_priority.sequence.enabled = False
    cfg.buffer.scratch_dir = "replay"
    cfg.validation.run_at_start = True
    cfg.validation.every = 200000
    if "compile" in cfg:
        cfg.compile = False
    if "model" in cfg and "compile" in cfg.model:
        cfg.model.compile = False
    if "wandb" in cfg:
        cfg.wandb.run_name = "tactical_v12_option_critic_v5_stability_1m"
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    target = repo / "configs/r2_2100_jepa_option_critic_2_v5_stability_1m.yaml"
    return target, cfg


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

    # Inspect /proc directly and exclude this installer plus its ancestor shell
    # chain. This avoids false positives when a validation command itself
    # contains the trainer filename in its command line.
    excluded: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    active = []
    proc = pathlib.Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) in excluded:
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                continue
            if "train_r2dreamer_smaclite_multimap.py" in command:
                active.append((entry.name, command.strip()))
    if active:
        die(
            "stop the active multimap trainer before applying the v5 stability patch: "
            + repr(active[:4])
        )

    for rel in REPLACED:
        if not (repo / rel).is_file():
            die(f"required installed file missing: {rel}")
    current_options = (repo / "external/r2dreamer/hierarchical_options.py").read_text(
        encoding="utf-8"
    )
    if V5_ARCH in current_options:
        die("v5 stability patch is already installed")
    if V4_ARCH not in current_options:
        die("installed hierarchy is not the expected v4 P1-final source")
    dreamer_text = (repo / "external/r2dreamer/dreamer.py").read_text(encoding="utf-8")
    if "# OPTION_CRITIC_P0P1_HOTFIX_V3" not in dreamer_text:
        die("dreamer.py lacks the integrated v3 gradient guard marker")
    tools_text = (repo / "external/r2dreamer/tools.py").read_text(encoding="utf-8")
    if "torch.bfloat16" not in tools_text or "x = x.float()" not in tools_text:
        die("BF16 logging fix is missing")

    for destination, source in PAYLOAD_MAP.items():
        path = PAYLOAD / source
        if not path.is_file():
            die(f"payload file missing: {source}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for destination, source in INTRODUCED_MAP.items():
        if source is None:
            continue
        path = PAYLOAD / source
        if not path.is_file():
            die(f"payload file missing: {source}")
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    target_config, config = build_config(repo)
    collisions = [
        rel for rel in INTRODUCED_MAP
        if (repo / rel).exists()
    ]
    if collisions:
        die(f"refusing to overwrite existing v5 stability files: {collisions}")

    if args.dry_run:
        print(
            "[OK] v5 stability dry-run matched integrated v4, parsed payloads, "
            "and resolved the 1M config"
        )
        return

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo.parent / f"{repo.name}_option_critic_v5_stability_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    try:
        for rel in REPLACED:
            src = repo / rel
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        for destination, source in PAYLOAD_MAP.items():
            src = PAYLOAD / source
            dst = repo / destination
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        OmegaConf.save(config, target_config)
        for destination, source in INTRODUCED_MAP.items():
            if source is None:
                continue
            src = PAYLOAD / source
            dst = repo / destination
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if dst.suffix == ".sh" or dst.suffix == ".py":
                dst.chmod(0o755)

        manifest = {
            "schema_version": 1,
            "repo": str(repo),
            "backed_up_files": list(REPLACED),
            "backed_up_sha256": {
                rel: sha256(backup / rel) for rel in REPLACED
            },
            "introduced_files": list(INTRODUCED_MAP),
        }
        (backup / "option_critic_v5_stability_backup_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        for rel in REPLACED:
            src = backup / rel
            if src.exists():
                dst = repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for rel in INTRODUCED_MAP:
            path = repo / rel
            if path.exists():
                path.unlink()
        raise

    print("[OK] installed Option-Critic v5 stability patch")
    print(f"[OK] backup: {backup}")
    print(f"[OK] config: {target_config}")
    print("[NEXT] run scripts/static_audit_option_critic_v5_stability.sh")


if __name__ == "__main__":
    main()
