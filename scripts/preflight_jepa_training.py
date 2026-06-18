#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint, sha256_file
from validate_jepa_r2_integration import run_integration_parity
from validate_jepa_token_parity import _load_checkpoint_contract, run_token_parity


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _live_metadata(meta, cfg, vis):
    live = dict(meta)
    live.setdefault("latent_dim", int(cfg.get("latent_dim", meta.get("latent_dim", 64))))
    live.setdefault("memory_dim", int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", meta.get("memory_dim", 128)))))
    live.setdefault("action_conditioned_memory", bool(cfg.get("action_conditioned_memory", False)))
    live.update(vis.metadata())
    live.setdefault("latent_normalization", cfg.get("latent_normalization", cfg.get("latent_normalize", "none")))
    return live


def run_preflight(args) -> dict:
    checkpoint = pathlib.Path(args.checkpoint)
    episode = pathlib.Path(args.episode_npz)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not episode.exists():
        raise FileNotFoundError(f"episode-npz not found: {episode}")
    meta, cfg, vis = _load_checkpoint_contract(checkpoint)
    live = _live_metadata(meta, cfg, vis)
    core, memory, info = load_frozen_jepa_checkpoint(
        checkpoint,
        map_location=torch.device(args.device),
        live_metadata=live,
    )
    frozen_count = sum(p.numel() for p in list(core.parameters()) + list(memory.parameters()))
    assert frozen_count > 0
    assert all(not p.requires_grad for p in list(core.parameters()) + list(memory.parameters()))

    token = run_token_parity(checkpoint, episode, int(args.step))
    integration = run_integration_parity(
        checkpoint,
        episode,
        device=args.device,
        rollout_horizon=int(args.rollout_horizon),
        step=int(args.step),
    )
    return {
        "result": "pass",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(pathlib.Path(args.config)),
        "episode_npz": str(episode),
        "device": args.device,
        "rollout_horizon": int(args.rollout_horizon),
        "checkpoint_metadata": meta,
        "checkpoint_config": cfg,
        "visibility": vis.metadata(),
        "frozen_parameter_count": frozen_count,
        "token_parity": {
            "max_error": token.max_error,
            "comparisons": token.comparisons,
        },
        "integration_parity": {
            "max_error": integration.max_error,
            "comparisons": integration.comparisons,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all JEPA R2-Dreamer preflight checks before smoke training.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rollout-horizon", type=int, default=10)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()
    report_path = pathlib.Path(args.report_json)
    try:
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "result": "fail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "checkpoint": args.checkpoint,
            "config": args.config,
            "episode_npz": args.episode_npz,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"JEPA R2-DREAMER PREFLIGHT: FAIL ({exc})", file=sys.stderr)
        raise SystemExit(1) from exc
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("JEPA R2-DREAMER PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
