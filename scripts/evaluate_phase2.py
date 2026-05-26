"""
Phase 2 per-map evaluation for DreamerV3 × SMAClite.

Loads a trained checkpoint and evaluates N episodes on each map in the manifest.
Reports per-map and aggregate metrics.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\evaluate_phase2.py \\
        --manifest configs\\maps\\phase2_manifest.yaml \\
        --logdir   logs\\smaclite_phase2\\debug_5k \\
        --episodes 5 \\
        --output   results\\eval_phase2.json
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import ruamel.yaml as yaml
import elements
import embodied
import portal

from dreamerv3.main import wrap_env
from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
from smacdreamer.envs.map_sampler import MapSampler, MapEntry, validate_manifest

# Re-use helpers from evaluate.py.
from evaluate import (
    load_training_config,
    build_agent,
    load_checkpoint,
    run_episode,
)


def make_eval_env_for_map(
    entry: MapEntry,
    max_episode_steps: int,
    seed: int,
    config: elements.Config,
):
    """Create a fixed-scenario eval env for a single map entry."""
    if entry.type == 'builtin':
        env = SMACliteDreamerEnv(
            scenario=entry.name,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )
    else:
        from smacdreamer.envs.map_sampler import MapSampler
        sampler = MapSampler([entry], mode='fixed')
        env = SMACliteDreamerEnv(
            scenario=entry.name,
            max_episode_steps=max_episode_steps,
            seed=seed,
            map_sampler=sampler,
        )
    return wrap_env(env, config)


def aggregate_metrics(all_episodes: list) -> dict:
    rewards    = [e["reward"]                    for e in all_episodes]
    lengths    = [e["length"]                    for e in all_episodes]
    wins       = [e["battle_won"]                for e in all_episodes]
    post_inv_r = [e["post_mask_invalid_rate"]    for e in all_episodes]
    lag_r      = [e["timing_lag_rate"]           for e in all_episodes]
    fail_r     = [e["masking_failure_rate"]      for e in all_episodes]
    totals     = [e["total_action_count"]        for e in all_episodes]
    mismatch   = [e["avail_mask_mismatch_slots"] for e in all_episodes]
    return {
        "episodes":                        len(all_episodes),
        "mean_episode_reward":             float(np.mean(rewards)),
        "std_episode_reward":              float(np.std(rewards)),
        "min_episode_reward":              float(np.min(rewards)),
        "max_episode_reward":              float(np.max(rewards)),
        "mean_episode_length":             float(np.mean(lengths)),
        "win_rate":                        float(np.mean(wins)),
        "mean_total_action_count":         float(np.mean(totals)),
        "mean_post_mask_invalid_rate":     float(np.mean(post_inv_r)),
        "mean_timing_lag_rate":            float(np.mean(lag_r)),
        "mean_masking_failure_rate":       float(np.mean(fail_r)),
        "mean_avail_mask_mismatch_slots":  float(np.mean(mismatch)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-map evaluation for Phase 2 DreamerV3 × SMAClite."
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to the Phase 2 map manifest YAML.")
    parser.add_argument("--logdir", required=True,
                        help="Path to a completed DreamerV3 training log directory.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes per map (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Environment random seed (default: 42).")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Episode step limit (default: 200).")
    parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True,
                        help="Use eval mode (deterministic policy). Default: true.")
    parser.add_argument("--output", default="results/eval_phase2.json",
                        help="JSON output path (default: results/eval_phase2.json).")
    return parser.parse_args()


def main():
    args = parse_args()

    logdir = pathlib.Path(args.logdir).resolve()
    ckpt_dir = logdir / "ckpt"
    output_path = pathlib.Path(args.output)
    mode = "eval" if args.deterministic else "train"
    manifest_path = str(pathlib.Path(args.manifest).resolve())

    if not logdir.exists():
        raise FileNotFoundError(f"--logdir does not exist: {logdir}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Run training first, or check --logdir."
        )

    print(f"\nPhase 2 per-map evaluation")
    print(f"Manifest  : {args.manifest}")
    print(f"Logdir    : {logdir}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Episodes  : {args.episodes} per map")
    print(f"Seed      : {args.seed}")
    print(f"Mode      : {mode}")

    # Validate and load manifest.
    raw = validate_manifest(manifest_path)
    map_entries = [
        MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
        for e in raw['maps']
    ]
    print(f"Maps      : {[e.name for e in map_entries]}\n")

    config = load_training_config(logdir)

    def _init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=False,
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[_init],
        ipv6=config.ipv6,
    )

    # Build agent once using the first map's env (all maps share the same shape).
    first_env = make_eval_env_for_map(
        map_entries[0], args.max_episode_steps, args.seed, config)
    agent = build_agent(config, first_env)
    load_checkpoint(agent, ckpt_dir)
    first_env.close()

    # Evaluate per map.
    per_map_results = {}
    all_episodes_combined = []

    for entry in map_entries:
        print(f"\n--- Map: {entry.name} ---")
        env = make_eval_env_for_map(entry, args.max_episode_steps, args.seed, config)
        carry = agent.init_policy(batch_size=1)

        map_episodes = []
        for ep_idx in range(args.episodes):
            carry, metrics = run_episode(env, agent, carry, mode=mode)
            metrics["episode"] = ep_idx + 1
            map_episodes.append(metrics)
            status = "WIN" if metrics["battle_won"] else "loss"
            print(
                f"  Episode {ep_idx + 1:>3}/{args.episodes}: "
                f"reward={metrics['reward']:.3f}  "
                f"length={metrics['length']}  "
                f"{status}"
            )

        env.close()
        agg = aggregate_metrics(map_episodes)
        per_map_results[entry.name] = {
            "aggregate": agg,
            "episodes_data": [
                {
                    "episode":                   e["episode"],
                    "reward":                    float(e["reward"]),
                    "length":                    int(e["length"]),
                    "battle_won":                bool(e["battle_won"]),
                    "total_action_count":        int(e["total_action_count"]),
                    "post_mask_invalid_count":   int(e["post_mask_invalid_count"]),
                    "post_mask_invalid_rate":    float(e["post_mask_invalid_rate"]),
                    "timing_lag_count":          int(e["timing_lag_count"]),
                    "timing_lag_rate":           float(e["timing_lag_rate"]),
                    "masking_failure_count":     int(e["masking_failure_count"]),
                    "masking_failure_rate":      float(e["masking_failure_rate"]),
                    "avail_mask_mismatch_slots": int(e["avail_mask_mismatch_slots"]),
                }
                for e in map_episodes
            ],
        }
        all_episodes_combined.extend(map_episodes)

        print(
            f"  {entry.name}: win_rate={agg['win_rate']:.3f}  "
            f"mean_reward={agg['mean_episode_reward']:.3f}  "
            f"post_mask_rate={agg['mean_post_mask_invalid_rate']:.4f}  "
            f"timing_lag_rate={agg['mean_timing_lag_rate']:.4f}  "
            f"masking_failure_rate={agg['mean_masking_failure_rate']:.4f}"
        )

    # Print combined summary.
    overall_agg = aggregate_metrics(all_episodes_combined)
    print(f"\n{'='*60}")
    print("Phase 2 Evaluation Summary")
    print(f"{'='*60}")
    for name, res in per_map_results.items():
        a = res["aggregate"]
        print(
            f"  {name:<20} win_rate={a['win_rate']:.3f}"
            f"  mean_reward={a['mean_episode_reward']:.3f}"
            f"  post_mask_rate={a['mean_post_mask_invalid_rate']:.4f}"
            f"  timing_lag={a['mean_timing_lag_rate']:.4f}"
            f"  mask_fail={a['mean_masking_failure_rate']:.4f}"
        )
    print(
        f"  {'OVERALL':<20} win_rate={overall_agg['win_rate']:.3f}"
        f"  mean_reward={overall_agg['mean_episode_reward']:.3f}"
        f"  post_mask_rate={overall_agg['mean_post_mask_invalid_rate']:.4f}"
        f"  timing_lag={overall_agg['mean_timing_lag_rate']:.4f}"
        f"  mask_fail={overall_agg['mean_masking_failure_rate']:.4f}"
    )
    print(f"{'='*60}\n")

    # Save results.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "manifest":        args.manifest,
        "logdir":          str(logdir),
        "checkpoint_path": str(ckpt_dir),
        "episodes_per_map": args.episodes,
        "seed":            args.seed,
        "maps":            per_map_results,
        "aggregate":       overall_agg,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
