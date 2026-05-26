"""
Phase 3 per-map evaluation for DreamerV3 × SMAClite (padded multi-map).

Loads a trained checkpoint and evaluates N episodes on each map in the manifest.
Reports per-map and aggregate metrics including Phase 3 padding metrics.
Per-episode results can optionally be written to a JSONL file.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\evaluate_phase3.py \\
        --manifest configs\\maps\\phase3_manifest.yaml \\
        --logdir   logs\\smaclite_phase3\\debug_5k \\
        --episodes 30 --seed 42 \\
        --output   results\\eval_phase3_debug_5k_30eps.json \\
        --jsonl_output results\\eval_phase3_debug_5k_30eps.jsonl
"""

import argparse
import json
import pathlib
import sys
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import ruamel.yaml as yaml
import elements
import portal

from dreamerv3.main import wrap_env
from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
from smacdreamer.envs.map_sampler import MapSampler, MapEntry, validate_manifest
from smacdreamer.envs.padding import PaddingDims

from evaluate import (
    load_training_config,
    build_agent,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _batch_obs(obs: dict) -> dict:
    """Add batch dimension (dim 0) and strip log/ keys for agent input."""
    return {k: v[None] for k, v in obs.items() if not k.startswith("log/")}


def _run_episode_p3(env, agent, carry, mode: str = "eval") -> tuple:
    """Run one episode; capture Phase 3 metrics from obs.

    Reads log/num_real_agents and log/map_id directly from the reset obs dict
    (not via env attribute access), so it works identically with wrapped envs.
    Returns (carry, metrics_dict).
    """
    reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    reset_act["reset"] = np.bool_(True)
    obs = env.step(reset_act)

    num_real_agents = int(float(obs.get("log/num_real_agents", np.array(0.0))))
    map_id = int(float(obs.get("log/map_id", np.array(0.0))))

    agent_obs = _batch_obs(obs)
    carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    ep_reward = 0.0
    ep_length = 0
    ep_mask_mismatch = 0
    final_obs = obs

    while not bool(obs["is_last"]):
        act = {k: v[0] for k, v in acts.items()}
        act["reset"] = obs["is_last"]
        obs = env.step(act)
        ep_reward += float(obs["reward"])
        ep_length += 1
        ep_mask_mismatch += int(obs.get("log/step_avail_mask_mismatch_count", np.array(0)))
        final_obs = obs
        if not bool(obs["is_last"]):
            agent_obs = _batch_obs(obs)
            carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    metrics = {
        "reward":                    ep_reward,
        "length":                    ep_length,
        "num_real_agents":           num_real_agents,
        "map_id":                    map_id,
        "battle_won":                bool(final_obs.get("log/battle_won", np.array(False))),
        "post_mask_invalid_count":   int(final_obs.get("log/post_mask_invalid_action_count", np.array(0))),
        "post_mask_invalid_rate":    float(final_obs.get("log/post_mask_invalid_action_rate", np.array(0.0))),
        "timing_lag_count":          int(final_obs.get("log/timing_lag_invalid_action_count", np.array(0))),
        "timing_lag_rate":           float(final_obs.get("log/timing_lag_invalid_action_rate", np.array(0.0))),
        "masking_failure_count":     int(final_obs.get("log/masking_failure_count", np.array(0))),
        "masking_failure_rate":      float(final_obs.get("log/masking_failure_rate", np.array(0.0))),
        "total_action_count":        int(final_obs.get("log/total_action_count", np.array(0))),
        "avail_mask_mismatch_slots": ep_mask_mismatch,
    }
    return carry, metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pad_dims_from_raw(raw: dict) -> Optional[PaddingDims]:
    if 'padding' not in raw:
        return None
    p = raw['padding']
    return PaddingDims(
        max_agents=p['max_agents'],
        max_enemies=p['max_enemies'],
        max_actions=p['max_actions'],
        max_obs_size=p['max_obs_size'],
    )


def make_eval_env_for_map(
    entry: MapEntry,
    pad_dims: Optional[PaddingDims],
    max_episode_steps: int,
    seed: int,
    config: elements.Config,
):
    """Create a fixed-scenario eval env for a single map entry, with optional padding."""
    if entry.type == 'builtin':
        env = SMACliteDreamerEnv(
            scenario=entry.name,
            max_episode_steps=max_episode_steps,
            seed=seed,
            pad_dims=pad_dims,
        )
    else:
        sampler = MapSampler([entry], mode='fixed')
        env = SMACliteDreamerEnv(
            scenario=entry.name,
            max_episode_steps=max_episode_steps,
            seed=seed,
            map_sampler=sampler,
            pad_dims=pad_dims,
        )
    return wrap_env(env, config)


def aggregate_metrics(all_episodes: list) -> dict:
    rewards         = [e["reward"]                    for e in all_episodes]
    lengths         = [e["length"]                    for e in all_episodes]
    wins            = [e["battle_won"]                for e in all_episodes]
    post_inv_r      = [e["post_mask_invalid_rate"]    for e in all_episodes]
    lag_r           = [e["timing_lag_rate"]           for e in all_episodes]
    fail_r          = [e["masking_failure_rate"]      for e in all_episodes]
    totals          = [e["total_action_count"]        for e in all_episodes]
    mismatch        = [e["avail_mask_mismatch_slots"] for e in all_episodes]
    num_real_agents = [e["num_real_agents"]           for e in all_episodes]
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
        "mean_num_real_agents":            float(np.mean(num_real_agents)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-map evaluation for Phase 3 DreamerV3 × SMAClite."
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to the Phase 3 map manifest YAML.")
    parser.add_argument("--logdir", required=True,
                        help="Path to a completed DreamerV3 training log directory.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes per map (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Environment random seed (default: 42).")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Episode step limit (default: 200).")
    parser.add_argument("--deterministic",
                        type=lambda x: x.lower() != "false", default=True,
                        help="Use eval mode (deterministic policy). Default: true.")
    parser.add_argument("--no_padding", action="store_true",
                        help="Disable Phase 3 padding (use each map's real dims).")
    parser.add_argument("--output", default="results/eval_phase3.json",
                        help="JSON output path.")
    parser.add_argument("--jsonl_output", default="",
                        help="Optional per-episode JSONL output path.")
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

    checkpoint_path = str(ckpt_dir)
    checkpoint_loaded = False

    print(f"\nPhase 3 per-map evaluation")
    print(f"Manifest        : {args.manifest}")
    print(f"Logdir          : {logdir}")
    print(f"Checkpoint path : {checkpoint_path}")
    print(f"Episodes        : {args.episodes} per map")
    print(f"Seed            : {args.seed}")
    print(f"Mode            : {mode}")

    raw = validate_manifest(manifest_path)
    map_entries = [
        MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
        for e in raw['maps']
    ]
    pad_dims = None if args.no_padding else _load_pad_dims_from_raw(raw)
    print(f"Padding         : {'disabled' if args.no_padding else str(pad_dims)}")
    print(f"Maps            : {[e.name for e in map_entries]}\n")

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

    first_env = make_eval_env_for_map(
        map_entries[0], pad_dims, args.max_episode_steps, args.seed, config)
    agent = build_agent(config, first_env)

    try:
        load_checkpoint(agent, ckpt_dir)
        checkpoint_loaded = True
        print(f"Checkpoint loaded : True\n")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARNING: Checkpoint load failed — {exc}")
        print("Evaluation will run with uninitialised weights.")
        print("Results with checkpoint_loaded=false are NOT valid for learning comparison.\n")

    first_env.close()

    jsonl_path = pathlib.Path(args.jsonl_output) if args.jsonl_output else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    per_map_results = {}
    all_episodes_combined = []

    jsonl_file = open(jsonl_path, 'w', encoding='utf-8') if jsonl_path else None
    try:
        for entry in map_entries:
            print(f"\n--- Map: {entry.name} ---")
            env = make_eval_env_for_map(
                entry, pad_dims, args.max_episode_steps, args.seed, config)
            carry = agent.init_policy(batch_size=1)

            map_episodes = []
            for ep_idx in range(args.episodes):
                carry, metrics = _run_episode_p3(env, agent, carry, mode=mode)
                metrics["episode"] = ep_idx + 1
                map_episodes.append(metrics)

                status = "WIN" if metrics["battle_won"] else "loss"
                print(
                    f"  Episode {ep_idx + 1:>3}/{args.episodes}: "
                    f"reward={metrics['reward']:.3f}  "
                    f"length={metrics['length']}  "
                    f"n_real={metrics['num_real_agents']}  "
                    f"{status}"
                )

                if jsonl_file is not None:
                    line = {
                        "map":                    entry.name,
                        "map_id":                 metrics["map_id"],
                        "episode":                ep_idx + 1,
                        "reward":                 float(metrics["reward"]),
                        "length":                 int(metrics["length"]),
                        "battle_won":             bool(metrics["battle_won"]),
                        "post_mask_invalid_rate": float(metrics["post_mask_invalid_rate"]),
                        "masking_failure_rate":   float(metrics["masking_failure_rate"]),
                        "num_real_agents":        int(metrics["num_real_agents"]),
                    }
                    jsonl_file.write(json.dumps(line) + '\n')
                    jsonl_file.flush()

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
                        "num_real_agents":           int(e["num_real_agents"]),
                        "map_id":                    int(e["map_id"]),
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
                f"  {entry.name}: win_rate={agg['win_rate']:.3f}"
                f"  mean_reward={agg['mean_episode_reward']:.3f}"
                f"  n_real={agg['mean_num_real_agents']:.0f}"
                f"  post_mask={agg['mean_post_mask_invalid_rate']:.4f}"
                f"  mask_fail={agg['mean_masking_failure_rate']:.4f}"
            )
    finally:
        if jsonl_file:
            jsonl_file.close()

    overall_agg = aggregate_metrics(all_episodes_combined)
    print(f"\n{'='*60}")
    print("Phase 3 Evaluation Summary")
    print(f"{'='*60}")
    for name, res in per_map_results.items():
        a = res["aggregate"]
        print(
            f"  {name:<22} win_rate={a['win_rate']:.3f}"
            f"  mean_reward={a['mean_episode_reward']:.3f}"
            f"  n_real={a['mean_num_real_agents']:.0f}"
            f"  mask_fail={a['mean_masking_failure_rate']:.4f}"
        )
    print(
        f"  {'OVERALL':<22} win_rate={overall_agg['win_rate']:.3f}"
        f"  mean_reward={overall_agg['mean_episode_reward']:.3f}"
    )
    print(f"{'='*60}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "manifest":          args.manifest,
        "logdir":            str(logdir),
        "checkpoint_path":   checkpoint_path,
        "checkpoint_loaded": checkpoint_loaded,
        "episodes_per_map":  args.episodes,
        "seed":              args.seed,
        "padding_dims": (
            None if pad_dims is None else {
                "max_agents":   pad_dims.max_agents,
                "max_enemies":  pad_dims.max_enemies,
                "max_actions":  pad_dims.max_actions,
                "max_obs_size": pad_dims.max_obs_size,
            }
        ),
        "maps":      per_map_results,
        "aggregate": overall_agg,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved to      : {output_path}")
    if args.jsonl_output:
        print(f"Per-episode JSONL     : {args.jsonl_output}")


if __name__ == "__main__":
    main()
