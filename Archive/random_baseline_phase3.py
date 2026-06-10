"""
Valid-action random baseline for Phase 3 padded multi-map SMAClite.

Does not use a DreamerV3 agent. At each step, actions are selected from the padded
adapter observation using obs["avail_actions"] and obs["agent_mask"]:

  - For each real agent slot i (agent_mask[i] == 1.0):
    sample uniformly from the valid actions in avail_actions[i*MC : i*MC + n_actions]
  - For each padded agent slot i (agent_mask[i] == 0.0):
    use action 0 (noop)

Timing note: obs["avail_actions"] at step t reflects availability at t-1 (the timing
convention used by the padded adapter and by the DreamerV3 policy). This makes the
random baseline a fair comparison to the Dreamer agent.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\random_baseline_phase3.py \\
        --manifest configs\\maps\\phase3_manifest.yaml \\
        --episodes 30 --seed 42 \\
        --output results\\random_phase3_30eps.json \\
        --jsonl_output results\\random_phase3_30eps.jsonl
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

from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
from smacdreamer.envs.map_sampler import MapEntry, MapSampler, validate_manifest
from smacdreamer.envs.padding import PaddingDims


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


def _select_action(
    obs: dict,
    n_actions_real: int,
    max_actions: int,
    max_agents: int,
    rng: np.random.RandomState,
) -> dict:
    """Select one action per agent slot using obs["avail_actions"] and obs["agent_mask"].

    For real agents (agent_mask[i] == 1.0): sample uniformly from valid actions.
    For padded agents (agent_mask[i] == 0.0): action 0 (noop).
    """
    MC = max_actions
    avail_flat = np.asarray(obs["avail_actions"])  # shape (max_agents * max_actions,)

    if "agent_mask" in obs:
        agent_mask = np.asarray(obs["agent_mask"])  # shape (max_agents,)
    else:
        # pad_dims=None: all slots are real
        agent_mask = np.ones(max_agents, dtype=np.float32)

    act = {"reset": np.bool_(False)}
    for i in range(max_agents):
        if float(agent_mask[i]) < 0.5:
            act[f"action_{i}"] = np.int32(0)
        else:
            start = i * MC
            avail_slice = avail_flat[start : start + n_actions_real]
            valid = np.where(avail_slice > 0)[0]
            act[f"action_{i}"] = np.int32(rng.choice(valid) if len(valid) > 0 else 0)
    return act


def run_random_episode(
    env: SMACliteDreamerEnv,
    pad_dims: Optional[PaddingDims],
    rng: np.random.RandomState,
) -> dict:
    """Run one episode with random valid actions selected from padded adapter obs."""
    MA = pad_dims.max_agents if pad_dims else env.n_agents
    MC = pad_dims.max_actions if pad_dims else env.n_actions

    # Reset
    reset_act = {f"action_{i}": np.int32(0) for i in range(MA)}
    reset_act["reset"] = np.bool_(True)
    obs = env.step(reset_act)

    num_real_agents = int(float(obs.get("log/num_real_agents", np.array(float(env.n_agents)))))
    map_id = int(float(obs.get("log/map_id", np.array(0.0))))

    ep_reward = 0.0
    ep_length = 0
    final_obs = obs

    while not bool(obs["is_last"]):
        act = _select_action(obs, env.n_actions, MC, MA, rng)
        obs = env.step(act)
        ep_reward += float(obs["reward"])
        ep_length += 1
        final_obs = obs

    return {
        "reward":                  ep_reward,
        "length":                  ep_length,
        "num_real_agents":         num_real_agents,
        "map_id":                  map_id,
        "battle_won":              bool(final_obs.get("log/battle_won", np.array(False))),
        "masking_failure_count":   int(final_obs.get("log/masking_failure_count", np.array(0))),
        "masking_failure_rate":    float(final_obs.get("log/masking_failure_rate", np.array(0.0))),
        "total_action_count":      int(final_obs.get("log/total_action_count", np.array(0))),
        "post_mask_invalid_count": int(final_obs.get("log/post_mask_invalid_action_count", np.array(0))),
        "post_mask_invalid_rate":  float(final_obs.get("log/post_mask_invalid_action_rate", np.array(0.0))),
    }


def aggregate_metrics(episodes: list) -> dict:
    rewards = [e["reward"]               for e in episodes]
    lengths = [e["length"]               for e in episodes]
    wins    = [e["battle_won"]           for e in episodes]
    fail_r  = [e["masking_failure_rate"] for e in episodes]
    inv_r   = [e["post_mask_invalid_rate"] for e in episodes]
    n_real  = [e["num_real_agents"]      for e in episodes]
    return {
        "episodes":                    len(episodes),
        "mean_episode_reward":         float(np.mean(rewards)),
        "std_episode_reward":          float(np.std(rewards)),
        "min_episode_reward":          float(np.min(rewards)),
        "max_episode_reward":          float(np.max(rewards)),
        "mean_episode_length":         float(np.mean(lengths)),
        "win_rate":                    float(np.mean(wins)),
        "mean_masking_failure_rate":   float(np.mean(fail_r)),
        "mean_post_mask_invalid_rate": float(np.mean(inv_r)),
        "mean_num_real_agents":        float(np.mean(n_real)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valid-action random baseline for Phase 3 padded multi-map SMAClite."
    )
    parser.add_argument("--manifest", default="configs/maps/phase3_manifest.yaml",
                        help="Path to Phase 3 manifest YAML.")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per map (default: 30).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/random_phase3_30eps.json")
    parser.add_argument("--jsonl_output", default="",
                        help="Optional per-episode JSONL output path.")
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = str(ROOT / args.manifest)
    raw = validate_manifest(manifest_path)
    map_entries = [
        MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
        for e in raw['maps']
    ]
    pad_dims = _load_pad_dims_from_raw(raw)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = pathlib.Path(args.jsonl_output) if args.jsonl_output else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    print(f"\nValid-action random baseline — Phase 3")
    print(f"Manifest : {args.manifest}")
    print(f"Padding  : {pad_dims}")
    print(f"Episodes : {args.episodes} per map")
    print(f"Seed     : {args.seed}\n")

    per_map_results = {}
    all_episodes_combined = []

    jsonl_file = open(jsonl_path, 'w', encoding='utf-8') if jsonl_path else None
    try:
        for entry in map_entries:
            print(f"\n--- Map: {entry.name} ---")
            sampler = MapSampler([entry], mode='fixed')
            env = SMACliteDreamerEnv(
                scenario=entry.name,
                max_episode_steps=200,
                seed=args.seed,
                map_sampler=sampler,
                pad_dims=pad_dims,
            )

            map_episodes = []
            for ep_idx in range(args.episodes):
                metrics = run_random_episode(env, pad_dims, rng)
                metrics["episode"] = ep_idx + 1
                map_episodes.append(metrics)

                status = "WIN" if metrics["battle_won"] else "loss"
                print(
                    f"  Episode {ep_idx+1:>3}/{args.episodes}: "
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
                "aggregate":    agg,
                "episodes_data": [
                    {
                        "episode":                 e["episode"],
                        "reward":                  float(e["reward"]),
                        "length":                  int(e["length"]),
                        "battle_won":              bool(e["battle_won"]),
                        "num_real_agents":         int(e["num_real_agents"]),
                        "map_id":                  int(e["map_id"]),
                        "total_action_count":      int(e["total_action_count"]),
                        "post_mask_invalid_count": int(e["post_mask_invalid_count"]),
                        "post_mask_invalid_rate":  float(e["post_mask_invalid_rate"]),
                        "masking_failure_count":   int(e["masking_failure_count"]),
                        "masking_failure_rate":    float(e["masking_failure_rate"]),
                    }
                    for e in map_episodes
                ],
            }
            all_episodes_combined.extend(map_episodes)

            print(
                f"  {entry.name}: win_rate={agg['win_rate']:.3f}"
                f"  mean_reward={agg['mean_episode_reward']:.3f}"
                f"  n_real={agg['mean_num_real_agents']:.0f}"
            )
    finally:
        if jsonl_file:
            jsonl_file.close()

    overall_agg = aggregate_metrics(all_episodes_combined)
    print(f"\n{'='*60}")
    print("Valid-action random baseline summary")
    print(f"{'='*60}")
    for name, res in per_map_results.items():
        a = res["aggregate"]
        print(
            f"  {name:<22} win_rate={a['win_rate']:.3f}"
            f"  mean_reward={a['mean_episode_reward']:.3f}"
            f"  n_real={a['mean_num_real_agents']:.0f}"
        )
    print(
        f"  {'OVERALL':<22} win_rate={overall_agg['win_rate']:.3f}"
        f"  mean_reward={overall_agg['mean_episode_reward']:.3f}"
    )
    print(f"{'='*60}\n")

    result = {
        "label":          "valid-action random baseline",
        "manifest":       args.manifest,
        "episodes_per_map": args.episodes,
        "seed":           args.seed,
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
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Results saved to : {output_path}")
    if args.jsonl_output:
        print(f"Per-episode JSONL: {args.jsonl_output}")


if __name__ == "__main__":
    main()
