"""
Phase 4 per-map evaluation for DreamerV3 × SMAClite (folder-driven multi-map).

Loads a trained checkpoint and evaluates N episodes per map on the requested split
(train / validation / test). Reports per-map, per-family, and aggregate metrics.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\evaluate_phase4.py ^
        --manifest configs\\maps\\phase4_manifest.yaml ^
        --split validation ^
        --logdir logs\\smaclite_phase4\\size1m_1m ^
        --episodes_per_map 10 --seed 42 ^
        --output results\\phase4_validation.json ^
        --jsonl_output results\\phase4_validation.jsonl

    python scripts\\evaluate_phase4.py ^
        --manifest configs\\maps\\phase4_manifest.yaml ^
        --split test ^
        --logdir logs\\smaclite_phase4\\size1m_1m ^
        --episodes_per_map 20 --seed 42 ^
        --output results\\phase4_test.json ^
        --jsonl_output results\\phase4_test.jsonl
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
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
from smacdreamer.envs.map_sampler import MapSampler, MapEntry
from smacdreamer.envs.padding import PaddingDims

from evaluate import (
    load_training_config,
    build_agent,
    load_checkpoint,
)
from smacdreamer.envs.reward_shaping import from_dict as _rs_from_dict


# ---------------------------------------------------------------------------
# Episode runner (reused from Phase 3 logic)
# ---------------------------------------------------------------------------

def _batch_obs(obs: dict) -> dict:
    return {k: v[None] for k, v in obs.items() if not k.startswith("log/")}


def _run_episode(env, agent, carry, mode: str = "eval", ep_idx: int = 0) -> tuple:
    reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    reset_act["reset"] = np.bool_(True)
    obs = env.step(reset_act)

    num_real_agents = int(float(obs.get("log/num_real_agents", np.array(0.0))))
    map_id = int(float(obs.get("log/map_id", np.array(0.0))))

    action_keys = sorted(
        [k for k in env.act_space if k.startswith("action_")],
        key=lambda k: int(k.split("_")[1]),
    )

    agent_obs = _batch_obs(obs)
    carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    ep_reward_original = 0.0
    ep_length = 0
    final_obs = obs

    while not bool(obs["is_last"]):
        act = {k: v[0] for k, v in acts.items()}
        act["reset"] = obs["is_last"]
        obs = env.step(act)
        ep_reward_original += float(obs.get("log/original_env_reward", obs["reward"]))
        ep_length += 1
        final_obs = obs

        if not bool(obs["is_last"]):
            agent_obs = _batch_obs(obs)
            carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    metrics = {
        "reward":                  ep_reward_original,
        "length":                  ep_length,
        "num_real_agents":         num_real_agents,
        "map_id":                  map_id,
        "battle_won":              bool(final_obs.get("log/battle_won", np.array(False))),
        "post_mask_invalid_rate":  float(final_obs.get("log/post_mask_invalid_action_rate", np.array(0.0))),
        "timing_lag_rate":         float(final_obs.get("log/timing_lag_invalid_action_rate", np.array(0.0))),
        "masking_failure_rate":    float(final_obs.get("log/masking_failure_rate", np.array(0.0))),
        "total_action_count":      int(final_obs.get("log/total_action_count", np.array(0))),
    }
    return carry, metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(episodes: list) -> dict:
    if not episodes:
        return {}
    rewards   = [e["reward"]               for e in episodes]
    lengths   = [e["length"]               for e in episodes]
    wins      = [e["battle_won"]           for e in episodes]
    post_inv  = [e["post_mask_invalid_rate"] for e in episodes]
    lag_r     = [e["timing_lag_rate"]       for e in episodes]
    fail_r    = [e["masking_failure_rate"]  for e in episodes]
    n_real    = [e["num_real_agents"]       for e in episodes]
    totals    = [e["total_action_count"]    for e in episodes]
    return {
        "episodes":                      len(episodes),
        "mean_episode_reward":           float(np.mean(rewards)),
        "std_episode_reward":            float(np.std(rewards)),
        "min_episode_reward":            float(np.min(rewards)),
        "max_episode_reward":            float(np.max(rewards)),
        "mean_episode_length":           float(np.mean(lengths)),
        "win_rate":                      float(np.mean(wins)),
        "mean_post_mask_invalid_rate":   float(np.mean(post_inv)),
        "mean_timing_lag_rate":          float(np.mean(lag_r)),
        "mean_masking_failure_rate":     float(np.mean(fail_r)),
        "mean_num_real_agents":          float(np.mean(n_real)),
        "mean_total_action_count":       float(np.mean(totals)),
    }


def aggregate_family_metrics(per_map: dict, entry_family_map: dict) -> dict:
    by_family = defaultdict(list)
    for map_name, res in per_map.items():
        fam = entry_family_map.get(map_name, "uncategorised")
        by_family[fam].append(res["aggregate"])
    family_agg = {}
    for fam, agg_list in sorted(by_family.items()):
        all_eps = []
        for agg in agg_list:
            # Reconstruct pseudo-episode list from agg for consistent aggregation
            # Use mean_episode_reward and win_rate as best proxies
            family_agg[fam] = {
                "maps":                  len(agg_list),
                "mean_episode_reward":   float(np.mean([a["mean_episode_reward"] for a in agg_list])),
                "std_episode_reward":    float(np.std([a["mean_episode_reward"] for a in agg_list])),
                "win_rate":              float(np.mean([a["win_rate"] for a in agg_list])),
                "mean_episode_length":   float(np.mean([a["mean_episode_length"] for a in agg_list])),
                "mean_masking_failure":  float(np.mean([a["mean_masking_failure_rate"] for a in agg_list])),
                "mean_num_real_agents":  float(np.mean([a["mean_num_real_agents"] for a in agg_list])),
            }
    return family_agg


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_eval_env(entry_dict: dict, pad_dims: Optional[PaddingDims],
                  max_episode_steps: int, seed: int, config,
                  reward_shaping_config=None):
    e = MapEntry(
        name=entry_dict['name'],
        type=entry_dict.get('type', 'custom'),
        path=entry_dict.get('path'),
        family=entry_dict.get('family', 'uncategorised'),
        map_id=entry_dict.get('map_id', 0),
    )
    sampler = MapSampler([e], mode='fixed')
    env = SMACliteDreamerEnv(
        scenario=e.name,
        max_episode_steps=max_episode_steps,
        seed=seed,
        map_sampler=sampler,
        pad_dims=pad_dims,
        reward_shaping_config=reward_shaping_config,
    )
    return wrap_env(env, config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 per-map evaluation for DreamerV3 × SMAClite."
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to Phase 4 manifest YAML (version: 1).")
    parser.add_argument("--split", required=True,
                        choices=["train", "validation", "test"],
                        help="Which manifest split to evaluate.")
    parser.add_argument("--logdir", required=True,
                        help="DreamerV3 training log directory with checkpoint.")
    parser.add_argument("--episodes_per_map", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--deterministic",
                        type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--output", default="results/phase4_eval.json")
    parser.add_argument("--jsonl_output", default="")
    parser.add_argument("--max_maps", type=int, default=None,
                        help="Evaluate only the first N maps (for debugging).")
    parser.add_argument("--wandb_project", default="",
                        help="If set, upload evaluation summary as W&B artifact.")
    return parser.parse_args()


def main():
    args = parse_args()

    logdir      = pathlib.Path(args.logdir).resolve()
    ckpt_dir    = logdir / "ckpt"
    output_path = pathlib.Path(args.output)
    mode        = "eval" if args.deterministic else "train"
    manifest_p  = pathlib.Path(args.manifest).resolve()

    if not logdir.exists():
        raise FileNotFoundError(f"--logdir does not exist: {logdir}")

    raw = yaml.YAML(typ='safe').load(manifest_p.read_text(encoding='utf-8'))
    if raw.get('version') != 1:
        raise ValueError(
            f"Expected a Phase 4 manifest (version: 1), got version={raw.get('version')!r}."
        )

    split_entries = raw.get('splits', {}).get(args.split, [])
    if args.max_maps is not None:
        split_entries = split_entries[:args.max_maps]

    if not split_entries:
        raise ValueError(
            f"Phase 4 manifest split='{args.split}' is empty or missing."
        )

    pad_raw  = raw.get('padding', {})
    pad_dims = PaddingDims(
        max_agents=pad_raw['max_agents'],
        max_enemies=pad_raw['max_enemies'],
        max_actions=pad_raw['max_actions'],
        max_obs_size=pad_raw['max_obs_size'],
    )

    entry_family_map = {e['name']: e.get('family', 'uncategorised') for e in split_entries}

    print(f"\nPhase 4 Evaluation")
    print(f"  Manifest     : {args.manifest}")
    print(f"  Split        : {args.split} ({len(split_entries)} maps)")
    print(f"  Logdir       : {logdir}")
    print(f"  Episodes/map : {args.episodes_per_map}")
    print(f"  Seed         : {args.seed}")
    print(f"  Mode         : {mode}")
    print(f"  Padding      : agents={pad_dims.max_agents} enemies={pad_dims.max_enemies} "
          f"actions={pad_dims.max_actions} obs={pad_dims.max_obs_size}")
    families = sorted(set(entry_family_map.values()))
    print(f"  Families     : {families}\n")

    config = load_training_config(logdir)

    eval_reward_shaping_config = None
    try:
        sc = config.env.get("smaclite", {})
        eval_reward_shaping_config = _rs_from_dict(sc.get("reward_shaping", {}))
    except Exception:
        pass

    def _init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=False,
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[_init],
        ipv6=config.ipv6,
    )

    first_env = make_eval_env(
        split_entries[0], pad_dims, args.max_episode_steps, args.seed, config,
        reward_shaping_config=eval_reward_shaping_config)
    agent = build_agent(config, first_env)

    checkpoint_loaded = False
    try:
        load_checkpoint(agent, ckpt_dir)
        checkpoint_loaded = True
        print(f"Checkpoint loaded: True\n")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARNING: Checkpoint load failed — {exc}")
        print("Results with checkpoint_loaded=false are NOT valid for learning comparison.\n")

    first_env.close()

    jsonl_path = pathlib.Path(args.jsonl_output) if args.jsonl_output else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    per_map_results = {}
    all_episodes    = []

    jsonl_file = open(jsonl_path, 'w', encoding='utf-8') if jsonl_path else None
    try:
        for entry_dict in split_entries:
            map_name = entry_dict['name']
            family   = entry_dict.get('family', 'uncategorised')
            print(f"\n--- {map_name} (family={family}, "
                  f"agents={entry_dict.get('n_agents')}, enemies={entry_dict.get('n_enemies')}) ---")

            env = make_eval_env(
                entry_dict, pad_dims, args.max_episode_steps, args.seed, config,
                reward_shaping_config=eval_reward_shaping_config)
            carry = agent.init_policy(batch_size=1)

            map_eps = []
            for ep_idx in range(args.episodes_per_map):
                carry, metrics = _run_episode(env, agent, carry, mode=mode, ep_idx=ep_idx+1)
                metrics["episode"] = ep_idx + 1
                metrics["map_name"] = map_name
                metrics["family"]   = family
                map_eps.append(metrics)

                status = "WIN" if metrics["battle_won"] else "loss"
                print(
                    f"  ep {ep_idx+1:>3}/{args.episodes_per_map}: "
                    f"reward={metrics['reward']:.3f}  length={metrics['length']}  "
                    f"agents={metrics['num_real_agents']}  {status}"
                )

                if jsonl_file is not None:
                    jsonl_file.write(json.dumps({
                        "split":                  args.split,
                        "map":                    map_name,
                        "map_id":                 int(metrics["map_id"]),
                        "family":                 family,
                        "episode":                ep_idx + 1,
                        "reward":                 float(metrics["reward"]),
                        "length":                 int(metrics["length"]),
                        "battle_won":             bool(metrics["battle_won"]),
                        "post_mask_invalid_rate": float(metrics["post_mask_invalid_rate"]),
                        "masking_failure_rate":   float(metrics["masking_failure_rate"]),
                        "num_real_agents":        int(metrics["num_real_agents"]),
                    }) + '\n')
                    jsonl_file.flush()

            env.close()
            agg = aggregate_metrics(map_eps)
            per_map_results[map_name] = {
                "family":    family,
                "map_id":    entry_dict.get('map_id', 0),
                "n_agents":  entry_dict.get('n_agents'),
                "n_enemies": entry_dict.get('n_enemies'),
                "aggregate": agg,
            }
            all_episodes.extend(map_eps)

            print(
                f"  {map_name}: win={agg['win_rate']:.3f}  "
                f"reward={agg['mean_episode_reward']:.3f}  "
                f"mask_fail={agg['mean_masking_failure_rate']:.4f}"
            )
    finally:
        if jsonl_file:
            jsonl_file.close()

    overall_agg    = aggregate_metrics(all_episodes)
    per_family_agg = aggregate_family_metrics(per_map_results, entry_family_map)

    print(f"\n{'='*70}")
    print(f"Phase 4 Evaluation — {args.split} split")
    print(f"{'='*70}")
    print(f"{'Family':<25}  {'Maps':>5}  {'Reward':>8}  {'Win%':>6}  {'Agents':>6}")
    print(f"{'-'*70}")
    for fam, fa in sorted(per_family_agg.items()):
        print(f"  {fam:<23}  {fa['maps']:>5}  {fa['mean_episode_reward']:>8.3f}  "
              f"{fa['win_rate']:>6.3f}  {fa['mean_num_real_agents']:>6.1f}")
    print(f"{'-'*70}")
    print(
        f"  {'OVERALL':<23}  {len(per_map_results):>5}  "
        f"{overall_agg['mean_episode_reward']:>8.3f}  "
        f"{overall_agg['win_rate']:>6.3f}  "
        f"{overall_agg['mean_num_real_agents']:>6.1f}"
    )
    print(f"{'='*70}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "manifest":          str(args.manifest),
        "split":             args.split,
        "logdir":            str(logdir),
        "checkpoint_path":   str(ckpt_dir),
        "checkpoint_loaded": checkpoint_loaded,
        "episodes_per_map":  args.episodes_per_map,
        "seed":              args.seed,
        "dataset_name":      raw.get("dataset_name", ""),
        "dataset_hash":      raw.get("dataset_hash", ""),
        "padding_dims": {
            "max_agents":   pad_dims.max_agents,
            "max_enemies":  pad_dims.max_enemies,
            "max_actions":  pad_dims.max_actions,
            "max_obs_size": pad_dims.max_obs_size,
        },
        "per_map":           per_map_results,
        "per_family":        per_family_agg,
        "aggregate":         overall_agg,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved : {output_path}")
    if args.jsonl_output:
        print(f"JSONL saved   : {args.jsonl_output}")

    # --- W&B artifact upload ---
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, job_type="eval",
                             config={"split": args.split, "manifest": str(args.manifest)})
            artifact = wandb.Artifact(
                name=f"phase4_eval_{args.split}",
                type="evaluation",
                description=f"Phase 4 evaluation results for split={args.split}",
            )
            artifact.add_file(str(output_path))
            if args.jsonl_output:
                artifact.add_file(args.jsonl_output)
            run.log_artifact(artifact)
            run.log({
                f"eval/{args.split}/win_rate":   overall_agg["win_rate"],
                f"eval/{args.split}/mean_reward": overall_agg["mean_episode_reward"],
            })
            run.finish()
            print(f"W&B artifact uploaded to project '{args.wandb_project}'")
        except Exception as e:
            print(f"WARNING: W&B upload failed — {e}")


if __name__ == "__main__":
    main()
