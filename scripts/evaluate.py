"""
Phase 1 evaluation script for DreamerV3 × SMAClite.

Loads a trained checkpoint from a DreamerV3 logdir and runs evaluation episodes
on a fixed SMAClite scenario. No training updates occur during evaluation.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite
    python scripts\\evaluate.py --logdir logs\\smaclite_phase1\\debug_10k --scenario 2s3z --episodes 10
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


# ---------------------------------------------------------------------------
# Config / agent helpers
# ---------------------------------------------------------------------------

def load_training_config(logdir: pathlib.Path) -> elements.Config:
    """Read config.yaml saved by the training run and return an elements.Config."""
    config_path = logdir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}\n"
            "Is this a valid DreamerV3 logdir?"
        )
    raw = yaml.YAML(typ="safe").load(config_path.read_text(encoding="utf-8"))
    return elements.Config(raw)


def make_eval_env(scenario: str, max_episode_steps: int, seed: int, config: elements.Config):
    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    return wrap_env(env, config)


def build_agent(config: elements.Config, env):
    from smacdreamer.agent import SMACliteAgent

    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}

    cpdir = elements.Path(config.logdir)
    cpdir = cpdir.parent if config.replicas > 1 else cpdir
    return SMACliteAgent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=str(cpdir),
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


def load_checkpoint(agent, ckpt_dir: pathlib.Path):
    """Load the latest agent weights from a DreamerV3 checkpoint directory.

    Uses standard pathlib + pickle directly, bypassing elements.Path and its
    Checkpoint class, which can fail on Windows due to path-separator handling
    in the internal exists() check (which looks for a 'done' sentinel file).
    """
    import pickle

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Run a training job first, or check --logdir."
        )

    latest_file = ckpt_dir / "latest"
    if not latest_file.exists():
        raise FileNotFoundError(
            f"No 'latest' file found in {ckpt_dir}.\n"
            "The training run may not have saved any checkpoint yet."
        )

    folder_name = latest_file.read_text(encoding="utf-8").strip()
    ckpt_folder = ckpt_dir / folder_name
    print(f"Loading checkpoint: {ckpt_folder}")

    if not ckpt_folder.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_folder}")

    done_file = ckpt_folder / "done"
    if not done_file.exists():
        raise RuntimeError(
            f"Checkpoint at {ckpt_folder} has no 'done' marker.\n"
            "The training save was likely interrupted. Re-run training to produce a complete checkpoint."
        )

    # Determine whether agent is saved as a single file or sharded.
    agent_pkl = ckpt_folder / "agent.pkl"
    agent_shard0 = ckpt_folder / "agent-0000.pkl"

    if agent_pkl.exists():
        data = pickle.loads(agent_pkl.read_bytes())
        agent.load(data)
    elif agent_shard0.exists():
        shards = sorted(ckpt_folder.glob("agent-*.pkl"))
        def _shard_gen(shards):
            for shard in shards:
                yield pickle.loads(shard.read_bytes())
        agent.load(_shard_gen(shards))
    else:
        raise FileNotFoundError(
            f"No agent.pkl or agent-0000.pkl found in {ckpt_folder}.\n"
            f"Contents: {list(ckpt_folder.iterdir())}"
        )

    print(f"Checkpoint loaded from: {ckpt_folder}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_episode(env, agent, carry, mode: str = "eval") -> tuple:
    """Run one episode. Returns (updated carry, episode metrics dict)."""

    # Send reset=True; env resets and returns is_first=True obs with reward=0.
    reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    reset_act["reset"] = np.bool_(True)
    obs = env.step(reset_act)

    # Query policy for first action (obs has no batch dim yet).
    agent_obs = _batch_obs(obs)
    carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    # Validate action heads present on first policy call.
    if "action_0" not in acts:
        raise RuntimeError(
            f"Agent output does not contain 'action_0'. "
            f"Got keys: {list(acts.keys())}"
        )

    ep_reward = 0.0
    ep_length = 0
    ep_mask_mismatch = 0
    final_obs = obs

    while not bool(obs["is_last"]):
        # Unbatch acts, pass reset=False (mid-episode).
        act = {k: v[0] for k, v in acts.items()}
        act["reset"] = obs["is_last"]  # False here

        obs = env.step(act)
        ep_reward += float(obs["reward"])
        ep_length += 1
        ep_mask_mismatch += int(obs.get("log/step_avail_mask_mismatch_count", np.array(0)))
        final_obs = obs

        if not bool(obs["is_last"]):
            agent_obs = _batch_obs(obs)
            carry, acts, _ = agent.policy(carry, agent_obs, mode=mode)

    metrics = {
        "reward":                   ep_reward,
        "length":                   ep_length,
        "battle_won":               bool(final_obs.get("log/battle_won", np.array(False))),
        # Post-mask invalid breakdown (read from episode-end obs)
        "post_mask_invalid_count":  int(final_obs.get("log/post_mask_invalid_action_count", np.array(0))),
        "post_mask_invalid_rate":   float(final_obs.get("log/post_mask_invalid_action_rate", np.array(0.0))),
        "timing_lag_count":         int(final_obs.get("log/timing_lag_invalid_action_count", np.array(0))),
        "timing_lag_rate":          float(final_obs.get("log/timing_lag_invalid_action_rate", np.array(0.0))),
        "masking_failure_count":    int(final_obs.get("log/masking_failure_count", np.array(0))),
        "masking_failure_rate":     float(final_obs.get("log/masking_failure_rate", np.array(0.0))),
        "total_action_count":       int(final_obs.get("log/total_action_count", np.array(0))),
        "avail_mask_mismatch_slots": ep_mask_mismatch,
    }
    return carry, metrics


def _batch_obs(obs: dict) -> dict:
    """Add batch dimension (dim 0) and strip log/ keys for agent input."""
    return {k: v[None] for k, v in obs.items() if not k.startswith("log/")}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(scenario: str, episodes_data: list):
    rewards       = [e["reward"]                    for e in episodes_data]
    lengths       = [e["length"]                    for e in episodes_data]
    wins          = [e["battle_won"]                for e in episodes_data]
    post_inv_r    = [e["post_mask_invalid_rate"]    for e in episodes_data]
    lag_r         = [e["timing_lag_rate"]           for e in episodes_data]
    fail_r        = [e["masking_failure_rate"]      for e in episodes_data]
    fail_c        = [e["masking_failure_count"]     for e in episodes_data]
    total_counts  = [e["total_action_count"]        for e in episodes_data]
    mismatch      = [e["avail_mask_mismatch_slots"] for e in episodes_data]

    n = len(episodes_data)
    print()
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  scenario                      : {scenario}")
    print(f"  episodes                      : {n}")
    print(f"  mean_episode_reward           : {np.mean(rewards):.4f}")
    print(f"  std_episode_reward            : {np.std(rewards):.4f}")
    print(f"  min_episode_reward            : {np.min(rewards):.4f}")
    print(f"  max_episode_reward            : {np.max(rewards):.4f}")
    print(f"  mean_episode_length           : {np.mean(lengths):.1f}")
    print(f"  win_rate                      : {np.mean(wins):.3f}")
    print(f"  mean_total_action_count       : {np.mean(total_counts):.1f}")
    print()
    print("  -- Invalid action breakdown --")
    print(f"  post_mask_invalid_rate        : {np.mean(post_inv_r):.4f}  (after policy masking)")
    print(f"  timing_lag_rate               : {np.mean(lag_r):.4f}  (state changed mid-step)")
    print(f"  masking_failure_rate          : {np.mean(fail_r):.4f}  (should be 0.0000)")
    print(f"  masking_failure_count_total   : {sum(fail_c)}  (should be 0)")
    print(f"  mean_avail_mask_mismatch_slots: {np.mean(mismatch):.2f}  (slots that changed)")
    print()

    col_w = [7, 8, 7, 12, 14, 12, 14, 15]
    header = (
        f"{'Episode':>{col_w[0]}} | "
        f"{'Reward':>{col_w[1]}} | "
        f"{'Length':>{col_w[2]}} | "
        f"{'Battle Won':>{col_w[3]}} | "
        f"{'PostMaskRate':>{col_w[4]}} | "
        f"{'TimingLag':>{col_w[5]}} | "
        f"{'MaskFail':>{col_w[6]}} | "
        f"{'TotalActions':>{col_w[7]}}"
    )
    print(header)
    print("-" * len(header))
    for e in episodes_data:
        print(
            f"{e['episode']:>{col_w[0]}} | "
            f"{e['reward']:>{col_w[1]}.3f} | "
            f"{e['length']:>{col_w[2]}} | "
            f"{str(e['battle_won']):>{col_w[3]}} | "
            f"{e['post_mask_invalid_rate']:>{col_w[4]}.4f} | "
            f"{e['timing_lag_count']:>{col_w[5]}} | "
            f"{e['masking_failure_count']:>{col_w[6]}} | "
            f"{e['total_action_count']:>{col_w[7]}}"
        )
    print("=" * 60)
    print()


def save_results(
    output_path: pathlib.Path,
    scenario: str,
    logdir: str,
    ckpt_dir: str,
    n_episodes: int,
    seed: int,
    episodes_data: list,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rewards    = [e["reward"]                    for e in episodes_data]
    lengths    = [e["length"]                    for e in episodes_data]
    wins       = [e["battle_won"]                for e in episodes_data]
    post_inv_r = [e["post_mask_invalid_rate"]    for e in episodes_data]
    lag_r      = [e["timing_lag_rate"]           for e in episodes_data]
    fail_r     = [e["masking_failure_rate"]      for e in episodes_data]
    totals     = [e["total_action_count"]        for e in episodes_data]
    mismatch   = [e["avail_mask_mismatch_slots"] for e in episodes_data]

    result = {
        "scenario":        scenario,
        "logdir":          str(logdir),
        "checkpoint_path": str(ckpt_dir),
        "episodes":        n_episodes,
        "seed":            seed,
        "aggregate": {
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
        },
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
            for e in episodes_data
        ],
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DreamerV3 agent on a fixed SMAClite scenario."
    )
    parser.add_argument("--logdir", required=True,
                        help="Path to the DreamerV3 training log directory.")
    parser.add_argument("--scenario", default="2s3z",
                        help="SMAClite scenario name (default: 2s3z).")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the evaluation environment (default: 42).")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                        help="Episode step limit (default: 200).")
    parser.add_argument("--deterministic", type=lambda x: x.lower() != "false", default=True,
                        help="Use eval mode (deterministic policy). Default: true.")
    parser.add_argument("--output", default="results/eval_smaclite_phase1.json",
                        help="Path for the JSON results file.")
    return parser.parse_args()


def main():
    args = parse_args()

    logdir = pathlib.Path(args.logdir).resolve()
    ckpt_dir = logdir / "ckpt"
    output_path = pathlib.Path(args.output)
    mode = "eval" if args.deterministic else "train"

    # --- Validate paths ---
    if not logdir.exists():
        raise FileNotFoundError(f"--logdir does not exist: {logdir}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "Run training first, or check --logdir."
        )

    print(f"Logdir    : {logdir}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Scenario  : {args.scenario}")
    print(f"Episodes  : {args.episodes}")
    print(f"Seed      : {args.seed}")
    print(f"Mode      : {mode}")

    # --- Load training config ---
    config = load_training_config(logdir)

    # --- Portal setup (required by DreamerV3 agent internals) ---
    def _init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=False,
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[_init],
        ipv6=config.ipv6,
    )

    # --- Build env ---
    env = make_eval_env(args.scenario, args.max_episode_steps, args.seed, config)

    if "action_0" not in env.act_space:
        raise RuntimeError(
            f"act_space missing 'action_0'. Got: {list(env.act_space.keys())}"
        )
    if "state" not in env.obs_space:
        raise RuntimeError("obs_space missing 'state' key.")

    print(f"obs state shape : {env.obs_space['state'].shape}")
    print(f"act_space keys  : {[k for k in env.act_space if k != 'reset']}")

    # --- Build agent ---
    agent = build_agent(config, env)

    # --- Load checkpoint ---
    load_checkpoint(agent, ckpt_dir)

    # --- Run evaluation episodes ---
    carry = agent.init_policy(batch_size=1)
    all_episodes = []
    print(f"\nRunning {args.episodes} evaluation episode(s)...\n")

    for ep_idx in range(args.episodes):
        carry, metrics = run_episode(env, agent, carry, mode=mode)
        metrics["episode"] = ep_idx + 1
        all_episodes.append(metrics)
        status = "WIN" if metrics["battle_won"] else "loss"
        print(
            f"  Episode {ep_idx + 1:>3}/{args.episodes}: "
            f"reward={metrics['reward']:.3f}  "
            f"length={metrics['length']}  "
            f"{status}"
        )

    env.close()

    # --- Print and save ---
    print_summary(args.scenario, all_episodes)
    save_results(output_path, args.scenario, logdir, ckpt_dir, args.episodes, args.seed, all_episodes)


if __name__ == "__main__":
    main()
