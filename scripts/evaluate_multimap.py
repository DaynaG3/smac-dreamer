"""Held-out evaluation for the multimap R2-Dreamer × SMAClite pipeline.

Loads a checkpoint and evaluates deterministically on the HELD-OUT TEST maps only,
reporting per-map and per-family win rate + ORIGINAL (unshaped) return, with:
  * per-map Wilson confidence intervals on win rate (with n_episodes), and
  * an ACROSS-MAP confidence interval as the HEADLINE (each map's win rate is one sample),
    because between-map variance dominates with a small test set (~10 maps).

Asserts no TRAIN map is evaluated. Writes a JSON report.

Eval reward is the ORIGINAL SMAClite reward (smaclite_default) regardless of the training
reward, so the metric is comparable to baselines. Action selection is deterministic
(actor mode); the env's action sanitiser is the final safety net (policy-side eval masking
is a separate, later stage).

Usage (smac-r2 conda env, project root):
    python scripts\\evaluate_multimap.py --config configs\\multimap.yaml --checkpoint logs\\r2dreamer\\multimap\\latest.pt
    python scripts\\evaluate_multimap.py --config configs\\multimap.yaml --checkpoint latest.pt --episodes-per-map 16
"""

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from omegaconf import OmegaConf

import tools
from dreamer import Dreamer
from smacdreamer.envs.map_discovery import discover, SplitSpec
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env
from train_r2dreamer_smaclite_debug import make_config as _make_debug_config


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean_ci(values: list, z: float = 1.96) -> tuple:
    """Mean +/- normal CI across samples (each map's win rate is one sample)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (mean, max(0.0, mean - z * se), min(1.0, mean + z * se))


@torch.no_grad()
def _run_map(agent, env, n_episodes, device):
    """Run n_episodes on a single-env pool; return (wins, returns_original list)."""
    wins = 0
    returns_original = []
    for _ in range(n_episodes):
        obs = env.reset()
        state = agent.get_initial_state(1)
        act = state["prev_action"].clone()
        done = False
        ep_orig = 0.0
        won = False
        while not done:
            from tensordict import TensorDict
            td = TensorDict(
                {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()},
                batch_size=(1,),
            )
            td["action"] = act
            act, state = agent.act(td, state, eval=True)
            a = act.detach().cpu().numpy().reshape(-1)
            obs, reward, done, info = env.step(a)
            ep_orig += float(info.get("log_reward_original", reward))
            won = bool(info.get("battle_won", False))
        wins += int(won)
        returns_original.append(ep_orig)
    return wins, returns_original


def main():
    ap = argparse.ArgumentParser(description="Held-out multimap evaluation")
    ap.add_argument("--config", default="configs/multimap.yaml")
    ap.add_argument("--checkpoint", required=True, help="path to latest.pt")
    ap.add_argument("--episodes-per-map", type=int, default=None)
    ap.add_argument("--output", default=None, help="JSON report path")
    args = ap.parse_args()

    cfg_path = (ROOT / args.config) if not pathlib.Path(args.config).is_absolute() else pathlib.Path(args.config)
    cfg = OmegaConf.load(str(cfg_path))
    device = str(cfg.device)
    n_eps = int(args.episodes_per_map if args.episodes_per_map is not None else cfg.eval.episodes_per_map)

    # Re-run discovery deterministically (same split seed) to get the SAME held-out test set.
    train_entries, test_entries, pad_dims = discover(
        str(cfg.maps_folder),
        SplitSpec(**OmegaConf.to_container(cfg.split, resolve=True)),
        padding_override=OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None,
        verbose=True,
    )
    if not test_entries:
        sys.exit("No held-out test maps to evaluate.")
    train_names = {e.name for e in train_entries}

    # Build the agent with the SAME obs/action shape the model was trained with: construct a
    # one-map env to read the spaces, then load the checkpoint.
    probe = make_smaclite_multimap_env(
        [test_entries[0]], pad_dims, "fixed", 0, 0, "smaclite_default", {},
        float(cfg.gamma), int(cfg.max_episode_steps),
    )
    obs_space, act_space = probe.observation_space, probe.action_space

    config = _make_debug_config(argparse.Namespace(
        steps=1, batch_size=int(cfg.batch_size), batch_length=int(cfg.batch_length),
        units=int(cfg.units), deter=int(cfg.deter), imag_horizon=int(cfg.imag_horizon),
    ))
    config.device = device
    config.model.device = device
    config.model.rssm.device = device

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Evaluate each held-out test map --------------------------------------
    per_map = {}
    per_family_winrates = defaultdict(list)
    print(f"\nEvaluating {len(test_entries)} held-out maps × {n_eps} episodes ...")
    for entry in test_entries:
        assert entry.name not in train_names, f"LEAK: train map '{entry.name}' in eval set!"
        env = make_smaclite_multimap_env(
            [entry], pad_dims, "fixed", 0, 0, "smaclite_default", {},
            float(cfg.gamma), int(cfg.max_episode_steps),
        )
        wins, returns = _run_map(agent, env, n_eps, device)
        try:
            env.close()
        except Exception:
            pass
        wr = wins / n_eps if n_eps else 0.0
        lo, hi = wilson_interval(wins, n_eps)
        per_map[entry.name] = {
            "family": entry.family, "n_episodes": n_eps, "wins": wins,
            "win_rate": wr, "win_rate_ci95": [lo, hi],
            "mean_original_return": (sum(returns) / len(returns)) if returns else 0.0,
        }
        per_family_winrates[entry.family].append(wr)
        print(f"  {entry.name:<32} win_rate={wr:.2f} (95% CI [{lo:.2f},{hi:.2f}]) "
              f"orig_return={per_map[entry.name]['mean_original_return']:.3f}")

    # ---- Headline: ACROSS-MAP CI (each map = one sample) ----------------------
    map_winrates = [m["win_rate"] for m in per_map.values()]
    headline_mean, headline_lo, headline_hi = _mean_ci(map_winrates)

    per_family = {
        fam: {"n_maps": len(wrs), "mean_win_rate": sum(wrs) / len(wrs)}
        for fam, wrs in per_family_winrates.items()
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "maps_folder": str(cfg.maps_folder),
        "split": OmegaConf.to_container(cfg.split, resolve=True),
        "n_test_maps": len(test_entries),
        "episodes_per_map": n_eps,
        "headline_held_out_win_rate": {
            "mean_across_maps": headline_mean,
            "ci95_across_maps": [headline_lo, headline_hi],
            "n_maps": len(map_winrates),
            "note": "Across-map CI (each map = one sample) is the headline; between-map "
                    "variance dominates with a small test set.",
        },
        "per_family": per_family,
        "per_map": per_map,
    }

    out = pathlib.Path(args.output) if args.output else (
        ROOT / "results" / f"multimap_eval_{pathlib.Path(str(cfg.maps_folder)).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'='*64}")
    print(f"HEADLINE held-out win rate (across {len(map_winrates)} maps): "
          f"{headline_mean:.3f}  95% CI [{headline_lo:.3f}, {headline_hi:.3f}]")
    print(f"Report written: {out}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
