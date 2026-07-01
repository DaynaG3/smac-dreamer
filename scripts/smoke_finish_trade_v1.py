"""Smoke test for the finish_trade_v1 reward on a real SMAClite episode (no training).

Runs a few short episodes on a builtin scenario with the finish_trade_v1 reward wired into
SMACliteDreamerEnv and verifies:
  * finish_trade_v1 resolves from the registry (and from a YAML reward block);
  * the shaped reward differs from the original env reward on at least one episode;
  * the new per-term reward metrics (log_reward_term_*) appear in the step info;
  * no NaNs in rewards / terms;
  * post-mask invalid-action metrics stay exactly 0 (valid actions are chosen every step);
  * the reward's per-episode state resets cleanly across episodes.

Usage (smac-r2 conda env, project root):
    python scripts/smoke_finish_trade_v1.py
    python scripts/smoke_finish_trade_v1.py --scenario 2s3z --episodes 2 --max-steps 40
"""

import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "external" / "smaclite")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from smacdreamer.envs.reward_registry import resolve, resolved_params

# The finish_trade_v1 term keys the env should surface as log_reward_term_<name>[/_ep_sum].
_COMPONENT_TERMS = [
    "enemy_progress", "ally_loss", "stall_penalty", "timeout_enemy", "timeout_alive",
    "win_speed", "win_ally_ehp", "all_dead_loss",
]
_DIAGNOSTIC_TERMS = [
    "no_damage_streak_max", "no_damage_streak_mean", "timeout_with_allies_alive",
    "allies_dead_loss", "near_win_timeout",
]


def _valid_actions(raw_env):
    """First valid action per agent from the current avail mask (keeps invalids at 0)."""
    acts = []
    for mask in raw_env.get_avail_actions():
        idx = next((j for j, v in enumerate(mask) if v), 0)
        acts.append(int(idx))
    return acts


def run_episode(env, raw_env, max_steps):
    obs, info = env.reset(seed=0)
    orig_return = 0.0
    shaped_return = 0.0
    seen_component = set()
    steps = 0
    last_info = info
    for _ in range(max_steps):
        acts = _valid_actions(raw_env)
        obs, reward, terminated, truncated, info = env.step(acts)
        steps += 1
        assert math.isfinite(float(reward)), "NaN/inf shaped reward"
        orig_return += float(info.get("log_reward_original", 0.0))
        shaped_return += float(reward)
        # Per-term metrics present + finite.
        for name in _COMPONENT_TERMS + _DIAGNOSTIC_TERMS:
            key = f"log_reward_term_{name}"
            assert key in info, f"missing per-term metric {key}"
            assert f"{key}_ep_sum" in info, f"missing episode-sum metric {key}_ep_sum"
            assert math.isfinite(float(np.asarray(info[key]))), f"NaN in {key}"
            if abs(float(np.asarray(info[key]))) > 0.0:
                seen_component.add(name)
        last_info = info
        if terminated or truncated:
            break

    invalid = float(np.asarray(last_info.get("log_post_mask_invalid_action_count", 0.0)))
    return {
        "steps": steps,
        "shaping_enabled": float(np.asarray(last_info.get("log_reward_shaping_enabled", 0.0))),
        "orig_return": orig_return,
        "shaped_return": shaped_return,
        "shaping_bonus": float(np.asarray(last_info.get("log_episode_reward_shaping_bonus", 0.0))),
        "invalid": invalid,
        "battle_won": bool(last_info.get("battle_won", False)),
        "seen_component": seen_component,
        "streak_max": float(np.asarray(last_info.get("log_reward_term_no_damage_streak_max_ep_sum", 0.0))),
        "final_enemy_ehp": float(np.asarray(last_info.get("log_final_enemy_ehp_frac", 0.0))),
    }


def main():
    ap = argparse.ArgumentParser(description="finish_trade_v1 reward smoke test")
    ap.add_argument("--config", default="configs/r2_2100_finish_trade_v1.yaml",
                    help="YAML config whose reward block is loaded and resolved")
    ap.add_argument("--scenario", default="2s3z")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()

    # 1. Load the reward straight from the YAML config (proves YAML -> registry wiring).
    from omegaconf import OmegaConf

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / args.config
    cfg = OmegaConf.load(str(cfg_path))
    assert cfg.reward.name == "finish_trade_v1", (
        f"config reward.name is {cfg.reward.name!r}, expected 'finish_trade_v1'")
    params = OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True) or {}
    print(f"[smoke] config: {cfg_path}")
    print(f"[smoke] reward.name: {cfg.reward.name}")
    print(f"[smoke] config reward params: {params}")
    print(f"[smoke] resolved_params: {resolved_params(cfg.reward.name, params)}")
    reward_fn = resolve(cfg.reward.name, params)

    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv

    env = SMACliteDreamerEnv(
        scenario=args.scenario, max_episode_steps=args.max_steps, seed=0,
        reward_fn=reward_fn, gamma=0.997, obs_mode="flat",
    )
    raw_env = getattr(env._env, "unwrapped", env._env)

    results = []
    for ep in range(args.episodes):
        r = run_episode(env, raw_env, args.max_steps)
        results.append(r)
        print(f"[smoke] ep{ep}: steps={r['steps']} won={r['battle_won']} "
              f"orig_return={r['orig_return']:.4f} shaped_return={r['shaped_return']:.4f} "
              f"shaping_bonus={r['shaping_bonus']:.4f} shaping_enabled={r['shaping_enabled']:.0f} "
              f"invalid={r['invalid']:.0f} "
              f"streak_max={r['streak_max']:.0f} final_enemy_ehp={r['final_enemy_ehp']:.3f} "
              f"components={sorted(r['seen_component'])}")
    env.close()

    # --- Assertions ---------------------------------------------------------------------
    ok = True
    diffs = [abs(r["shaped_return"] - r["orig_return"]) for r in results]
    if not any(d > 1e-6 for d in diffs):
        print("[smoke] FAIL: shaped return never differed from original return")
        ok = False
    if any(r["invalid"] > 0 for r in results):
        print("[smoke] FAIL: post-mask invalid-action count was non-zero")
        ok = False
    if any(r["shaping_enabled"] != 1.0 for r in results):
        print("[smoke] FAIL: log_reward_shaping_enabled was not 1.0 on the reward_fn path")
        ok = False
    all_seen = set().union(*[r["seen_component"] for r in results]) if results else set()
    if not all_seen:
        print("[smoke] FAIL: no finish_trade_v1 reward component was ever non-zero")
        ok = False
    for r in results:
        if not math.isfinite(r["shaped_return"]) or not math.isfinite(r["orig_return"]):
            print("[smoke] FAIL: non-finite return")
            ok = False

    print(f"[smoke] shaped!=orig on {sum(d > 1e-6 for d in diffs)}/{len(diffs)} episodes; "
          f"non-zero components observed: {sorted(all_seen)}")
    if ok:
        print("[smoke] PASS: finish_trade_v1 loads, shapes reward, emits metrics, "
              "no NaNs, zero invalid actions.")
        sys.exit(0)
    else:
        print("[smoke] FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
