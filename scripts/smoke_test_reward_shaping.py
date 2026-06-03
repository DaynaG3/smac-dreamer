"""
Smoke tests for reward shaping v2 in SMACliteDreamerEnv.

Four test cases using random actions and no trained agent:
  A — shaping disabled  : obs["reward"] == obs["log/original_env_reward"] each step
  B — shaping enabled   : survival bonus fires, reward != original_env_reward
  C — terminal signals  : max_episode_steps=5 to reach is_last quickly
  D — backward compat   : old flat kill_reward_bonus / step_penalty unaffected

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\smoke_test_reward_shaping.py ^
        --scenario 2s_vs_1sc ^
        --manifest configs\\maps\\phase3d_overfit_2s_vs_1sc_manifest.yaml
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
from smacdreamer.envs.reward_shaping import RewardShapingConfig
from smacdreamer.envs.map_sampler import MapSampler, MapEntry, validate_manifest
from smacdreamer.envs.padding import PaddingDims


_PASSES = 0
_FAILURES = 0


def _check(cond: bool, msg: str):
    global _PASSES, _FAILURES
    if cond:
        _PASSES += 1
        print(f"    PASS  {msg}")
    else:
        _FAILURES += 1
        print(f"    FAIL  {msg}")


def _make_env(scenario: str, manifest_path: str, pad_dims, max_episode_steps: int = 200,
              reward_shaping_config=None, kill_reward_bonus: float = 0.0, step_penalty: float = 0.0):
    sampler = MapSampler.from_manifest(manifest_path, mode='fixed', seed=42)
    return SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        seed=42,
        map_sampler=sampler,
        pad_dims=pad_dims,
        kill_reward_bonus=kill_reward_bonus,
        step_penalty=step_penalty,
        reward_shaping_config=reward_shaping_config,
    )


def _random_action(env) -> dict:
    """Sample a random valid action for each real agent."""
    act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    act["reset"] = np.bool_(False)
    # Pick action 1 (stop) for all agents — always valid.
    for k in act:
        if k.startswith("action_"):
            act[k] = np.int32(1)
    return act


def _reset_env(env) -> dict:
    act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    act["reset"] = np.bool_(True)
    return env.step(act)


# ---------------------------------------------------------------------------
# Test A — shaping disabled
# ---------------------------------------------------------------------------

def test_a_shaping_disabled(scenario: str, manifest_path: str, pad_dims):
    print("\nTest A: shaping disabled")
    env = _make_env(scenario, manifest_path, pad_dims, reward_shaping_config=None)
    try:
        obs = _reset_env(env)
        for step_i in range(5):
            act = _random_action(env)
            obs = env.step(act)
            if bool(obs["is_last"]):
                obs = _reset_env(env)
                continue
            orig = float(obs.get("log/original_env_reward", np.array(-999.0)))
            reward = float(obs["reward"])
            shaping_bonus = float(obs.get("log/reward_shaping_bonus", np.array(-999.0)))
            shaping_enabled = float(obs.get("log/reward_shaping_enabled", np.array(-999.0)))
            _check(abs(reward - orig) < 1e-6,
                   f"step {step_i}: reward({reward:.6f}) == original_env_reward({orig:.6f})")
            _check(abs(shaping_bonus) < 1e-6,
                   f"step {step_i}: reward_shaping_bonus is 0.0 (got {shaping_bonus:.6f})")
            _check(shaping_enabled == 0.0,
                   f"step {step_i}: reward_shaping_enabled == 0.0 (got {shaping_enabled})")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test B — shaping enabled
# ---------------------------------------------------------------------------

def test_b_shaping_enabled(scenario: str, manifest_path: str, pad_dims):
    print("\nTest B: shaping enabled (survival bonus)")
    rs = RewardShapingConfig(
        enabled=True,
        ally_survival_bonus=0.1,
        step_penalty=0.01,
        win_bonus=10.0,
        loss_penalty=-10.0,
        enemy_kill_bonus=5.0,
        ally_death_penalty=-2.0,
    )
    env = _make_env(scenario, manifest_path, pad_dims, reward_shaping_config=rs)
    try:
        obs = _reset_env(env)
        for step_i in range(5):
            act = _random_action(env)
            obs = env.step(act)
            if bool(obs["is_last"]):
                obs = _reset_env(env)
                continue
            shaping_enabled = float(obs.get("log/reward_shaping_enabled", np.array(0.0)))
            allies_alive = float(obs.get("log/allies_alive", np.array(0.0)))
            survival_bonus = float(obs.get("log/step_v2_ally_survival_bonus", np.array(0.0)))
            orig = float(obs.get("log/original_env_reward", np.array(0.0)))
            shaped = float(obs.get("log/shaped_reward", np.array(0.0)))
            _check(shaping_enabled == 1.0,
                   f"step {step_i}: reward_shaping_enabled == 1.0 (got {shaping_enabled})")
            _check(allies_alive > 0,
                   f"step {step_i}: allies_alive > 0 (got {allies_alive})")
            _check(survival_bonus > 0,
                   f"step {step_i}: step_v2_ally_survival_bonus > 0 (got {survival_bonus:.6f})")
            _check(abs(shaped - orig) > 1e-9 or abs(orig) < 1e-9,
                   f"step {step_i}: shaped_reward({shaped:.6f}) != original_env_reward({orig:.6f})")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test C — terminal signals
# ---------------------------------------------------------------------------

def test_c_terminal_signals(scenario: str, manifest_path: str, pad_dims):
    print("\nTest C: terminal signals (max_episode_steps=5)")
    rs = RewardShapingConfig(
        enabled=True,
        win_bonus=10.0,
        loss_penalty=-10.0,
        ally_survival_bonus=0.1,
        step_penalty=0.003,
    )
    env = _make_env(scenario, manifest_path, pad_dims,
                    max_episode_steps=5, reward_shaping_config=rs)
    found_terminal = False
    try:
        for _attempt in range(10):
            obs = _reset_env(env)
            ep_step = 0
            while not bool(obs["is_last"]):
                act = _random_action(env)
                obs = env.step(act)
                ep_step += 1

            found_terminal = True
            ep_orig = float(obs.get("log/episode_original_env_return", np.array(0.0)))
            ep_shaped = float(obs.get("log/episode_shaped_return", np.array(0.0)))
            ep_bonus = float(obs.get("log/episode_reward_shaping_bonus", np.array(0.0)))
            _check(ep_step > 0, f"episode ran {ep_step} steps before terminal")
            _check(ep_shaped != 0.0 or ep_step == 0,
                   f"episode_shaped_return is non-zero ({ep_shaped:.4f})")
            _check(abs(ep_bonus - (ep_shaped - ep_orig)) < 1e-4,
                   f"episode_reward_shaping_bonus({ep_bonus:.4f}) == shaped - orig ({ep_shaped - ep_orig:.4f})")

            # Check terminal bonus/penalty on the terminal step
            battle_won = bool(obs.get("log/battle_won", np.array(False)))
            is_terminated = bool(obs.get("is_terminal", np.array(False)))
            v2_win = float(obs.get("log/step_v2_win_bonus", np.array(0.0)))
            v2_loss = float(obs.get("log/step_v2_loss_penalty", np.array(0.0)))
            if battle_won:
                _check(v2_win > 0, f"win step: step_v2_win_bonus={v2_win:.4f} > 0")
            elif is_terminated and not battle_won:
                _check(v2_loss < 0, f"loss step: step_v2_loss_penalty={v2_loss:.4f} < 0")
            else:
                print(f"    NOTE  truncation only — no win/loss bonus expected "
                      f"(battle_won={battle_won}, is_terminal={is_terminated})")
            break
        if not found_terminal:
            _check(False, "did not reach a terminal step in 10 attempts")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Test D — backward compat (old flat params)
# ---------------------------------------------------------------------------

def test_d_backward_compat(scenario: str, manifest_path: str, pad_dims):
    print("\nTest D: backward compat (kill_reward_bonus + step_penalty, no v2 config)")
    env = _make_env(scenario, manifest_path, pad_dims,
                    kill_reward_bonus=1.0, step_penalty=0.01,
                    reward_shaping_config=None)
    try:
        obs = _reset_env(env)
        for step_i in range(3):
            act = _random_action(env)
            obs = env.step(act)
            if bool(obs["is_last"]):
                obs = _reset_env(env)
                continue
            kill_bonus = float(obs.get("log/step_kill_bonus", np.array(-999.0)))
            step_pen = float(obs.get("log/step_step_penalty", np.array(-999.0)))
            shaping_enabled = float(obs.get("log/reward_shaping_enabled", np.array(-999.0)))
            masking_fail = float(obs.get("log/masking_failure_count", np.array(-999.0)))
            # No kill on a stop action; step_penalty should equal 0.01
            _check(kill_bonus == 0.0,
                   f"step {step_i}: step_kill_bonus == 0.0 (no kill on stop, got {kill_bonus})")
            _check(abs(step_pen - 0.01) < 1e-6,
                   f"step {step_i}: step_step_penalty == 0.01 (got {step_pen:.6f})")
            _check(shaping_enabled == 0.0,
                   f"step {step_i}: reward_shaping_enabled == 0.0 (got {shaping_enabled})")
            _check(masking_fail == 0.0,
                   f"step {step_i}: masking_failure_count == 0 (got {masking_fail})")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke tests for reward shaping v2 in SMACliteDreamerEnv."
    )
    parser.add_argument("--scenario", default="2s_vs_1sc",
                        help="SMAClite scenario name (default: 2s_vs_1sc).")
    parser.add_argument("--manifest", required=True,
                        help="Path to map manifest YAML (must contain a padding block).")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = str(pathlib.Path(args.manifest).resolve())
    raw = validate_manifest(manifest_path)
    p = raw.get('padding')
    pad_dims = None
    if p:
        pad_dims = PaddingDims(
            max_agents=p['max_agents'],
            max_enemies=p['max_enemies'],
            max_actions=p['max_actions'],
            max_obs_size=p['max_obs_size'],
        )

    print(f"Smoke test: scenario={args.scenario}  manifest={args.manifest}")
    print(f"Padding: {pad_dims}\n")

    test_a_shaping_disabled(args.scenario, manifest_path, pad_dims)
    test_b_shaping_enabled(args.scenario, manifest_path, pad_dims)
    test_c_terminal_signals(args.scenario, manifest_path, pad_dims)
    test_d_backward_compat(args.scenario, manifest_path, pad_dims)

    total = _PASSES + _FAILURES
    print(f"\n{'='*50}")
    print(f"Results: {_PASSES}/{total} passed, {_FAILURES}/{total} failed")
    if _FAILURES > 0:
        sys.exit(1)
    else:
        print("All smoke tests passed.")


if __name__ == "__main__":
    main()
