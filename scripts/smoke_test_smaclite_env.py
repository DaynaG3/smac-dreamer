"""
Standalone smoke test for SMACliteDreamerEnv.
No DreamerV3 training required — only tests the adapter interface.

Usage (PowerShell):
    $env:PYTHONPATH = "$PWD\src;$PWD\external\dreamerv3;$PWD\external\smaclite"
    python scripts\smoke_test_smaclite_env.py --scenario 2s3z
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv


def make_reset_action(env: SMACliteDreamerEnv) -> dict:
    act = {f"action_{i}": np.int32(0) for i in range(env.n_agents)}
    act["reset"] = np.bool_(True)
    return act


def make_random_valid_action(env: SMACliteDreamerEnv) -> dict:
    avail = env._env.unwrapped.get_avail_actions()
    act = {"reset": np.bool_(False)}
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        act[f"action_{i}"] = np.int32(np.random.choice(valid))
    return act


def make_invalid_action(env: SMACliteDreamerEnv) -> dict:
    """Deliberately invalid: action index far outside valid range."""
    act = {"reset": np.bool_(False)}
    for i in range(env.n_agents):
        act[f"action_{i}"] = np.int32(env.n_actions + 999)
    return act


def assert_obs_shapes(obs: dict, env: SMACliteDreamerEnv, label: str):
    expected = env.obs_space
    for key, space in expected.items():
        assert key in obs, f"[{label}] Missing obs key: '{key}'"
        val = obs[key]
        assert isinstance(val, np.ndarray), f"[{label}] '{key}' is not ndarray: {type(val)}"
        assert val.shape == space.shape, (
            f"[{label}] '{key}' shape {val.shape} != expected {space.shape}"
        )
        assert val.dtype == space.dtype or np.can_cast(val.dtype, space.dtype, casting="same_kind"), (
            f"[{label}] '{key}' dtype {val.dtype} incompatible with {space.dtype}"
        )


def run_smoke_test(scenario: str):
    print(f"\n{'='*60}")
    print(f"SMACliteDreamerEnv smoke test — scenario: {scenario}")
    print(f"{'='*60}\n")

    env = SMACliteDreamerEnv(scenario=scenario, max_episode_steps=200, seed=0)

    print(f"  scenario        : {scenario}")
    print(f"  n_agents        : {env.n_agents}")
    print(f"  n_enemies       : {env.n_enemies}")
    print(f"  n_actions       : {env.n_actions}")
    print(f"  obs_size        : {env.obs_size}")
    print(f"  state shape     : {(env.n_agents * env.obs_size,)}")
    print(f"  avail_actions   : {(env.n_agents * env.n_actions,)}")
    print(f"  act_space keys  : {list(env.act_space.keys())}")
    print()

    # ---- Test 1: reset -----------------------------------------------
    obs = env.step(make_reset_action(env))
    assert obs["is_first"] == True, "Expected is_first=True after reset"
    assert obs["is_last"] == False
    assert_obs_shapes(obs, env, "reset")
    print("[PASS] Reset returns valid first-step obs.")

    # ---- Test 2: invalid actions -------------------------------------
    invalid_obs = env.step(make_invalid_action(env))
    assert_obs_shapes(invalid_obs, env, "invalid_action_step")
    print("[PASS] Invalid actions do not crash; obs shapes stable.")

    # Run a few more steps so we can reach episode end and check log/ metrics.
    for _ in range(5):
        obs = env.step(make_random_valid_action(env))
        assert_obs_shapes(obs, env, "random_step")
        if obs["is_last"]:
            break

    # Force end of this episode by resetting.
    obs = env.step(make_reset_action(env))

    # ---- Test 3: full episode with valid actions ----------------------
    obs = env.step(make_reset_action(env))
    episode_return = 0.0
    episode_length = 0
    final_obs = obs
    while not final_obs["is_last"]:
        final_obs = env.step(make_random_valid_action(env))
        assert_obs_shapes(final_obs, env, "episode_step")
        episode_return += float(final_obs["reward"])
        episode_length += 1

    battle_won = bool(final_obs["log/battle_won"])
    invalid_count = int(final_obs["log/episode_invalid_action_count"])
    total_count = int(final_obs["log/episode_total_action_count"])
    invalid_rate = float(final_obs["log/episode_invalid_action_rate"])

    print(f"\n  episode_return  : {episode_return:.4f}")
    print(f"  episode_length  : {episode_length}")
    print(f"  battle_won      : {battle_won}")
    print(f"  invalid_count   : {invalid_count}")
    print(f"  total_count     : {total_count}")
    print(f"  invalid_rate    : {invalid_rate:.4f}")
    print()

    assert isinstance(episode_return, float), "reward must be numeric"
    assert episode_length > 0, "episode must have at least one step"
    assert total_count == env.n_agents * episode_length, (
        f"total_count {total_count} != n_agents*length {env.n_agents}*{episode_length}"
    )
    assert 0.0 <= invalid_rate <= 1.0
    print("[PASS] Full episode completed with correct metrics.")

    # ---- Test 4: second episode (sequential reset) ------------------
    obs2 = env.step(make_reset_action(env))
    assert obs2["is_first"] == True, "Second episode must start with is_first=True"
    assert_obs_shapes(obs2, env, "second_episode_reset")
    step2 = env.step(make_random_valid_action(env))
    assert_obs_shapes(step2, env, "second_episode_step")
    print("[PASS] Sequential reset and second episode work correctly.")

    # ---- Test 5: log/ keys are 0-d scalars ---------------------------
    for key in ["log/battle_won", "log/episode_invalid_action_count",
                "log/episode_total_action_count", "log/episode_invalid_action_rate"]:
        val = final_obs[key]
        assert val.ndim == 0, f"'{key}' must be 0-d scalar, got ndim={val.ndim}"
        assert val.dtype == np.float32, f"'{key}' must be float32, got {val.dtype}"
    print("[PASS] All log/ metrics are 0-d float32 scalars.")

    env.close()
    print(f"\n{'='*60}")
    print("All smoke tests PASSED.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="2s3z")
    args = parser.parse_args()
    run_smoke_test(args.scenario)
