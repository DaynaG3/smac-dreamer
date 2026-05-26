"""
Phase 2 smoke test: one episode per map, shape validation, sampler ordering.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite
    python scripts\\smoke_test_phase2.py --manifest configs\\maps\\phase2_manifest.yaml
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
from smacdreamer.envs.map_sampler import MapSampler, MapEntry, validate_manifest


def make_reset_action(n_agents: int) -> dict:
    act = {f"action_{i}": np.int32(0) for i in range(n_agents)}
    act["reset"] = np.bool_(True)
    return act


def make_random_valid_action(env: SMACliteDreamerEnv) -> dict:
    uw = getattr(env._env, 'unwrapped', env._env)
    avail = uw.get_avail_actions()
    act = {"reset": np.bool_(False)}
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        act[f"action_{i}"] = np.int32(np.random.choice(valid))
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


def run_one_episode(env: SMACliteDreamerEnv, label: str):
    """Run one full episode, assert shapes on every step. Return episode metrics."""
    obs = env.step(make_reset_action(env.n_agents))
    assert obs["is_first"], f"[{label}] Expected is_first=True after reset"
    assert_obs_shapes(obs, env, f"{label}/reset")

    ep_return = 0.0
    ep_length = 0
    last_obs = obs
    while not last_obs["is_last"]:
        last_obs = env.step(make_random_valid_action(env))
        assert_obs_shapes(last_obs, env, f"{label}/step")
        ep_return += float(last_obs["reward"])
        ep_length += 1

    return {
        "reward":    ep_return,
        "length":    ep_length,
        "battle_won": bool(last_obs["log/battle_won"]),
        "map_id":    float(last_obs["log/map_id"]),
        "map_name":  env._current_map_name,
    }


def test_shape_rejection(manifest_path: str):
    """Verify that a map with a different shape raises ValueError on reset."""
    print("\n[shape rejection test]")
    from smaclite.env.smaclite import SMACliteEnv
    from smaclite.env.maps.map import MapInfo

    # Build a sampler whose second map has incompatible shape (use 3s5z: n_agents=8).
    # We fake this by using a MapEntry pointing to the 3s5z built-in, which has n_agents=8.
    bad_entry = MapEntry(name='3s5z', type='builtin')
    good_entry = MapEntry(name='2s3z', type='builtin')
    sampler = MapSampler([good_entry, bad_entry], mode='round_robin')
    env = SMACliteDreamerEnv(scenario='2s3z', map_sampler=sampler)

    # First reset: uses good map (round-robin idx 0 -> 1).
    env.step(make_reset_action(env.n_agents))

    # Second reset: should try bad map and raise ValueError.
    raised = False
    try:
        env.step(make_reset_action(env.n_agents))
    except ValueError as e:
        raised = True
        print(f"  Correctly raised ValueError: {e}")
    env.close()
    assert raised, "Expected ValueError for incompatible map shape, but none was raised!"
    print("[PASS] Shape mismatch correctly rejected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "configs" / "maps" / "phase2_manifest.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = str(pathlib.Path(args.manifest).resolve())

    print(f"\n{'='*60}")
    print(f"Phase 2 smoke test")
    print(f"Manifest: {args.manifest}")
    print(f"{'='*60}\n")

    # ---- Validate manifest -----------------------------------------------
    print("[manifest validation]")
    raw = validate_manifest(manifest_path)
    maps = raw['maps']
    print(f"  maps found : {[m['name'] for m in maps]}")
    print("[PASS] Manifest valid.\n")

    # ---- Round-robin: one episode per map --------------------------------
    sampler = MapSampler.from_manifest(manifest_path, mode='round_robin', seed=args.seed)
    n_maps = len(sampler.maps)
    env = SMACliteDreamerEnv(scenario='2s3z', max_episode_steps=200, seed=args.seed,
                             map_sampler=sampler)

    print(f"  n_agents   : {env.n_agents}")
    print(f"  n_enemies  : {env.n_enemies}")
    print(f"  n_actions  : {env.n_actions}")
    print(f"  obs_size   : {env.obs_size}")
    print(f"  state shape: {(env.n_agents * env.obs_size,)}")
    print(f"  avail shape: {(env.n_agents * env.n_actions,)}")
    print()

    base_obs_shape = (env.n_agents * env.obs_size,)
    base_avail_shape = (env.n_agents * env.n_actions,)

    results = []
    for episode_idx in range(n_maps):
        label = f"episode_{episode_idx + 1}"
        res = run_one_episode(env, label)
        results.append(res)
        print(
            f"  {label}: map={res['map_name']}  map_id={res['map_id']:.0f}"
            f"  reward={res['reward']:.3f}  length={res['length']}"
            f"  won={res['battle_won']}"
        )

        # All maps must produce the same obs shapes.
        # (Already checked by assert_obs_shapes, but verify state/avail explicitly.)
        assert env.obs_space['state'].shape == base_obs_shape, (
            f"state shape changed for map '{res['map_name']}'"
        )
        assert env.obs_space['avail_actions'].shape == base_avail_shape, (
            f"avail_actions shape changed for map '{res['map_name']}'"
        )

    env.close()
    print(f"\n[PASS] All {n_maps} maps ran one complete episode with stable obs shapes.")

    # ---- Map cycling order -----------------------------------------------
    map_names_seen = [r['map_name'] for r in results]
    expected_names = [e.name for e in sampler.maps]
    assert map_names_seen == expected_names, (
        f"Round-robin order wrong: got {map_names_seen}, expected {expected_names}"
    )
    print("[PASS] Round-robin map order correct.")

    # ---- map_id values ---------------------------------------------------
    for r in results:
        expected_id = float([e.name for e in sampler.maps].index(r['map_name']))
        assert r['map_id'] == expected_id, (
            f"map_id wrong for '{r['map_name']}': got {r['map_id']}, expected {expected_id}"
        )
    print("[PASS] log/map_id values correct.")

    # ---- Shape rejection test --------------------------------------------
    test_shape_rejection(manifest_path)

    print(f"\n{'='*60}")
    print("All Phase 2 smoke tests PASSED.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
