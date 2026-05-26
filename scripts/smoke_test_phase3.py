"""
Phase 3 smoke test: padded multi-map training adapter.

Ten assertions per map:
  1.  Padded obs shapes: state, avail_actions, agent_mask, real_agent_action_mask
  2.  agent_mask: 1.0 for real slots, 0.0 for padded slots
  3.  real_agent_action_mask == np.repeat(agent_mask, max_actions)
  4.  avail_actions[n_real*max_actions:] all zeros (padded agent slots)
  5.  real_agent_action_mask[:n_real*max_actions] all 1.0 (semantic distinctness from avail)
  6.  Log metrics at reset: num_real_agents, num_real_enemies, padded_agent_count, etc.
  7.  5 valid steps: shapes stable, no crash
  8.  Forced padded-agent actions (large invalid): no crash, shapes/mask unchanged
  9.  validate_padding_dims raises ValueError for too-small PaddingDims
  10. Backward compat: pad_dims=None gives real-dim obs, no agent_mask keys

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite
    python scripts\\smoke_test_phase3.py --manifest configs\\maps\\phase3_manifest.yaml
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import ruamel.yaml as yaml

from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
from smacdreamer.envs.map_sampler import MapEntry, MapSampler
from smacdreamer.envs.padding import PaddingDims, validate_padding_dims


def load_manifest(path: str):
    """Return (PaddingDims, list[MapEntry])."""
    raw = yaml.YAML(typ='safe').load(pathlib.Path(path).read_text(encoding='utf-8'))
    p = raw['padding']
    pad_dims = PaddingDims(
        max_agents=p['max_agents'],
        max_enemies=p['max_enemies'],
        max_actions=p['max_actions'],
        max_obs_size=p['max_obs_size'],
    )
    entries = [
        MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
        for e in raw['maps']
    ]
    return pad_dims, entries


def make_reset_action(pad_dims: PaddingDims) -> dict:
    act = {f"action_{i}": np.int32(0) for i in range(pad_dims.max_agents)}
    act["reset"] = np.bool_(True)
    return act


def make_valid_action(env: SMACliteDreamerEnv, pad_dims: PaddingDims) -> dict:
    avail = env._env.unwrapped.get_avail_actions()
    act = {"reset": np.bool_(False)}
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        act[f"action_{i}"] = np.int32(np.random.choice(valid))
    for i in range(env.n_agents, pad_dims.max_agents):
        act[f"action_{i}"] = np.int32(0)
    return act


def make_forced_padded_invalid_action(env: SMACliteDreamerEnv, pad_dims: PaddingDims) -> dict:
    """Valid actions for real agents, large out-of-range values for padded slots."""
    avail = env._env.unwrapped.get_avail_actions()
    act = {"reset": np.bool_(False)}
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        act[f"action_{i}"] = np.int32(np.random.choice(valid))
    for i in range(env.n_agents, pad_dims.max_agents):
        act[f"action_{i}"] = np.int32(9999)
    return act


def test_map(entry: MapEntry, pad_dims: PaddingDims) -> None:
    MA = pad_dims.max_agents
    MC = pad_dims.max_actions
    MO = pad_dims.max_obs_size

    print(f"\n--- Map: {entry.name} ---")

    sampler = MapSampler([entry], mode='fixed')
    env = SMACliteDreamerEnv(
        scenario=entry.name,
        max_episode_steps=200,
        seed=0,
        map_sampler=sampler,
        pad_dims=pad_dims,
    )

    n_real = env.n_agents
    n_enemies = env.n_enemies
    n_actions = env.n_actions
    n_padded = MA - n_real

    print(f"  n_real={n_real}  n_enemies={n_enemies}  n_actions={n_actions}  "
          f"n_padded={n_padded}")

    # --- Test 1: Reset obs shapes ----------------------------------------
    obs = env.step(make_reset_action(pad_dims))

    assert obs['state'].shape == (MA * MO,), \
        f"[{entry.name}] state shape {obs['state'].shape} != ({MA * MO},)"
    assert obs['avail_actions'].shape == (MA * MC,), \
        f"[{entry.name}] avail_actions shape {obs['avail_actions'].shape} != ({MA * MC},)"
    assert obs['agent_mask'].shape == (MA,), \
        f"[{entry.name}] agent_mask shape {obs['agent_mask'].shape} != ({MA},)"
    assert obs['real_agent_action_mask'].shape == (MA * MC,), \
        f"[{entry.name}] real_agent_action_mask shape {obs['real_agent_action_mask'].shape} != ({MA * MC},)"
    print("[PASS] Test 1: Padded obs shapes correct.")

    # --- Test 2: agent_mask values ----------------------------------------
    mask = obs['agent_mask']
    np.testing.assert_array_equal(
        mask[:n_real], np.ones(n_real, dtype=np.float32),
        err_msg=f"[{entry.name}] agent_mask[:n_real] should all be 1.0")
    np.testing.assert_array_equal(
        mask[n_real:], np.zeros(n_padded, dtype=np.float32),
        err_msg=f"[{entry.name}] agent_mask[n_real:] should all be 0.0")
    print("[PASS] Test 2: agent_mask values correct.")

    # --- Test 3: real_agent_action_mask == np.repeat(agent_mask, MC) ------
    expected_raam = np.repeat(mask, MC)
    np.testing.assert_array_equal(
        obs['real_agent_action_mask'], expected_raam,
        err_msg=f"[{entry.name}] real_agent_action_mask != np.repeat(agent_mask, max_actions)")
    print("[PASS] Test 3: real_agent_action_mask == np.repeat(agent_mask, max_actions).")

    # --- Test 4: avail_actions padded slots all zero ----------------------
    if n_padded > 0:
        padded_avail = obs['avail_actions'][n_real * MC:]
        assert np.all(padded_avail == 0.0), \
            f"[{entry.name}] avail_actions[n_real*MC:] not all zero"
    print("[PASS] Test 4: avail_actions padded agent slots all zero.")

    # --- Test 5: real_agent_action_mask != avail_actions (distinct semantics)
    # real_agent_action_mask[i*MC:(i+1)*MC] = 1.0 for all real agent i, regardless of
    # whether each action is available. avail_actions encodes per-action availability.
    # They encode different information and must be different arrays.
    raam = obs['real_agent_action_mask']
    avail_flat = obs['avail_actions']
    np.testing.assert_array_equal(
        raam[:n_real * MC], np.ones(n_real * MC, dtype=np.float32),
        err_msg=f"[{entry.name}] real_agent_action_mask real-agent slots should all be 1.0")
    # Verify they are distinct objects (not the same array).
    assert raam is not avail_flat, \
        f"[{entry.name}] real_agent_action_mask and avail_actions are the same object"
    # For maps where max_actions > n_actions: extra slots in raam are 1.0 but 0 in avail.
    if MC > n_actions:
        extra_raam = raam[n_actions: MC]  # extra slots of first real agent
        extra_avail = avail_flat[n_actions: MC]
        np.testing.assert_array_equal(
            extra_raam, np.ones(MC - n_actions, dtype=np.float32),
            err_msg=f"[{entry.name}] raam extra action slots for real agents should be 1.0")
        np.testing.assert_array_equal(
            extra_avail, np.zeros(MC - n_actions, dtype=np.float32),
            err_msg=f"[{entry.name}] avail_actions extra action slots should be 0.0")
    print("[PASS] Test 5: real_agent_action_mask and avail_actions are semantically distinct.")

    # --- Test 6: Log metrics at reset ------------------------------------
    extra_real_slots = n_real * (MC - n_actions)
    padded_agent_slots = n_padded * MC
    ignored_pad_actions = n_padded

    assert float(obs['log/num_real_agents']) == float(n_real), \
        f"[{entry.name}] log/num_real_agents {obs['log/num_real_agents']} != {n_real}"
    assert float(obs['log/num_real_enemies']) == float(n_enemies), \
        f"[{entry.name}] log/num_real_enemies {obs['log/num_real_enemies']} != {n_enemies}"
    assert float(obs['log/padded_agent_count']) == float(n_padded), \
        f"[{entry.name}] log/padded_agent_count {obs['log/padded_agent_count']} != {n_padded}"
    assert float(obs['log/extra_real_agent_action_slot_count']) == float(extra_real_slots), \
        (f"[{entry.name}] log/extra_real_agent_action_slot_count "
         f"{obs['log/extra_real_agent_action_slot_count']} != {extra_real_slots}")
    assert float(obs['log/padded_agent_action_slot_count']) == float(padded_agent_slots), \
        (f"[{entry.name}] log/padded_agent_action_slot_count "
         f"{obs['log/padded_agent_action_slot_count']} != {padded_agent_slots}")
    assert float(obs['log/ignored_padded_agent_action_count']) == float(ignored_pad_actions), \
        (f"[{entry.name}] log/ignored_padded_agent_action_count "
         f"{obs['log/ignored_padded_agent_action_count']} != {ignored_pad_actions}")
    assert float(obs['log/agent_mask_sum']) == float(n_real), \
        f"[{entry.name}] log/agent_mask_sum {obs['log/agent_mask_sum']} != {n_real}"
    assert 'log/map_id' in obs, f"[{entry.name}] log/map_id missing from obs"
    assert float(obs['log/map_id']) >= 0.0, \
        f"[{entry.name}] log/map_id {obs['log/map_id']} < 0"
    print("[PASS] Test 6: All Phase 3 log metrics correct at reset step.")

    # --- Test 7: 5 valid steps, shapes stable ----------------------------
    for _ in range(5):
        step_obs = env.step(make_valid_action(env, pad_dims))
        assert step_obs['state'].shape == (MA * MO,), \
            f"[{entry.name}] state shape changed mid-episode"
        assert step_obs['avail_actions'].shape == (MA * MC,), \
            f"[{entry.name}] avail_actions shape changed mid-episode"
        assert step_obs['agent_mask'].shape == (MA,), \
            f"[{entry.name}] agent_mask shape changed mid-episode"
        assert step_obs['real_agent_action_mask'].shape == (MA * MC,), \
            f"[{entry.name}] real_agent_action_mask shape changed mid-episode"
        if step_obs['is_last']:
            break
    print("[PASS] Test 7: Shapes stable over 5 valid steps.")

    # --- Test 8: Forced padded-agent actions (large invalid) -------------
    env.step(make_reset_action(pad_dims))  # fresh episode
    obs_after_reset = env.step(make_reset_action(pad_dims))
    forced_act = make_forced_padded_invalid_action(env, pad_dims)
    forced_obs = env.step(forced_act)
    assert forced_obs['state'].shape == (MA * MO,), \
        f"[{entry.name}] state shape changed after forced padded actions"
    assert forced_obs['agent_mask'].shape == (MA,), \
        f"[{entry.name}] agent_mask shape changed after forced padded actions"
    np.testing.assert_array_equal(
        forced_obs['agent_mask'], obs_after_reset['agent_mask'],
        err_msg=f"[{entry.name}] agent_mask changed after forced padded action step")
    print("[PASS] Test 8: Forced padded-agent large invalid actions: no crash, shapes unchanged.")

    env.close()


def test_validate_padding_dims_error(entries: list) -> None:
    """validate_padding_dims raises ValueError naming the map and dimension."""
    # Use a map known to have n_agents > 1 so max_agents=1 is definitely too small.
    target = next(
        (e for e in entries if e.name in ('3s5z', '3s5z_vs_3s6z')),
        entries[0],
    )
    small_dims = PaddingDims(max_agents=1, max_enemies=99, max_actions=99, max_obs_size=999)
    try:
        validate_padding_dims([target], small_dims)
        raise AssertionError(
            f"validate_padding_dims should have raised ValueError for "
            f"{target.name} with max_agents=1"
        )
    except ValueError as exc:
        msg = str(exc)
        assert target.name in msg, \
            f"Error message missing map name '{target.name}': {msg}"
        assert 'max_agents' in msg, \
            f"Error message missing 'max_agents': {msg}"
    print("[PASS] Test 9: validate_padding_dims raises ValueError naming map and dim.")


def test_backward_compat() -> None:
    """pad_dims=None: no agent_mask or real_agent_action_mask; real map dims used."""
    env = SMACliteDreamerEnv(scenario='2s3z', max_episode_steps=200, seed=0)
    act = {f"action_{i}": np.int32(0) for i in range(env.n_agents)}
    act["reset"] = np.bool_(True)
    obs = env.step(act)

    assert 'agent_mask' not in obs, \
        "agent_mask present in obs when pad_dims=None (backward compat broken)"
    assert 'real_agent_action_mask' not in obs, \
        "real_agent_action_mask present in obs when pad_dims=None (backward compat broken)"

    expected_state = (env.n_agents * env.obs_size,)
    assert obs['state'].shape == expected_state, \
        f"state shape {obs['state'].shape} != {expected_state} when pad_dims=None"

    expected_avail = (env.n_agents * env.n_actions,)
    assert obs['avail_actions'].shape == expected_avail, \
        f"avail_actions shape {obs['avail_actions'].shape} != {expected_avail} when pad_dims=None"

    env.close()
    print("[PASS] Test 10: Backward compat — pad_dims=None gives real-dim obs, no mask keys.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 smoke test for padded multi-map SMACliteDreamerEnv."
    )
    parser.add_argument("--manifest", default="configs/maps/phase3_manifest.yaml",
                        help="Path to the Phase 3 manifest YAML.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = str(ROOT / args.manifest)
    pad_dims, entries = load_manifest(manifest_path)

    print(f"\n{'='*60}")
    print("Phase 3 smoke test")
    print(f"Manifest : {args.manifest}")
    print(f"Padding  : max_agents={pad_dims.max_agents}  max_actions={pad_dims.max_actions}"
          f"  max_obs_size={pad_dims.max_obs_size}  max_enemies={pad_dims.max_enemies}")
    print(f"Maps     : {[e.name for e in entries]}")
    print(f"{'='*60}")

    np.random.seed(args.seed)

    for entry in entries:
        test_map(entry, pad_dims)

    test_validate_padding_dims_error(entries)
    test_backward_compat()

    print(f"\n{'='*60}")
    print("All Phase 3 smoke tests PASSED.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
