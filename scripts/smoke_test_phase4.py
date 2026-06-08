"""
Phase 4 smoke test for DreamerV3 × SMAClite.

Verifies:
  1. Phase 4 manifest loads with version == 1.
  2. Train/validation/test sets are disjoint.
  3. Dataset hash is stable (recomputed in-memory and compared).
  4. Padding dims fit every train-split map (validate_padding_dims).
  5. All train maps (up to --max_maps) produce fixed padded observation shapes.
  6. shuffled_round_robin visits every map exactly once per cycle.
  7. uniform_family sampling is roughly balanced across families.
  8. Padded action indices >= n_real_agents never reach SMAClite.
  9. masking_failure_rate remains zero across several episodes.
 10. Phase 3 manifest still loads correctly (backward compatibility).
 11. W&B disabled mode does not raise.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\smoke_test_phase4.py ^
      --manifest configs\\maps\\phase4_manifest.yaml ^
      --max_maps 10
"""

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import ruamel.yaml as yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw(manifest_path: str) -> dict:
    return yaml.YAML(typ='safe').load(
        pathlib.Path(manifest_path).read_text(encoding='utf-8'))


def _recompute_dataset_hash(raw: dict, manifest_dir: pathlib.Path) -> str:
    all_entries = (
        raw.get('splits', {}).get('train', [])
        + raw.get('splits', {}).get('validation', [])
        + raw.get('splits', {}).get('test', [])
    )
    pairs = sorted((e['path'], e['file_hash']) for e in all_entries)
    blob = json.dumps(pairs, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _make_padded_env(entry_dict: dict, pad_dims, seed: int = 0):
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.map_sampler import MapSampler, MapEntry
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
        max_episode_steps=200,
        seed=seed,
        map_sampler=sampler,
        pad_dims=pad_dims,
    )
    return env


def _run_episode(env, max_steps: int = 50) -> dict:
    reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
    reset_act["reset"] = np.bool_(True)
    obs = env.step(reset_act)

    action_keys = sorted(
        [k for k in env.act_space if k.startswith("action_")],
        key=lambda k: int(k.split("_")[1]),
    )
    steps = 0
    while not bool(obs["is_last"]) and steps < max_steps:
        act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
        act["reset"] = np.bool_(False)
        avail_flat = obs["avail_actions"]
        n_max_actions = env.act_space[action_keys[0]].high
        n_max_agents = len(action_keys)
        for i, key in enumerate(action_keys):
            agent_avail = avail_flat[i * n_max_actions:(i + 1) * n_max_actions]
            valid = [j for j, v in enumerate(agent_avail) if v > 0]
            act[key] = np.int32(valid[0] if valid else 0)
        obs = env.step(act)
        steps += 1

    return {
        "masking_failure_rate": float(obs.get("log/masking_failure_rate", np.array(0.0))),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {name}" + (f": {detail}" if detail else ""))
    return condition


def run_tests(args):
    manifest_path = str(pathlib.Path(args.manifest).resolve())
    max_maps = args.max_maps
    seed = args.seed

    print(f"\nPhase 4 Smoke Test")
    print(f"  manifest  : {args.manifest}")
    print(f"  max_maps  : {max_maps}")
    print(f"  seed      : {seed}")
    print()

    # ------------------------------------------------------------------
    # Test 1: Version check
    # ------------------------------------------------------------------
    print("--- Test 1: Manifest version ---")
    try:
        raw = _load_raw(manifest_path)
        version = raw.get('version')
        check("manifest version == 1", version == 1, f"got {version!r}")
    except Exception as e:
        check("manifest loads", False, str(e))
        print("\nFATAL: cannot load manifest. Aborting.")
        return

    # ------------------------------------------------------------------
    # Test 2: Split disjoint
    # ------------------------------------------------------------------
    print("\n--- Test 2: Split disjoint ---")
    splits = raw.get('splits', {})
    train_paths = {e['path'] for e in splits.get('train', [])}
    val_paths   = {e['path'] for e in splits.get('validation', [])}
    test_paths  = {e['path'] for e in splits.get('test', [])}
    check("train ∩ validation = ∅", not (train_paths & val_paths),
          f"{len(train_paths & val_paths)} overlap(s)")
    check("train ∩ test = ∅", not (train_paths & test_paths),
          f"{len(train_paths & test_paths)} overlap(s)")
    check("validation ∩ test = ∅", not (val_paths & test_paths),
          f"{len(val_paths & test_paths)} overlap(s)")
    check("train not empty", len(train_paths) > 0, f"{len(train_paths)} maps")

    # ------------------------------------------------------------------
    # Test 3: Dataset hash stability
    # ------------------------------------------------------------------
    print("\n--- Test 3: Dataset hash stability ---")
    try:
        manifest_dir = pathlib.Path(args.manifest).resolve().parent
        expected_hash = raw.get('dataset_hash', '')
        recomputed    = _recompute_dataset_hash(raw, manifest_dir)
        check("dataset hash stable", expected_hash == recomputed,
              f"stored={expected_hash[:16]}... recomputed={recomputed[:16]}...")
    except Exception as e:
        check("dataset hash recompute", False, str(e))

    # ------------------------------------------------------------------
    # Test 4: Padding dims validation
    # ------------------------------------------------------------------
    print("\n--- Test 4: Padding dims fit all train maps ---")
    try:
        from smacdreamer.envs.padding import PaddingDims, validate_padding_dims
        from smacdreamer.envs.map_sampler import MapEntry
        pad_raw = raw.get('padding', {})
        pad_dims = PaddingDims(
            max_agents=pad_raw['max_agents'],
            max_enemies=pad_raw['max_enemies'],
            max_actions=pad_raw['max_actions'],
            max_obs_size=pad_raw['max_obs_size'],
        )
        train_entries = [
            MapEntry(name=e['name'], type=e.get('type','custom'), path=e.get('path'),
                     family=e.get('family','uncategorised'), map_id=e.get('map_id',0))
            for e in splits.get('train', [])[:max_maps]
        ]
        validate_padding_dims(train_entries, pad_dims)
        check("padding dims fit train maps", True,
              f"checked {len(train_entries)} maps, "
              f"agents={pad_dims.max_agents} enemies={pad_dims.max_enemies} "
              f"actions={pad_dims.max_actions} obs={pad_dims.max_obs_size}")
    except Exception as e:
        check("padding dims fit train maps", False, str(e))
        pad_dims = None

    # ------------------------------------------------------------------
    # Test 5: Fixed padded shapes
    # ------------------------------------------------------------------
    print("\n--- Test 5: Fixed padded observation shapes ---")
    if pad_dims is None:
        print("  [SKIP] No pad_dims available")
    else:
        train_list = splits.get('train', [])[:max_maps]
        all_state_shapes  = set()
        all_avail_shapes  = set()
        all_action_counts = set()
        failures = 0
        for entry_dict in train_list:
            try:
                env = _make_padded_env(entry_dict, pad_dims, seed=seed)
                reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
                reset_act["reset"] = np.bool_(True)
                obs = env.step(reset_act)
                all_state_shapes.add(obs["state"].shape)
                all_avail_shapes.add(obs["avail_actions"].shape)
                all_action_counts.add(len([k for k in env.act_space if k.startswith("action_")]))
                env.close()
            except Exception as e:
                failures += 1
                print(f"    ERROR for map {entry_dict.get('name')}: {e}")
        check("all maps produce same state shape", len(all_state_shapes) == 1,
              f"shapes seen: {all_state_shapes}")
        check("all maps produce same avail shape", len(all_avail_shapes) == 1,
              f"shapes seen: {all_avail_shapes}")
        check("all maps produce same action key count", len(all_action_counts) == 1,
              f"counts: {all_action_counts}")
        check("no env failures", failures == 0, f"{failures} failure(s)")

    # ------------------------------------------------------------------
    # Test 6: shuffled_round_robin visits each map once per cycle
    # ------------------------------------------------------------------
    print("\n--- Test 6: shuffled_round_robin cycle coverage ---")
    try:
        from smacdreamer.envs.map_sampler import MapSampler
        sampler = MapSampler.from_phase4_manifest(
            manifest_path, split='train', mode='shuffled_round_robin', seed=seed)
        n = len(sampler.maps)
        # Run a full cycle when n is small enough; otherwise run max_maps*5 steps
        # and check that there are NO duplicates (partial-cycle uniqueness guarantee).
        n_probe = min(n, max(max_maps * 5, 50))
        visited = []
        for _ in range(n_probe):
            e = sampler.next()
            visited.append(e.name)

        # The partial cycle must have no duplicates.
        check("shuffled_round_robin: no duplicates in partial cycle",
              len(set(visited)) == len(visited),
              f"called {n_probe} times, got {len(set(visited))} unique (expected {n_probe})")

        # If we ran a full cycle, verify complete coverage.
        if n_probe == n:
            check("shuffled_round_robin: full cycle covers all maps",
                  len(set(visited)) == n,
                  f"cycle={n}, unique={len(set(visited))}")
        else:
            print(f"    (full-cycle check skipped: n={n} > probe={n_probe}; "
                  f"use --max_maps {n//5 + 1} for a full-cycle check)")

        check("coverage fraction increases", sampler.dataset_coverage_fraction > 0,
              f"coverage={sampler.dataset_coverage_fraction:.3f}")
    except Exception as e:
        check("shuffled_round_robin cycle", False, str(e))

    # ------------------------------------------------------------------
    # Test 7: uniform_family balance
    # ------------------------------------------------------------------
    print("\n--- Test 7: uniform_family sampling balance ---")
    try:
        from smacdreamer.envs.map_sampler import MapSampler
        sampler_uf = MapSampler.from_phase4_manifest(
            manifest_path, split='train', mode='uniform_family', seed=seed)
        families = list({e.family for e in sampler_uf.maps})
        if len(families) <= 1:
            print("  [SKIP] Only one family in dataset")
        else:
            n_samples = max(200, 10 * len(families))
            fam_counts: Counter = Counter()
            for _ in range(n_samples):
                e = sampler_uf.next()
                fam_counts[e.family] += 1
            counts = list(fam_counts.values())
            expected = n_samples / len(families)
            max_dev = max(abs(c - expected) / expected for c in counts)
            check("uniform_family: max deviation < 50%", max_dev < 0.5,
                  f"families={len(families)} max_deviation={max_dev:.2%}")
    except Exception as e:
        check("uniform_family balance", False, str(e))

    # ------------------------------------------------------------------
    # Test 8: Padded actions don't reach SMAClite
    # ------------------------------------------------------------------
    print("\n--- Test 8: Padded actions are ignored ---")
    if pad_dims is None or not train_list:
        print("  [SKIP] No pad_dims or no train maps")
    else:
        entry_dict = train_list[0]
        try:
            env = _make_padded_env(entry_dict, pad_dims, seed=seed)
            n_real = entry_dict.get('n_agents', 1)
            n_max  = len([k for k in env.act_space if k.startswith("action_")])

            reset_act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
            reset_act["reset"] = np.bool_(True)
            obs = env.step(reset_act)

            # Set padded agent actions to a high value that would be invalid
            act = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
            act["reset"] = np.bool_(False)
            avail_flat = obs["avail_actions"]
            n_max_a = pad_dims.max_actions
            for i in range(n_real):
                avail = avail_flat[i * n_max_a:(i + 1) * n_max_a]
                valid = [j for j, v in enumerate(avail) if v > 0]
                act[f"action_{i}"] = np.int32(valid[0] if valid else 0)
            for i in range(n_real, n_max):
                # Set padded agent actions to something that would crash SMAClite
                act[f"action_{i}"] = np.int32(999)

            # Should NOT crash (padded actions are ignored in _reset step logic)
            obs2 = env.step(act)
            check("padded actions don't crash step", True, f"n_real={n_real} n_max={n_max}")
            env.close()
        except Exception as e:
            check("padded actions don't crash step", False, str(e))

    # ------------------------------------------------------------------
    # Test 9: masking_failure_rate remains zero
    # ------------------------------------------------------------------
    print("\n--- Test 9: masking_failure_rate = 0 ---")
    if pad_dims is None or not train_list:
        print("  [SKIP] No pad_dims or no train maps")
    else:
        test_entries = train_list[:min(3, len(train_list))]
        all_fail_rates = []
        ep_errors = 0
        for entry_dict in test_entries:
            try:
                env = _make_padded_env(entry_dict, pad_dims, seed=seed)
                for _ in range(3):
                    metrics = _run_episode(env, max_steps=30)
                    all_fail_rates.append(metrics["masking_failure_rate"])
                env.close()
            except Exception as e:
                ep_errors += 1
                print(f"    ERROR for {entry_dict.get('name')}: {e}")
        if all_fail_rates:
            max_fail = max(all_fail_rates)
            check("masking_failure_rate == 0", max_fail == 0.0,
                  f"max across {len(all_fail_rates)} episodes: {max_fail:.6f}")
        if ep_errors > 0:
            check("episode runs without error", False, f"{ep_errors} error(s)")

    # ------------------------------------------------------------------
    # Test 10: Phase 3 manifest backward compatibility
    # ------------------------------------------------------------------
    print("\n--- Test 10: Phase 3 manifest backward compatibility ---")
    p3_manifest = ROOT / "configs" / "manifests" / "phase3_manifest.yaml"
    if not p3_manifest.exists():
        print("  [SKIP] Phase 3 manifest not found at configs/manifests/phase3_manifest.yaml")
    else:
        try:
            from smacdreamer.envs.map_sampler import MapSampler, validate_manifest
            raw3 = validate_manifest(str(p3_manifest))
            sampler3 = MapSampler.from_manifest(str(p3_manifest), mode='round_robin', seed=0)
            e1 = sampler3.next()
            e2 = sampler3.next()
            check("phase3 manifest loads", True,
                  f"{len(raw3['maps'])} maps, first={e1.name}")
        except Exception as e:
            check("phase3 manifest loads", False, str(e))

    # ------------------------------------------------------------------
    # Test 11: W&B disabled mode
    # ------------------------------------------------------------------
    print("\n--- Test 11: W&B disabled mode ---")
    try:
        import wandb
        wandb.init(project="test", mode="disabled")
        wandb.log({"test_metric": 1.0})
        wandb.finish()
        check("wandb disabled mode", True)
    except Exception as e:
        check("wandb disabled mode", False, str(e))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"Smoke test complete: {n_pass} passed, {n_fail} failed")
    print(f"{'='*60}")
    if n_fail > 0:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  ✗ {name}: {detail}")
        sys.exit(1)
    print("\nAll Phase 4 smoke tests passed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 smoke test for DreamerV3 × SMAClite."
    )
    parser.add_argument("--manifest", default="configs/maps/phase4_manifest.yaml",
                        help="Path to the Phase 4 manifest YAML.")
    parser.add_argument("--max_maps", type=int, default=10,
                        help="Maximum maps to probe in shape/masking tests.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    args = parser.parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
