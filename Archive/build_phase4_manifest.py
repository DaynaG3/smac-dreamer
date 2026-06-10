"""
Build a Phase 4 folder-driven dataset manifest for DreamerV3 × SMAClite.

Recursively discovers all *.json map files under --map_dir, validates each one,
splits them into train/validation/test sets, and writes a versioned manifest YAML
plus a JSON report.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\build_phase4_manifest.py ^
      --map_dir configs\\maps\\500map_v1 ^
      --output_manifest configs\\maps\\phase4_manifest.yaml ^
      --output_report results\\phase4_manifest_report.json ^
      --seed 42 ^
      --train_ratio 0.80 --validation_ratio 0.10 --test_ratio 0.10 ^
      --recursive --family_from_parent --on_exceed exclude
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ruamel.yaml as yaml

# Shared scan/probe core — single source of truth (no duplicated scan logic).
# validate_map / sha256_file return the same result-dict contract this builder relies on.
from smacdreamer.envs.map_discovery import (
    KNOWN_UNIT_TYPES,          # noqa: F401 (kept for backward import compatibility)
    validate_map,
    sha256_file as _sha256_file,
)


# ---------------------------------------------------------------------------
# Dataset-level hash
# ---------------------------------------------------------------------------

def _dataset_hash(included: list) -> str:
    """Stable hash over sorted (rel_path, file_hash) pairs."""
    pairs = sorted((r["rel_path"], r["file_hash"]) for r in included)
    blob = json.dumps(pairs, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(included: list, train_r: float, val_r: float, test_r: float,
                      seed: int) -> tuple:
    """Split by family; returns (train, validation, test) lists of result dicts."""
    by_family = defaultdict(list)
    for r in included:
        by_family[r["map_info"]["family"]].append(r)

    train, val, test = [], [], []
    rng = random.Random(seed)

    for family, maps in sorted(by_family.items()):
        rng.shuffle(maps)
        n = len(maps)
        if n < 3:
            print(f"  WARNING: family '{family}' has only {n} map(s); "
                  "all assigned to train.")
            train.extend(maps)
            continue
        n_val  = max(1, round(n * val_r))
        n_test = max(1, round(n * test_r))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = (n - 1) // 2
            n_test = n - 1 - n_val
        train.extend(maps[:n_train])
        val.extend(maps[n_train:n_train + n_val])
        test.extend(maps[n_train + n_val:])

    return train, val, test


# ---------------------------------------------------------------------------
# Manifest entry builder
# ---------------------------------------------------------------------------

def _make_entry(r: dict, map_id: int) -> dict:
    mi = r["map_info"]
    return {
        "map_id":    map_id,
        "name":      mi["name"],
        "type":      "custom",
        "path":      r["rel_path"],
        "family":    mi["family"],
        "n_agents":  mi["n_agents"],
        "n_enemies": mi["n_enemies"],
        "n_actions": mi["n_actions"],
        "obs_size":  mi["obs_size"],
        "file_hash": r["file_hash"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Phase 4 folder-driven map manifest."
    )
    parser.add_argument("--map_dir", default="configs/maps/500map_v1",
                        help="Folder to scan for *.json maps.")
    parser.add_argument("--output_manifest", default="configs/maps/phase4_manifest.yaml",
                        help="Output manifest YAML path.")
    parser.add_argument("--output_report", default="results/phase4_manifest_report.json",
                        help="Output JSON report path.")
    parser.add_argument("--dataset_name", default="smaclite_phase4",
                        help="Dataset name written into the manifest.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio",      type=float, default=0.80)
    parser.add_argument("--validation_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio",       type=float, default=0.10)
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Recurse into subdirectories (default: on).")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false")
    parser.add_argument("--family_from_parent", action="store_true", default=True,
                        help="Derive family from first parent dir below map_dir.")
    parser.add_argument("--no_family_from_parent", dest="family_from_parent",
                        action="store_false")
    parser.add_argument("--max_agents",  type=int, default=None,
                        help="Safety limit on n_agents per map.")
    parser.add_argument("--max_enemies", type=int, default=None)
    parser.add_argument("--max_actions", type=int, default=None)
    parser.add_argument("--max_obs_size",type=int, default=None)
    parser.add_argument("--on_exceed",
                        choices=["error", "exclude", "separate_bucket"],
                        default="exclude",
                        help="Action when a map exceeds safety limits.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the reason for every invalid/excluded map.")
    return parser.parse_args()


def main():
    args = parse_args()

    map_dir = (ROOT / args.map_dir).resolve()
    if not map_dir.exists():
        sys.exit(f"ERROR: --map_dir does not exist: {map_dir}")

    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        sys.exit(f"ERROR: ratios must sum to 1.0, got {ratio_sum:.4f}")

    limits = {
        "max_agents":  args.max_agents,
        "max_enemies": args.max_enemies,
        "max_actions": args.max_actions,
        "max_obs_size":args.max_obs_size,
    }

    print(f"\nPhase 4 manifest builder")
    print(f"  map_dir         : {map_dir}")
    print(f"  recursive       : {args.recursive}")
    print(f"  family_from_dir : {args.family_from_parent}")
    print(f"  seed            : {args.seed}")
    print(f"  split           : {args.train_ratio:.0%} / {args.validation_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"  on_exceed       : {args.on_exceed}")
    active_limits = {k: v for k, v in limits.items() if v is not None}
    if active_limits:
        print(f"  limits          : {active_limits}")

    # --- Discover maps ---
    pattern = "**/*.json" if args.recursive else "*.json"
    all_paths = sorted(map_dir.glob(pattern))
    print(f"\nDiscovered {len(all_paths)} JSON file(s).\n")

    if not all_paths:
        sys.exit("ERROR: no .json files found under map_dir.")

    # Detect duplicate filenames
    seen_names: dict = {}
    for p in all_paths:
        stem = p.stem
        if stem in seen_names:
            print(f"  WARNING: duplicate filename '{stem}' at {p} "
                  f"(first: {seen_names[stem]}); skipping duplicate.")
        else:
            seen_names[stem] = p

    unique_paths = [seen_names[stem] for stem in sorted(seen_names)]

    # --- Validate each map ---
    included = []
    excluded = []
    invalid  = []
    seen_hashes: dict = {}

    print(f"Validating {len(unique_paths)} unique map file(s) ...")
    for i, path in enumerate(unique_paths):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique_paths)}] {path.name} ...")
        result = validate_map(
            path, map_dir, args.family_from_parent, limits, args.on_exceed)

        if not result["ok"]:
            if result.get("limit_exceeded"):
                excluded.append({
                    "path":   result["rel_path"],
                    "reason": result["reason"],
                    "dimensions": result.get("dimensions", {}),
                })
                if args.verbose:
                    print(f"  EXCLUDED  {path.name}: {result['reason']}")
            else:
                invalid.append({
                    "path":   result["rel_path"],
                    "reason": result["reason"],
                })
                # Always print the first invalid map; rest only with --verbose
                if args.verbose or len(invalid) == 1:
                    print(f"  INVALID   {path.name}: {result['reason']}")
            continue

        fh = result["file_hash"]
        if fh in seen_hashes:
            print(f"  EXCLUDED  {path.name}: duplicate content of {seen_hashes[fh]}")
            excluded.append({"path": result["rel_path"],
                             "reason": f"duplicate content of {seen_hashes[fh]}"})
            continue
        seen_hashes[fh] = result["rel_path"]
        included.append(result)

    print(f"\nValidation complete:")
    print(f"  included : {len(included)}")
    print(f"  excluded : {len(excluded)}")
    print(f"  invalid  : {len(invalid)}")

    if not included:
        sys.exit("ERROR: no valid maps remain after validation.")

    # --- Assign stable map_id (order of included list, sorted by rel_path) ---
    included.sort(key=lambda r: r["rel_path"])
    for mid, r in enumerate(included):
        r["map_id"] = mid

    # --- Dataset hash ---
    ds_hash = _dataset_hash(included)
    print(f"\nDataset hash: {ds_hash[:16]}...")

    # --- Stratified split ---
    train, val, test = _stratified_split(
        included, args.train_ratio, args.validation_ratio, args.test_ratio, args.seed)

    # Verify disjoint
    train_paths = {r["rel_path"] for r in train}
    val_paths   = {r["rel_path"] for r in val}
    test_paths  = {r["rel_path"] for r in test}
    assert not (train_paths & val_paths),  "ASSERT FAIL: train ∩ val overlap"
    assert not (train_paths & test_paths), "ASSERT FAIL: train ∩ test overlap"
    assert not (val_paths   & test_paths), "ASSERT FAIL: val ∩ test overlap"

    print(f"\nSplit:")
    print(f"  train      : {len(train)}")
    print(f"  validation : {len(val)}")
    print(f"  test       : {len(test)}")

    # Family split counts
    def _family_counts(split_list):
        c = defaultdict(int)
        for r in split_list:
            c[r["map_info"]["family"]] += 1
        return dict(c)

    fam_train = _family_counts(train)
    fam_val   = _family_counts(val)
    fam_test  = _family_counts(test)
    all_families = sorted(set(list(fam_train) + list(fam_val) + list(fam_test)))

    family_split_counts = {
        f: {"train": fam_train.get(f, 0),
            "validation": fam_val.get(f, 0),
            "test": fam_test.get(f, 0)}
        for f in all_families
    }
    for f in all_families:
        c = family_split_counts[f]
        print(f"    family '{f}': "
              f"train={c['train']} val={c['validation']} test={c['test']}")

    # --- Compute padding dims ---
    max_agents  = max(r["map_info"]["n_agents"]  for r in included)
    max_enemies = max(r["map_info"]["n_enemies"] for r in included)
    max_actions = max(r["map_info"]["n_actions"] for r in included)
    max_obs_size= max(r["map_info"]["obs_size"]  for r in included)

    print(f"\nPadding dimensions (from {len(included)} included maps):")
    print(f"  max_agents   = {max_agents}")
    print(f"  max_enemies  = {max_enemies}")
    print(f"  max_actions  = {max_actions}")
    print(f"  max_obs_size = {max_obs_size}")

    # Padding waste ratio for a representative map (median agent count)
    median_r = sorted(included, key=lambda r: r["map_info"]["n_agents"])[len(included)//2]
    mi = median_r["map_info"]
    agent_waste  = 1.0 - mi["n_agents"]  / max_agents  if max_agents  > 0 else 0.0
    action_waste = 1.0 - mi["n_actions"] / max_actions if max_actions > 0 else 0.0
    obs_waste    = 1.0 - mi["obs_size"]  / max_obs_size if max_obs_size > 0 else 0.0
    print(f"\nPadding waste (median map: {mi['name']}):")
    print(f"  agent slot waste  = {agent_waste:.1%}")
    print(f"  action slot waste = {action_waste:.1%}")
    print(f"  obs slot waste    = {obs_waste:.1%}")

    # --- Build manifest YAML ---
    manifest = {
        "version": 1,
        "dataset_name": args.dataset_name,
        "dataset_root": str(map_dir.relative_to(ROOT)).replace("\\", "/"),
        "dataset_hash": ds_hash,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_algorithm": "stratified_family",
        "padding": {
            "max_agents":   max_agents,
            "max_enemies":  max_enemies,
            "max_actions":  max_actions,
            "max_obs_size": max_obs_size,
        },
        "limits": {k: v for k, v in limits.items()},
        "split_counts": {
            "train":      len(train),
            "validation": len(val),
            "test":       len(test),
        },
        "family_split_counts": family_split_counts,
        "splits": {
            "train":      [_make_entry(r, r["map_id"]) for r in train],
            "validation": [_make_entry(r, r["map_id"]) for r in val],
            "test":       [_make_entry(r, r["map_id"]) for r in test],
        },
        "excluded_maps": excluded,
        "invalid_maps":  invalid,
    }

    out_manifest = ROOT / args.output_manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    ym = yaml.YAML()
    ym.default_flow_style = False
    ym.width = 120
    with open(out_manifest, "w", encoding="utf-8") as f:
        ym.dump(manifest, f)
    print(f"\nManifest written : {out_manifest}")

    # --- Build JSON report ---
    report = {
        "dataset_name":   args.dataset_name,
        "dataset_hash":   ds_hash,
        "map_dir":        str(map_dir),
        "seed":           args.seed,
        "generated_at":   manifest["generated_at"],
        "total_discovered": len(all_paths),
        "total_unique":     len(unique_paths),
        "total_included":   len(included),
        "total_excluded":   len(excluded),
        "total_invalid":    len(invalid),
        "split_counts":     manifest["split_counts"],
        "family_split_counts": family_split_counts,
        "padding":          manifest["padding"],
        "padding_waste_median": {
            "map": mi["name"],
            "agent_waste":  round(agent_waste,  4),
            "action_waste": round(action_waste, 4),
            "obs_waste":    round(obs_waste,    4),
        },
        "excluded_maps": excluded,
        "invalid_maps":  invalid,
        "included_maps": [
            {
                "path":      r["rel_path"],
                "name":      r["map_info"]["name"],
                "family":    r["map_info"]["family"],
                "n_agents":  r["map_info"]["n_agents"],
                "n_enemies": r["map_info"]["n_enemies"],
                "n_actions": r["map_info"]["n_actions"],
                "obs_size":  r["map_info"]["obs_size"],
                "file_hash": r["file_hash"],
            }
            for r in included
        ],
    }

    out_report = ROOT / args.output_report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written   : {out_report}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Phase 4 manifest build summary")
    print(f"{'='*60}")
    print(f"  Dataset name : {args.dataset_name}")
    print(f"  Dataset hash : {ds_hash[:24]}...")
    print(f"  Total maps   : {len(included)} included, {len(excluded)} excluded, {len(invalid)} invalid")
    print(f"  Train        : {len(train)}")
    print(f"  Validation   : {len(val)}")
    print(f"  Test         : {len(test)}")
    print(f"  Padding      : agents={max_agents} enemies={max_enemies} "
          f"actions={max_actions} obs={max_obs_size}")
    print(f"{'='*60}\n")

    if not val:
        print("WARNING: validation split is empty.")
    if not test:
        print("WARNING: test split is empty.")

    sys.exit(0)


if __name__ == "__main__":
    main()
