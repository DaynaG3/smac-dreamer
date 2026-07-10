from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

SEED = 19062026
MAP_WIDTH = 32
MAP_HEIGHT = 32
GROUP_BUFFER = 0.05
DATASET_NAME = "r2_smaclite_simple_2v1_stalker_240_configs"
GLOBAL_UNIT_TYPE_IDS = {
    "BANELING": 0,
    "COLOSSUS": 1,
    "MARAUDER": 2,
    "MARINE": 3,
    "MEDIVAC": 4,
    "SPINE_CRAWLER": 5,
    "STALKER": 6,
    "ZEALOT": 7,
    "ZERGLING": 8,
}

# Minimal unit catalogue needed for validation and metadata. The numbers match the
# values used in the previous generated benchmark script and are only used here for
# static checks and difficulty metadata, not for SMAClite runtime.
UNITS = {
    "STALKER": {
        "hp": 80, "shield": 80, "damage": 13, "attacks": 1, "cooldown": 1.34,
        "speed": 4.13, "attack_range": 6.0, "size": 1.25, "plane": "GROUND",
        "combat_type": "DAMAGE", "valid_targets": ("GROUND", "AIR"), "value": 125,
    }
}
TERRAINS = ("SIMPLE", "NARROW", "OCTAGON")
FORMATIONS = ("compact", "close_split", "staggered", "wide_split")
ENGAGEMENT_TARGETS = {
    "immediate": 5.2,
    "near": 8.2,
    "medium": 11.4,
    "far": 14.4,
}
ENGAGEMENT_BOUNDS = {
    "immediate": (4.4, 6.4),
    "near": (7.0, 9.6),
    "medium": (10.1, 12.9),
    "far": (13.1, 16.2),
}
SPLIT_TARGETS = {
    "train": {"total": 160, "engagement": {"immediate": 40, "near": 72, "medium": 40, "far": 8}},
    "validation": {"total": 40, "engagement": {"immediate": 10, "near": 18, "medium": 10, "far": 2}},
    "blind_iid": {"total": 40, "engagement": {"immediate": 10, "near": 18, "medium": 10, "far": 2}},
}


def unit_radius(unit: str) -> float:
    return float(UNITS[unit]["size"]) / 2.0


def comp_count(comp: Mapping[str, int]) -> int:
    return sum(comp.values())


def comp_value(comp: Mapping[str, int]) -> float:
    return sum(float(UNITS[u]["value"]) * n for u, n in comp.items())


def comp_has_shields(comp: Mapping[str, int]) -> bool:
    flags = {float(UNITS[u]["shield"]) > 0 for u in comp}
    if len(flags) != 1:
        raise ValueError(f"Faction mixes shielded and unshielded units: {comp}")
    return next(iter(flags))


def emulate_group_units(group: Mapping[str, object]) -> List[Dict[str, object]]:
    faction = str(group["faction"])
    units_map: Mapping[str, int] = group["units"]  # type: ignore[assignment]
    expanded: List[str] = []
    for unit_type, count in units_map.items():
        expanded.extend([unit_type] * int(count))
    size = len(expanded)
    side = int(math.ceil(math.sqrt(size)))
    grid: List[List[str | None]] = [[None for _ in range(side)] for _ in range(side)]
    a = b = 0
    for unit_type in expanded:
        grid[b][a] = unit_type
        a += 1
        if a == side:
            a = 0
            b += 1
    row_radii = [max((unit_radius(u) if u else 0.0) for u in row) for row in grid]
    prev_row_height = 0.0
    group_height = 2 * sum(row_radii) + (side - 1) * GROUP_BUFFER
    row_widths = [sum((float(UNITS[u]["size"]) if u else 0.0) for u in row) for row in grid]
    group_width = max(row_widths)
    m = 1.0 if faction == "ALLY" else -1.0
    x0 = float(group["x"]) - m * group_width / 2.0
    y = float(group["y"]) - m * group_height / 2.0
    out: List[Dict[str, object]] = []
    for i, row in enumerate(grid):
        x = x0
        y += m * (prev_row_height + row_radii[i])
        prev_row_height = row_radii[i]
        prev_unit_width = 0.0
        for unit_type in row:
            if unit_type is None:
                continue
            info = UNITS[unit_type]
            radius = unit_radius(unit_type)
            x += m * (prev_unit_width + radius)
            prev_unit_width = radius
            out.append({"unit": unit_type, "faction": faction, "x": x, "y": y, "radius": radius, "plane": info["plane"]})
            x += m * GROUP_BUFFER
        y += m * GROUP_BUFFER
    return out


def actual_units(groups: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for g in groups:
        out.extend(emulate_group_units(g))
    return out


def walkable(terrain: str, x: float, y: float, radius: float = 0.0) -> bool:
    pts = [(x, y), (x + radius, y), (x - radius, y), (x, y + radius), (x, y - radius)]
    for px, py in pts:
        if not (0.0 <= px < MAP_WIDTH and 0.0 <= py < MAP_HEIGHT):
            return False
        ix, iy = int(px), int(py)
        if terrain == "SIMPLE":
            ok = 8 <= iy <= 23
        elif terrain == "NARROW":
            ok = 8 <= iy <= 23 and not (ix in (14, 15) and iy not in (15, 16))
        elif terrain == "OCTAGON":
            if iy < 5 or iy > 26:
                ok = False
            else:
                edge = max(5, 10 - min(iy - 5, 26 - iy))
                right = 31 - edge
                ok = edge <= ix <= right
        else:
            raise KeyError(terrain)
        if not ok:
            return False
    return True


def no_overlap(units: Sequence[Mapping[str, object]]) -> bool:
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if a["plane"] != b["plane"]:
                continue
            dist = math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))
            if dist + 1e-6 < float(a["radius"]) + float(b["radius"]) + 0.01:
                return False
    return True


def min_cross_distance(units: Sequence[Mapping[str, object]]) -> float:
    allies = [u for u in units if u["faction"] == "ALLY"]
    enemies = [u for u in units if u["faction"] == "ENEMY"]
    return min(math.dist((float(a["x"]), float(a["y"])), (float(e["x"]), float(e["y"]))) for a in allies for e in enemies)


def mean_point(units: Sequence[Mapping[str, object]], faction: str) -> Tuple[float, float]:
    chosen = [u for u in units if u["faction"] == faction]
    return (sum(float(u["x"]) for u in chosen) / len(chosen), sum(float(u["y"]) for u in chosen) / len(chosen))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def base_centers(terrain: str, engagement: str, rng: random.Random, variant_number: int) -> Tuple[Tuple[float, float], Tuple[float, float], str]:
    target = ENGAGEMENT_TARGETS[engagement] + rng.uniform(-0.25, 0.25)
    if terrain == "NARROW":
        # Keep both teams on the same side of the narrow wall to avoid turning this
        # toy task into an accidental pathfinding problem. Alternate left/right side.
        side_x = 7.5 if variant_number % 2 == 0 else 23.5
        center_y = 16.0 + rng.uniform(-0.25, 0.25)
        sign = 1 if variant_number % 4 in (0, 1) else -1
        ally = (side_x + rng.uniform(-0.35, 0.35), center_y - sign * target / 2.0)
        enemy = (side_x + rng.uniform(-0.35, 0.35), center_y + sign * target / 2.0)
        orientation = "vertical_same_side"
    else:
        # Mostly left-vs-right, with a small angle so not every battle is perfectly horizontal.
        theta = rng.uniform(-0.35, 0.35)
        if variant_number % 11 == 0:
            # Rare diagonal/vertical-facing layouts for robustness, still generally facing.
            theta += rng.choice([-0.55, 0.55])
        cx = 16.0 + rng.uniform(-0.55, 0.55)
        cy = 16.0 + rng.uniform(-0.45, 0.45)
        dx = math.cos(theta) * target / 2.0
        dy = math.sin(theta) * target / 2.0
        ally = (cx - dx, cy - dy)
        enemy = (cx + dx, cy + dy)
        orientation = "opposing"
    return ally, enemy, orientation


def build_groups(terrain: str, formation: str, engagement: str, rng: random.Random, variant_number: int) -> Tuple[List[Dict[str, object]], str, str]:
    ally_center, enemy_center, orientation = base_centers(terrain, engagement, rng, variant_number)
    ax, ay = ally_center
    ex, ey = enemy_center
    angle = math.atan2(ey - ay, ex - ax)
    # Perpendicular vector for side-by-side split.
    px, py = -math.sin(angle), math.cos(angle)
    # Along-facing vector for stagger.
    fx, fy = math.cos(angle), math.sin(angle)

    if formation == "compact":
        ally_specs = [(ax, ay, {"STALKER": 2})]
    elif formation == "close_split":
        spread = 1.55
        ally_specs = [(ax + px * spread / 2, ay + py * spread / 2, {"STALKER": 1}),
                      (ax - px * spread / 2, ay - py * spread / 2, {"STALKER": 1})]
    elif formation == "wide_split":
        spread = 2.45
        ally_specs = [(ax + px * spread / 2, ay + py * spread / 2, {"STALKER": 1}),
                      (ax - px * spread / 2, ay - py * spread / 2, {"STALKER": 1})]
    else:  # staggered
        ally_specs = [(ax + px * 0.75 - fx * 0.35, ay + py * 0.75 - fy * 0.35, {"STALKER": 1}),
                      (ax - px * 0.75 + fx * 0.35, ay - py * 0.75 + fy * 0.35, {"STALKER": 1})]

    groups: List[Dict[str, object]] = []
    for gx, gy, comp in ally_specs:
        groups.append({"x": round(gx, 3), "y": round(gy, 3), "faction": "ALLY", "units": comp})
    groups.append({"x": round(ex, 3), "y": round(ey, 3), "faction": "ENEMY", "units": {"STALKER": 1}})
    return groups, formation, orientation


def terrain_for(split: str, idx: int) -> str:
    # Bias toward open presets while still covering existing presets.
    if split == "train":
        cycle = ["SIMPLE", "SIMPLE", "OCTAGON", "SIMPLE", "NARROW", "OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON"]
    elif split == "validation":
        cycle = ["SIMPLE", "OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "SIMPLE", "NARROW"]
    else:
        cycle = ["OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "SIMPLE", "SIMPLE", "NARROW"]
    return cycle[idx % len(cycle)]


def formation_for(idx: int) -> str:
    return FORMATIONS[idx % len(FORMATIONS)]


def make_config(split: str, split_idx: int, global_idx: int, engagement: str, rng: random.Random) -> Tuple[Dict[str, object], Dict[str, object]]:
    # Try deterministic variants until one passes all constraints and distance bucket.
    for attempt in range(150):
        terrain = terrain_for(split, global_idx + attempt)
        # Avoid far maps on NARROW, because the central wall/gate can dominate the task.
        if engagement == "far" and terrain == "NARROW":
            terrain = "SIMPLE"
        formation = formation_for(global_idx + attempt)
        groups, formation, orientation = build_groups(terrain, formation, engagement, rng, global_idx + attempt)
        placed = actual_units(groups)
        if not all(walkable(terrain, float(u["x"]), float(u["y"]), float(u["radius"])) for u in placed):
            continue
        if not no_overlap(placed):
            continue
        dist = min_cross_distance(placed)
        lo, hi = ENGAGEMENT_BOUNDS[engagement]
        if not (lo <= dist <= hi):
            continue
        ally_x, ally_y = mean_point(placed, "ALLY")
        name = f"r2_2v1_stalker_{split}_{split_idx:04d}"
        config: Dict[str, object] = {
            "name": name,
            "num_allied_units": 2,
            "num_enemy_units": 1,
            "groups": groups,
            "attack_point": [round(ally_x, 3), round(ally_y, 3)],
            "terrain_preset": terrain,
            "num_unit_types": len(GLOBAL_UNIT_TYPE_IDS),
            "ally_has_shields": True,
            "enemy_has_shields": True,
            "unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        }
        ratio = comp_value({"STALKER": 2}) / comp_value({"STALKER": 1})
        meta: Dict[str, object] = {
            "name": name,
            "split": split,
            "family_id": "stalker_2v1_mirror_only",
            "archetype": "stalker_2v1_mirror",
            "heldout_compositional": False,
            "variant_index": split_idx,
            "engagement_class": engagement,
            "initial_min_cross_distance": round(dist, 4),
            "terrain": terrain,
            "formation": formation,
            "orientation": orientation,
            "num_allies": 2,
            "num_enemies": 1,
            "ally_composition": {"STALKER": 2},
            "enemy_composition": {"STALKER": 1},
            "ally_value": comp_value({"STALKER": 2}),
            "enemy_value": comp_value({"STALKER": 1}),
            "ally_value_ratio": round(ratio, 4),
            "difficulty_proxy": "very_easy_theoretical_2v1",
        }
        return config, meta
    raise RuntimeError(f"Could not place config split={split} idx={split_idx} engagement={engagement}")


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def static_validate_config(config: Mapping[str, object], expected_name: str | None = None) -> List[str]:
    errors: List[str] = []
    required = {"name", "num_allied_units", "num_enemy_units", "groups", "attack_point", "terrain_preset", "num_unit_types", "ally_has_shields", "enemy_has_shields", "unit_type_ids"}
    if set(config) != required:
        errors.append(f"keys mismatch: missing={required-set(config)}, extra={set(config)-required}")
        return errors
    if expected_name and config["name"] != expected_name:
        errors.append("name does not match filename")
    if config["terrain_preset"] not in TERRAINS:
        errors.append("unsupported terrain")
    if config["unit_type_ids"] != GLOBAL_UNIT_TYPE_IDS:
        errors.append("unit_type_ids not equal to global vocabulary")
    if config["num_unit_types"] != len(GLOBAL_UNIT_TYPE_IDS):
        errors.append("num_unit_types mismatch")
    if config["ally_has_shields"] is not True or config["enemy_has_shields"] is not True:
        errors.append("stalker maps must have shields enabled for both factions")
    faction_counts = {"ALLY": Counter(), "ENEMY": Counter()}
    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        errors.append("groups missing")
        return errors
    for gi, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"group {gi} not object")
            continue
        if set(group) != {"x", "y", "faction", "units"}:
            errors.append(f"group {gi} keys invalid")
            continue
        faction = group["faction"]
        if faction not in faction_counts:
            errors.append(f"group {gi} invalid faction")
            continue
        if not (0 <= float(group["x"]) < MAP_WIDTH and 0 <= float(group["y"]) < MAP_HEIGHT):
            errors.append(f"group {gi} center out of bounds")
        units = group["units"]
        if not isinstance(units, dict) or not units:
            errors.append(f"group {gi} units invalid")
            continue
        for unit, count in units.items():
            if unit != "STALKER":
                errors.append(f"non-stalker unit {unit}")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"invalid count {unit}={count}")
            faction_counts[faction][unit] += count
    if faction_counts["ALLY"] != Counter({"STALKER": 2}):
        errors.append("allied composition must be exactly 2 STALKER")
    if faction_counts["ENEMY"] != Counter({"STALKER": 1}):
        errors.append("enemy composition must be exactly 1 STALKER")
    if config["num_allied_units"] != 2 or config["num_enemy_units"] != 1:
        errors.append("unit count fields must be 2 allies vs 1 enemy")
    try:
        placed = actual_units(groups)
        for u in placed:
            if not walkable(str(config["terrain_preset"]), float(u["x"]), float(u["y"]), float(u["radius"])):
                errors.append(f"spawn not walkable: {u}")
        if not no_overlap(placed):
            errors.append("initial unit overlap")
    except Exception as exc:
        errors.append(f"placement emulation failed: {exc}")
    ap = config["attack_point"]
    if not isinstance(ap, list) or len(ap) != 2 or not (0 <= float(ap[0]) < MAP_WIDTH and 0 <= float(ap[1]) < MAP_HEIGHT):
        errors.append("attack_point invalid")
    return errors


def write_dynamic_validator(root: Path) -> None:
    content = r'''#!/usr/bin/env python3
"""Dynamic SMAClite validation for the simple 2v1 stalker map set.

Run from your smac-dreamer repository root after copying this folder in:

    PYTHONPATH=src:external/r2dreamer:external/smaclite \\
      python configs/maps/r2_2v1_stalker/validate_in_smaclite.py \\
      --root configs/maps/r2_2v1_stalker --episodes 10 --max-steps 160

The script checks that each map instantiates and runs a deterministic scripted
focus-fire policy. This is a smoke test, not a replacement for trained-policy evaluation.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from smaclite.env.util.direction import Direction


def choose_actions(env):
    avail = env.get_avail_actions()
    actions = []
    for i in range(env.n_agents):
        valid = np.flatnonzero(avail[i])
        if i not in env.agents:
            actions.append(0)
            continue
        unit = env.agents[i]
        attack_choices = [a for a in valid if a >= 6]
        if attack_choices:
            # 2v1 stalker: attack target 0 whenever legal.
            actions.append(int(min(attack_choices)))
            continue
        if env.enemies:
            enemy = next(iter(env.enemies.values()))
            prefs = []
            for a in [2, 3, 4, 5]:
                if a not in valid:
                    continue
                dest = unit.pos + Direction(a - 2).dx_dy * 2
                prefs.append((float(np.linalg.norm(dest - enemy.pos)), a))
            if prefs:
                actions.append(int(min(prefs)[1]))
                continue
        actions.append(1 if 1 in valid else int(valid[0]))
    return actions


def run_one(path, seed, max_steps):
    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=str(path), seed=seed)
    obs, info = env.reset(seed=seed)
    assert np.isfinite(np.asarray(obs)).all(), path
    assert np.isfinite(env.get_state()).all(), path
    total = 0.0
    won = False
    done = False
    truncated = False
    for t in range(max_steps):
        obs, reward, done, truncated, info = env.step(choose_actions(env))
        assert np.isfinite(np.asarray(obs)).all(), path
        assert np.isfinite(float(reward)), path
        total += float(reward)
        if done or truncated:
            won = bool(info.get('battle_won', False))
            break
    env.close()
    return won, (not (done or truncated)), total, t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=160)
    args = p.parse_args()
    maps = sorted((args.root / 'configs').glob('*/*.json'))
    rows = []
    for idx, path in enumerate(maps, 1):
        wins = timeouts = 0
        returns = []
        lengths = []
        for ep in range(args.episodes):
            w, to, r, l = run_one(path, 1000 + ep, args.max_steps)
            wins += int(w)
            timeouts += int(to)
            returns.append(r)
            lengths.append(l)
        rows.append({
            'path': str(path),
            'win_rate': wins / args.episodes,
            'timeout_rate': timeouts / args.episodes,
            'mean_return': sum(returns) / len(returns),
            'mean_length': sum(lengths) / len(lengths),
        })
        print(f'[{idx:03d}/{len(maps)}] {path.name}: win={rows[-1]["win_rate"]:.2f} timeout={rows[-1]["timeout_rate"]:.2f}')
    out = args.root / 'dynamic_validation.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
'''
    (root / "validate_in_smaclite.py").write_text(content, encoding="utf-8")


def write_readme(root: Path, summary: Mapping[str, object]) -> None:
    text = f"""# R2-Dreamer × SMAClite simple 2v1 stalker benchmark

This directory contains **240 deterministic custom SMAClite map JSON files** for a deliberately simple high-winrate target:

- Allied team: exactly `2 × STALKER`
- Enemy team: exactly `1 × STALKER`
- Unit type: stalker only, on both teams
- Difficulty target: theoretically easy 2v1 mirror fight
- Terrain source: existing SMAClite presets only — `SIMPLE`, `NARROW`, `OCTAGON`

## Split

- `configs/train`: 160 maps
- `configs/validation`: 40 maps
- `configs/blind_iid`: 40 maps

There is intentionally no `blind_compositional` split because the composition is fixed by design. The files contain only fields accepted by SMAClite's custom map parser; research metadata is stored separately in `manifest.jsonl` and `manifest.csv`.

## Design choices

1. **Single scenario family:** all maps are `2 STALKER` allies versus `1 STALKER` enemy.
2. **High-winnability bias:** the static allied combat-value ratio is exactly 2.0× in every map.
3. **Existing terrain presets only:** no custom terrain arrays are generated.
4. **Mostly facing layouts:** most maps are left-vs-right or same-side vertical encounters, with small angular/positional variation.
5. **Controlled variation:** spawn positions, ally grouping, formation, engagement distance and terrain are randomized deterministically.
6. **No accidental pathfinding traps:** `NARROW` maps keep both teams on the same side of the wall; far maps avoid `NARROW`.
7. **Global type vocabulary:** every map keeps the same nine-entry `unit_type_ids` mapping used in the previous benchmark format.
8. **Reproducibility:** fixed seed `{SEED}`; rerun `generate_simple_2v1_stalker_240.py` to recreate the dataset.

## Engagement distribution

- Train: 40 immediate, 72 near, 40 medium, 8 far
- Validation: 10 immediate, 18 near, 10 medium, 2 far
- Blind-IID: 10 immediate, 18 near, 10 medium, 2 far

## Files

- `generate_simple_2v1_stalker_240.py`: self-contained deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted focus-fire policy
- `manifest.jsonl` / `manifest.csv`: per-map split, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: the fixed 2v1 stalker family definition
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: config-file content checksums

## Static validation result

- Files: {summary['total_configs']}
- Errors: {summary['validation_errors']}
- Unique semantic configs: {summary['unique_semantic_configs']}
- Seed: {SEED}

## Required dynamic validation

Static checks cannot prove trained-policy win rate. After copying this folder into your repo, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \\
python configs/maps/r2_2v1_stalker/validate_in_smaclite.py \\
  --root configs/maps/r2_2v1_stalker \\
  --episodes 10 \\
  --max-steps 160
```

Use `dynamic_validation.csv` to identify any layouts where even the scripted focus-fire baseline times out or loses. For the Dreamer run, select checkpoints using validation/blind-IID win rate and original SMAClite return.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate(out_root: Path, make_zip: bool = True) -> Dict[str, object]:
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "configs").mkdir(parents=True)
    rng = random.Random(SEED)
    records: List[Dict[str, object]] = []
    split_files: Dict[str, List[str]] = defaultdict(list)
    configs_by_name: Dict[str, Dict[str, object]] = {}
    global_idx = 1
    for split, spec in SPLIT_TARGETS.items():
        d = out_root / "configs" / split
        d.mkdir(parents=True, exist_ok=True)
        engagements: List[str] = []
        for cls, n in spec["engagement"].items():  # type: ignore[index,union-attr]
            engagements.extend([cls] * int(n))
        rng.shuffle(engagements)
        assert len(engagements) == spec["total"]
        for split_idx, engagement in enumerate(engagements, 1):
            cfg, meta = make_config(split, split_idx, global_idx, engagement, rng)
            configs_by_name[str(cfg["name"])] = cfg
            records.append(meta)
            rel = f"configs/{split}/{cfg['name']}.json"
            split_files[split].append(rel)
            (out_root / rel).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            global_idx += 1

    # Static validation and duplicate detection.
    validation_errors: Dict[str, List[str]] = {}
    content_hashes: Dict[str, str] = {}
    semantic_hashes: Counter = Counter()
    for split, paths in split_files.items():
        for rel in paths:
            p = out_root / rel
            cfg = json.loads(p.read_text())
            errs = static_validate_config(cfg, p.stem)
            if errs:
                validation_errors[rel] = errs
            content_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            semantic = dict(cfg)
            semantic.pop("name")
            semantic_hashes[sha256_json(semantic)] += 1
    duplicate_semantics = [h for h, n in semantic_hashes.items() if n > 1]
    if duplicate_semantics:
        raise ValueError(f"Duplicate semantic configs: {len(duplicate_semantics)}")
    if validation_errors:
        raise ValueError(json.dumps(validation_errors, indent=2))

    # Manifests.
    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec0 in records:
            rec = dict(rec0)
            rel = next(r for r in split_files[rec["split"]] if Path(r).stem == rec["name"])
            rec["path"] = rel
            rec["sha256"] = content_hashes[rel]
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    csv_fields = [
        "name", "path", "split", "family_id", "archetype", "heldout_compositional", "variant_index",
        "engagement_class", "initial_min_cross_distance", "terrain", "formation", "orientation",
        "num_allies", "num_enemies", "ally_value", "enemy_value", "ally_value_ratio", "difficulty_proxy",
        "ally_composition", "enemy_composition", "sha256",
    ]
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for rec0 in records:
            rec = dict(rec0)
            rel = next(r for r in split_files[rec["split"]] if Path(r).stem == rec["name"])
            rec["path"] = rel
            rec["sha256"] = content_hashes[rel]
            rec["ally_composition"] = json.dumps(rec["ally_composition"], sort_keys=True)
            rec["enemy_composition"] = json.dumps(rec["enemy_composition"], sort_keys=True)
            w.writerow({k: rec[k] for k in csv_fields})

    (out_root / "split_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "splits": split_files,
    }, indent=2) + "\n", encoding="utf-8")

    family_catalog = {
        "families": [{
            "family_id": "stalker_2v1_mirror_only",
            "archetype": "stalker_2v1_mirror",
            "ally": {"STALKER": 2},
            "enemy": {"STALKER": 1},
            "heldout_compositional": False,
            "static_value_ratio": 2.0,
        }]
    }
    (out_root / "family_catalog.json").write_text(json.dumps(family_catalog, indent=2) + "\n", encoding="utf-8")

    split_counts = Counter(str(r["split"]) for r in records)
    engagement_counts = {split: Counter(str(r["engagement_class"]) for r in records if r["split"] == split) for split in split_counts}
    terrain_counts = {split: Counter(str(r["terrain"]) for r in records if r["split"] == split) for split in split_counts}
    formation_counts = {split: Counter(str(r["formation"]) for r in records if r["split"] == split) for split in split_counts}
    orientation_counts = {split: Counter(str(r["orientation"]) for r in records if r["split"] == split) for split in split_counts}
    distances = [float(r["initial_min_cross_distance"]) for r in records]
    summary = {
        "seed": SEED,
        "total_configs": len(records),
        "validation_errors": len(validation_errors),
        "unique_semantic_configs": len(semantic_hashes),
        "split_counts": dict(split_counts),
        "engagement_counts": {k: dict(v) for k, v in engagement_counts.items()},
        "terrain_counts": {k: dict(v) for k, v in terrain_counts.items()},
        "formation_counts": {k: dict(v) for k, v in formation_counts.items()},
        "orientation_counts": {k: dict(v) for k, v in orientation_counts.items()},
        "composition": {"ally": {"STALKER": 2}, "enemy": {"STALKER": 1}},
        "ally_value_ratio": {"min": 2.0, "max": 2.0, "mean": 2.0, "median": 2.0},
        "initial_min_cross_distance": {
            "min": min(distances), "max": max(distances), "mean": statistics.mean(distances), "median": statistics.median(distances),
        },
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "static_checks": [
            "exact JSON schema", "name/filename agreement", "only STALKER units", "exact 2v1 composition",
            "unit-count field agreement", "global unit-type vocabulary", "shield flags enabled for both factions",
            "group placement emulation", "spawn bounds", "terrain walkability", "same-plane overlap rejection",
            "attack-point bounds", "engagement bucket", "duplicate semantic map rejection",
        ],
        "dynamic_validation_required": True,
    }
    (out_root / "validation_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_root / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, digest in sorted(content_hashes.items()):
            f.write(f"{digest}  {rel}\n")

    # Copy generator script under the dataset's expected name.
    shutil.copy2(Path(__file__).resolve(), out_root / "generate_simple_2v1_stalker_240.py")
    write_dynamic_validator(out_root)
    write_readme(out_root, summary)

    if make_zip:
        zip_path = out_root.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(out_root.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(out_root.name) / p.relative_to(out_root))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(DATASET_NAME), help="Output directory for the dataset")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a .zip archive beside the output directory")
    args = parser.parse_args()
    summary = generate(args.out, make_zip=not args.no_zip)
    print(json.dumps(summary, indent=2))
