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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

SEED = 21072026
MAP_WIDTH = 32
MAP_HEIGHT = 32
GROUP_BUFFER = 0.05
DATASET_NAME = "r2_smaclite_3v2_basic_mixed_480_configs"

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

# Basic combat-only pool. Excluded special units in generated groups:
# BANELING (burst/suicide), MEDIVAC (healer), SPINE_CRAWLER (static), COLOSSUS (large/splash-like).
ALLOWED_UNITS = {"STALKER", "ZEALOT", "MARINE", "MARAUDER", "ZERGLING"}
SPECIAL_EXCLUDED = {"BANELING", "MEDIVAC", "SPINE_CRAWLER", "COLOSSUS"}

UNITS = {
    "BANELING": {"hp": 30, "shield": 0, "damage": 80, "attacks": 1, "cooldown": 1.0, "speed": 3.15, "attack_range": 0.25, "size": 0.75, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 50},
    "COLOSSUS": {"hp": 200, "shield": 150, "damage": 10, "attacks": 2, "cooldown": 1.07, "speed": 3.15, "attack_range": 7.0, "size": 2.0, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 300},
    "MARAUDER": {"hp": 125, "shield": 0, "damage": 10, "attacks": 1, "cooldown": 1.07, "speed": 3.15, "attack_range": 6.0, "size": 1.125, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 100},
    "MARINE": {"hp": 45, "shield": 0, "damage": 6, "attacks": 1, "cooldown": 0.61, "speed": 3.15, "attack_range": 5.0, "size": 0.75, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND", "AIR"), "value": 50},
    "MEDIVAC": {"hp": 150, "shield": 0, "damage": 0, "attacks": 1, "cooldown": 0.0, "speed": 3.5, "attack_range": 4.0, "size": 1.5, "plane": "AIR", "combat_type": "HEALING", "valid_targets": ("GROUND",), "value": 125},
    "SPINE_CRAWLER": {"hp": 300, "shield": 0, "damage": 25, "attacks": 1, "cooldown": 1.32, "speed": 0.0, "attack_range": 7.0, "size": 1.0, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 175},
    "STALKER": {"hp": 80, "shield": 80, "damage": 13, "attacks": 1, "cooldown": 1.34, "speed": 4.13, "attack_range": 6.0, "size": 1.25, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND", "AIR"), "value": 125},
    "ZEALOT": {"hp": 100, "shield": 50, "damage": 8, "attacks": 2, "cooldown": 0.86, "speed": 3.15, "attack_range": 0.25, "size": 1.0, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 100},
    "ZERGLING": {"hp": 35, "shield": 0, "damage": 5, "attacks": 1, "cooldown": 0.497, "speed": 4.13, "attack_range": 0.25, "size": 0.75, "plane": "GROUND", "combat_type": "DAMAGE", "valid_targets": ("GROUND",), "value": 25},
}

TERRAINS = ("SIMPLE", "NARROW", "OCTAGON")
FORMATIONS = ("compact", "close_split", "wide_split", "staggered", "type_split")
ENGAGEMENT_TARGETS = {
    "immediate": 5.2,
    "near": 8.0,
    "medium": 11.2,
    "far": 14.2,
}
ENGAGEMENT_BOUNDS = {
    "immediate": (4.35, 6.35),
    "near": (6.9, 9.4),
    "medium": (9.8, 12.7),
    "far": (13.0, 16.1),
}
SPLIT_TARGETS = {
    "train": {"total": 320, "variants_per_family": 20, "engagement": {"immediate": 96, "near": 128, "medium": 80, "far": 16}},
    "validation": {"total": 80, "variants_per_family": 5, "engagement": {"immediate": 24, "near": 32, "medium": 20, "far": 4}},
    "blind_iid": {"total": 80, "variants_per_family": 5, "engagement": {"immediate": 24, "near": 32, "medium": 20, "far": 4}},
}


def C(**kwargs: int) -> Dict[str, int]:
    return {k: int(v) for k, v in kwargs.items() if v > 0}

@dataclass(frozen=True)
class Family:
    family_id: str
    archetype: str
    ally: Mapping[str, int]
    enemy: Mapping[str, int]


def build_families() -> List[Family]:
    # 16 basic-combat 3v2 families. Each faction is internally shield-homogeneous.
    return [
        Family("protoss_stalker_mirror", "protoss_basic", C(STALKER=3), C(STALKER=2)),
        Family("protoss_zealot_mirror", "protoss_basic", C(ZEALOT=3), C(ZEALOT=2)),
        Family("protoss_mixed_equal", "protoss_basic", C(STALKER=2, ZEALOT=1), C(STALKER=1, ZEALOT=1)),
        Family("protoss_zealot_heavy", "protoss_basic", C(STALKER=1, ZEALOT=2), C(STALKER=1, ZEALOT=1)),
        Family("protoss_ranged_advantage", "protoss_basic", C(STALKER=2, ZEALOT=1), C(ZEALOT=2)),
        Family("protoss_melee_vs_ranged", "protoss_basic", C(STALKER=1, ZEALOT=2), C(STALKER=2)),

        Family("terran_marine_mirror", "terran_basic", C(MARINE=3), C(MARINE=2)),
        Family("terran_marauder_mirror", "terran_basic", C(MARAUDER=3), C(MARAUDER=2)),
        Family("terran_mixed_light", "terran_basic", C(MARINE=2, MARAUDER=1), C(MARINE=1, MARAUDER=1)),
        Family("terran_mixed_heavy", "terran_basic", C(MARINE=1, MARAUDER=2), C(MARINE=1, MARAUDER=1)),
        Family("terran_marine_advantage", "terran_basic", C(MARINE=2, MARAUDER=1), C(MARINE=2)),
        Family("terran_marauder_advantage", "terran_basic", C(MARINE=1, MARAUDER=2), C(MARAUDER=2)),

        Family("swarm_zergling_mirror", "swarm_basic", C(ZERGLING=3), C(ZERGLING=2)),
        Family("bio_zergling_light", "bio_swarm_basic", C(MARINE=2, ZERGLING=1), C(MARINE=1, ZERGLING=1)),
        Family("bio_zergling_melee", "bio_swarm_basic", C(MARINE=1, ZERGLING=2), C(ZERGLING=2)),
        Family("bio_marauder_zergling", "bio_swarm_basic", C(MARAUDER=1, ZERGLING=2), C(MARAUDER=1, ZERGLING=1)),
    ]

FAMILIES = build_families()


def unit_radius(unit: str) -> float:
    return float(UNITS[unit]["size"]) / 2.0


def comp_count(comp: Mapping[str, int]) -> int:
    return sum(comp.values())


def comp_value(comp: Mapping[str, int]) -> float:
    return sum(float(UNITS[u]["value"]) * n for u, n in comp.items())


def comp_has_shields(comp: Mapping[str, int]) -> bool:
    flags = {float(UNITS[u]["shield"]) > 0 for u in comp}
    if len(flags) != 1:
        raise ValueError(f"Faction mixes shielded and unshielded units: {dict(comp)}")
    return next(iter(flags))


def difficulty_proxy(ratio: float) -> str:
    if ratio >= 1.55:
        return "easy"
    if ratio >= 1.35:
        return "moderate"
    return "hard_winnable"


def validate_family(f: Family) -> None:
    if comp_count(f.ally) != 3 or comp_count(f.enemy) != 2:
        raise ValueError(f"{f.family_id}: must be exactly 3v2: {dict(f.ally)} vs {dict(f.enemy)}")
    for comp in (f.ally, f.enemy):
        if set(comp) - ALLOWED_UNITS:
            raise ValueError(f"{f.family_id}: contains non-basic units {set(comp) - ALLOWED_UNITS}")
        if set(comp) & SPECIAL_EXCLUDED:
            raise ValueError(f"{f.family_id}: contains excluded special unit")
        comp_has_shields(comp)
    ratio = comp_value(f.ally) / comp_value(f.enemy)
    if not (1.18 <= ratio <= 2.1):
        raise ValueError(f"{f.family_id}: ally/enemy value ratio outside safe range: {ratio:.3f}")

for _f in FAMILIES:
    validate_family(_f)
assert len(FAMILIES) == 16


def split_comp(comp: Mapping[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Split a composition into two nonempty groups, preserving insertion order."""
    items: List[str] = []
    for unit, count in comp.items():
        items.extend([unit] * int(count))
    # Put first ceil half in group A, rest in group B.
    cut = max(1, math.ceil(len(items) / 2))
    a = Counter(items[:cut])
    b = Counter(items[cut:])
    return dict(a), dict(b)


def type_groups(comp: Mapping[str, int]) -> List[Dict[str, int]]:
    return [{u: int(n)} for u, n in comp.items() if int(n) > 0]


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
            if dist + 1e-6 < float(a["radius"]) + float(b["radius"]) + 0.015:
                return False
    return True


def min_cross_distance(units: Sequence[Mapping[str, object]]) -> float:
    allies = [u for u in units if u["faction"] == "ALLY"]
    enemies = [u for u in units if u["faction"] == "ENEMY"]
    return min(math.dist((float(a["x"]), float(a["y"])), (float(e["x"]), float(e["y"]))) for a in allies for e in enemies)


def mean_point(units: Sequence[Mapping[str, object]], faction: str) -> Tuple[float, float]:
    chosen = [u for u in units if u["faction"] == faction]
    return (sum(float(u["x"]) for u in chosen) / len(chosen), sum(float(u["y"]) for u in chosen) / len(chosen))


def base_centers(terrain: str, engagement: str, rng: random.Random, variant_number: int) -> Tuple[Tuple[float, float], Tuple[float, float], str]:
    target = ENGAGEMENT_TARGETS[engagement] + rng.uniform(-0.25, 0.25)
    if terrain == "NARROW":
        # Keep both teams on the same side of the wall/gate to avoid accidental pathfinding tasks.
        side_x = 7.8 if variant_number % 2 == 0 else 23.2
        center_y = 16.0 + rng.uniform(-0.22, 0.22)
        sign = 1 if variant_number % 4 in (0, 1) else -1
        ally = (side_x + rng.uniform(-0.32, 0.32), center_y - sign * target / 2.0)
        enemy = (side_x + rng.uniform(-0.32, 0.32), center_y + sign * target / 2.0)
        orientation = "vertical_same_side"
    else:
        theta = rng.uniform(-0.35, 0.35)
        if variant_number % 13 == 0:
            # Rare more-diagonal facing layouts while still generally opposing.
            theta += rng.choice([-0.55, 0.55])
        cx = 16.0 + rng.uniform(-0.55, 0.55)
        cy = 16.0 + rng.uniform(-0.45, 0.45)
        dx = math.cos(theta) * target / 2.0
        dy = math.sin(theta) * target / 2.0
        ally = (cx - dx, cy - dy)
        enemy = (cx + dx, cy + dy)
        orientation = "opposing"
    return ally, enemy, orientation


def team_group_specs(center: Tuple[float, float], comp: Mapping[str, int], formation: str, faction: str,
                     angle: float, terrain: str) -> List[Tuple[float, float, Dict[str, int]]]:
    cx, cy = center
    px, py = -math.sin(angle), math.cos(angle)
    fx, fy = math.cos(angle), math.sin(angle)
    side_mul = 0.75 if terrain == "NARROW" else 1.0
    # For enemies, flip the facing vector so their stagger points toward the allied side.
    fm = 1.0 if faction == "ALLY" else -1.0

    if formation == "compact":
        return [(cx, cy, dict(comp))]

    if formation == "type_split" and len(comp) > 1:
        groups = type_groups(comp)
        if len(groups) == 2:
            spread = 1.7 * side_mul
            return [(cx + px * spread / 2, cy + py * spread / 2, groups[0]),
                    (cx - px * spread / 2, cy - py * spread / 2, groups[1])]
        # Fallback for >2 unit types, not used by current families.
        spread = 1.6 * side_mul
        offsets = [i - (len(groups) - 1) / 2 for i in range(len(groups))]
        return [(cx + px * spread * off, cy + py * spread * off, g) for off, g in zip(offsets, groups)]

    a, b = split_comp(comp)
    if formation == "close_split":
        spread = 1.55 * side_mul
        return [(cx + px * spread / 2, cy + py * spread / 2, a),
                (cx - px * spread / 2, cy - py * spread / 2, b)]
    if formation == "wide_split":
        spread = 2.35 * side_mul
        return [(cx + px * spread / 2, cy + py * spread / 2, a),
                (cx - px * spread / 2, cy - py * spread / 2, b)]
    # staggered
    return [(cx + px * 0.75 * side_mul + fm * fx * 0.35, cy + py * 0.75 * side_mul + fm * fy * 0.35, a),
            (cx - px * 0.75 * side_mul - fm * fx * 0.35, cy - py * 0.75 * side_mul - fm * fy * 0.35, b)]


def build_groups(family: Family, terrain: str, formation: str, engagement: str,
                 rng: random.Random, variant_number: int) -> Tuple[List[Dict[str, object]], str, str]:
    ally_center, enemy_center, orientation = base_centers(terrain, engagement, rng, variant_number)
    ax, ay = ally_center
    ex, ey = enemy_center
    angle = math.atan2(ey - ay, ex - ax)

    ally_specs = team_group_specs(ally_center, family.ally, formation, "ALLY", angle, terrain)
    # Use a related but not always identical enemy grouping pattern for extra variation.
    enemy_form = formation
    if formation == "wide_split" and variant_number % 3 == 0:
        enemy_form = "close_split"
    elif formation == "type_split" and len(family.enemy) == 1:
        enemy_form = "close_split"
    enemy_specs = team_group_specs(enemy_center, family.enemy, enemy_form, "ENEMY", angle, terrain)

    groups: List[Dict[str, object]] = []
    for gx, gy, comp in ally_specs:
        groups.append({"x": round(gx, 3), "y": round(gy, 3), "faction": "ALLY", "units": comp})
    for gx, gy, comp in enemy_specs:
        groups.append({"x": round(gx, 3), "y": round(gy, 3), "faction": "ENEMY", "units": comp})
    return groups, formation, orientation


def terrain_for(split: str, idx: int) -> str:
    if split == "train":
        cycle = ["SIMPLE", "SIMPLE", "OCTAGON", "SIMPLE", "NARROW", "OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON"]
    elif split == "validation":
        cycle = ["SIMPLE", "OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "SIMPLE", "NARROW"]
    else:
        cycle = ["OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "SIMPLE", "SIMPLE", "NARROW"]
    return cycle[idx % len(cycle)]


def formation_for(idx: int) -> str:
    return FORMATIONS[idx % len(FORMATIONS)]


def make_config(split: str, split_idx: int, global_idx: int, family: Family, engagement: str, rng: random.Random) -> Tuple[Dict[str, object], Dict[str, object]]:
    for attempt in range(250):
        terrain = terrain_for(split, global_idx + attempt)
        if engagement == "far" and terrain == "NARROW":
            terrain = "SIMPLE"
        formation = formation_for(global_idx + attempt)
        groups, formation, orientation = build_groups(family, terrain, formation, engagement, rng, global_idx + attempt)
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
        ratio = comp_value(family.ally) / comp_value(family.enemy)
        name = f"r2_3v2_basic_{split}_{split_idx:04d}_{family.family_id}"
        config: Dict[str, object] = {
            "name": name,
            "num_allied_units": 3,
            "num_enemy_units": 2,
            "groups": groups,
            "attack_point": [round(ally_x, 3), round(ally_y, 3)],
            "terrain_preset": terrain,
            "num_unit_types": len(GLOBAL_UNIT_TYPE_IDS),
            "ally_has_shields": comp_has_shields(family.ally),
            "enemy_has_shields": comp_has_shields(family.enemy),
            "unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        }
        meta: Dict[str, object] = {
            "name": name,
            "split": split,
            "family_id": family.family_id,
            "archetype": family.archetype,
            "heldout_compositional": False,
            "variant_index": split_idx,
            "engagement_class": engagement,
            "initial_min_cross_distance": round(dist, 4),
            "terrain": terrain,
            "formation": formation,
            "orientation": orientation,
            "num_allies": 3,
            "num_enemies": 2,
            "ally_composition": dict(family.ally),
            "enemy_composition": dict(family.enemy),
            "ally_value": comp_value(family.ally),
            "enemy_value": comp_value(family.enemy),
            "ally_value_ratio": round(ratio, 4),
            "difficulty_proxy": difficulty_proxy(ratio),
        }
        return config, meta
    raise RuntimeError(f"Could not place config split={split} idx={split_idx} family={family.family_id} engagement={engagement}")


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
            if unit not in ALLOWED_UNITS:
                errors.append(f"disallowed or special unit {unit}")
            if unit in SPECIAL_EXCLUDED:
                errors.append(f"excluded special unit {unit}")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"invalid count {unit}={count}")
            faction_counts[faction][unit] += count
    if sum(faction_counts["ALLY"].values()) != 3 or sum(faction_counts["ENEMY"].values()) != 2:
        errors.append("composition must be exactly 3 allies vs 2 enemies")
    if config["num_allied_units"] != 3 or config["num_enemy_units"] != 2:
        errors.append("unit count fields must be 3 allies vs 2 enemies")
    try:
        ally_shields = comp_has_shields(faction_counts["ALLY"])
        enemy_shields = comp_has_shields(faction_counts["ENEMY"])
        if bool(config["ally_has_shields"]) != ally_shields:
            errors.append("ally_has_shields flag mismatch")
        if bool(config["enemy_has_shields"]) != enemy_shields:
            errors.append("enemy_has_shields flag mismatch")
        ratio = comp_value(faction_counts["ALLY"]) / comp_value(faction_counts["ENEMY"])
        if ratio < 1.18:
            errors.append(f"ally value ratio too low: {ratio:.3f}")
    except Exception as exc:
        errors.append(f"composition shield/value validation failed: {exc}")
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
"""Dynamic SMAClite validation for the 3v2 basic mixed-unit map set.

Run from your smac-dreamer repository root after copying this folder in:

    PYTHONPATH=src:external/r2dreamer:external/smaclite \
      python configs/maps/r2_3v2_basic_mixed/validate_in_smaclite.py \
      --root configs/maps/r2_3v2_basic_mixed --episodes 5 --max-steps 180

The script checks that each map instantiates and runs a deterministic scripted
focus-fire policy. This is a smoke test, not a trained-policy benchmark.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from smaclite.env.util.direction import Direction


def choose_actions(env):
    avail = env.get_avail_actions()
    actions = []
    # Prefer focusing the lowest HP+shield attackable enemy. Otherwise move toward nearest enemy.
    enemy_order = sorted(env.enemies.values(), key=lambda e: (getattr(e, 'hp', 0) + getattr(e, 'shield', 0), e.id_in_faction))
    for i in range(env.n_agents):
        valid = np.flatnonzero(avail[i])
        if i not in env.agents:
            actions.append(0)
            continue
        unit = env.agents[i]
        attack_choices = [a for a in valid if a >= 6]
        if attack_choices:
            chosen = None
            for enemy in enemy_order:
                a = 6 + enemy.id_in_faction
                if a in attack_choices:
                    chosen = a
                    break
            actions.append(int(chosen if chosen is not None else min(attack_choices)))
            continue
        if env.enemies:
            nearest = min(env.enemies.values(), key=lambda e: float(np.linalg.norm(unit.pos - e.pos)))
            prefs = []
            for a in [2, 3, 4, 5]:
                if a not in valid:
                    continue
                dest = unit.pos + Direction(a - 2).dx_dy * 2
                prefs.append((float(np.linalg.norm(dest - nearest.pos)), a))
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
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=180)
    args = p.parse_args()
    maps = sorted((args.root / 'configs').glob('*/*.json'))
    rows = []
    for idx, path in enumerate(maps, 1):
        wins = timeouts = 0
        returns = []
        lengths = []
        for ep in range(args.episodes):
            w, to, r, l = run_one(path, 2000 + ep, args.max_steps)
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
    if rows:
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
    text = f"""# R2-Dreamer × SMAClite 3v2 basic mixed-unit benchmark

This directory contains **480 deterministic custom SMAClite map JSON files** for the next curriculum step after the 2v1 stalker-only benchmark.

## Scenario rule

Every map is exactly:

- `3` allied units
- `2` enemy units
- basic combat units only: `STALKER`, `ZEALOT`, `MARINE`, `MARAUDER`, `ZERGLING`
- no special/support/static/burst units in generated groups: `BANELING`, `MEDIVAC`, `SPINE_CRAWLER`, `COLOSSUS`
- existing SMAClite terrain presets only: `SIMPLE`, `NARROW`, `OCTAGON`
- theoretically winnable by static combat-value proxy; ally/enemy value ratio is always >= 1.20×

## Split

- `configs/train`: 320 maps
- `configs/validation`: 80 maps
- `configs/blind_iid`: 80 maps

There is intentionally no `blind_compositional` split in this first mixed-unit curriculum step. The validation and blind-IID splits use the same 16 composition families, but with unseen layouts, terrain/formation choices and spawn jitter.

## Composition families

There are 16 basic-combat families. Each family has 20 train variants, 5 validation variants and 5 blind-IID variants.

Examples:

- `3 STALKER` vs `2 STALKER`
- `3 ZEALOT` vs `2 ZEALOT`
- `2 STALKER + 1 ZEALOT` vs `1 STALKER + 1 ZEALOT`
- `3 MARINE` vs `2 MARINE`
- `2 MARINE + 1 MARAUDER` vs `1 MARINE + 1 MARAUDER`
- `3 ZERGLING` vs `2 ZERGLING`
- `2 MARINE + 1 ZERGLING` vs `1 MARINE + 1 ZERGLING`

Each faction is internally shield-homogeneous, so the map-level `ally_has_shields` and `enemy_has_shields` flags remain correct for SMAClite observations.

## Engagement distribution

- Train: 96 immediate, 128 near, 80 medium, 16 far
- Validation: 24 immediate, 32 near, 20 medium, 4 far
- Blind-IID: 24 immediate, 32 near, 20 medium, 4 far

This keeps the task mostly combat-rich while introducing a small amount of longer approach behaviour.

## Randomized dimensions

- spawn positions
- ally and enemy grouping
- formation: `compact`, `close_split`, `wide_split`, `staggered`, `type_split`
- terrain preset
- engagement distance
- facing orientation
- small deterministic jitter

`NARROW` maps keep teams on the same side of the wall/gate and far maps avoid `NARROW`, so the dataset does not accidentally become a pathfinding benchmark.

## Files

- `generate_3v2_basic_mixed_480.py`: deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted focus-fire policy
- `manifest.jsonl` / `manifest.csv`: per-map split, family, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: the 16 composition-family definitions
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: config-file content checksums

## Static validation result

- Files: {summary['total_configs']}
- Errors: {summary['validation_errors']}
- Unique semantic configs: {summary['unique_semantic_configs']}
- Seed: {SEED}

## Dynamic validation

Static checks cannot prove trained-policy win rate. After copying this folder into your repo, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_3v2_basic_mixed/validate_in_smaclite.py \
  --root configs/maps/r2_3v2_basic_mixed \
  --episodes 5 \
  --max-steps 180
```

Use `dynamic_validation.csv` to identify layouts where even the scripted focus-fire baseline has issues. For Dreamer, select checkpoints using validation/blind-IID win rate and original SMAClite return.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate(out_root: Path, make_zip: bool = True) -> Dict[str, object]:
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "configs").mkdir(parents=True)
    rng = random.Random(SEED)

    records: List[Dict[str, object]] = []
    split_files: Dict[str, List[str]] = defaultdict(list)
    global_idx = 1

    for split, spec in SPLIT_TARGETS.items():
        d = out_root / "configs" / split
        d.mkdir(parents=True, exist_ok=True)
        engagements: List[str] = []
        for cls, n in spec["engagement"].items():  # type: ignore[index,union-attr]
            engagements.extend([cls] * int(n))
        rng.shuffle(engagements)
        assert len(engagements) == spec["total"]
        family_plan: List[Family] = []
        variants_per_family = int(spec["variants_per_family"])
        for fam in FAMILIES:
            family_plan.extend([fam] * variants_per_family)
        assert len(family_plan) == len(engagements)
        rng.shuffle(family_plan)
        for split_idx, (family, engagement) in enumerate(zip(family_plan, engagements), 1):
            cfg, meta = make_config(split, split_idx, global_idx, family, engagement, rng)
            records.append(meta)
            rel = f"configs/{split}/{cfg['name']}.json"
            split_files[split].append(rel)
            (out_root / rel).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            global_idx += 1

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

    family_catalog = {"families": [
        {
            "family_id": f.family_id,
            "archetype": f.archetype,
            "ally": dict(f.ally),
            "enemy": dict(f.enemy),
            "heldout_compositional": False,
            "ally_value": comp_value(f.ally),
            "enemy_value": comp_value(f.enemy),
            "static_value_ratio": round(comp_value(f.ally) / comp_value(f.enemy), 4),
            "ally_has_shields": comp_has_shields(f.ally),
            "enemy_has_shields": comp_has_shields(f.enemy),
        }
        for f in FAMILIES
    ]}
    (out_root / "family_catalog.json").write_text(json.dumps(family_catalog, indent=2) + "\n", encoding="utf-8")

    split_counts = Counter(str(r["split"]) for r in records)
    engagement_counts = {split: Counter(str(r["engagement_class"]) for r in records if r["split"] == split) for split in split_counts}
    terrain_counts = {split: Counter(str(r["terrain"]) for r in records if r["split"] == split) for split in split_counts}
    formation_counts = {split: Counter(str(r["formation"]) for r in records if r["split"] == split) for split in split_counts}
    orientation_counts = {split: Counter(str(r["orientation"]) for r in records if r["split"] == split) for split in split_counts}
    family_counts = {split: Counter(str(r["family_id"]) for r in records if r["split"] == split) for split in split_counts}
    difficulty_counts = {split: Counter(str(r["difficulty_proxy"]) for r in records if r["split"] == split) for split in split_counts}
    ratios = [float(r["ally_value_ratio"]) for r in records]
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
        "family_counts": {k: dict(v) for k, v in family_counts.items()},
        "difficulty_proxy_counts": {k: dict(v) for k, v in difficulty_counts.items()},
        "ally_value_ratio": {
            "min": min(ratios), "max": max(ratios), "mean": statistics.mean(ratios), "median": statistics.median(ratios),
        },
        "initial_min_cross_distance": {
            "min": min(distances), "max": max(distances), "mean": statistics.mean(distances), "median": statistics.median(distances),
        },
        "allowed_units_in_groups": sorted(ALLOWED_UNITS),
        "excluded_special_units": sorted(SPECIAL_EXCLUDED),
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "static_checks": [
            "exact JSON schema", "name/filename agreement", "allowed basic units only",
            "excluded special units absent from groups", "exact 3v2 composition", "unit-count field agreement",
            "global unit-type vocabulary", "per-faction shield homogeneity and matching flags",
            "ally static value ratio >= 1.18", "group placement emulation", "spawn bounds",
            "terrain walkability", "same-plane overlap rejection", "attack-point bounds",
            "engagement bucket", "duplicate semantic map rejection",
        ],
        "dynamic_validation_required": True,
    }
    (out_root / "validation_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_root / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, digest in sorted(content_hashes.items()):
            f.write(f"{digest}  {rel}\n")

    shutil.copy2(Path(__file__).resolve(), out_root / "generate_3v2_basic_mixed_480.py")
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
