from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERRAIN_TYPES = {"_": 0.0, "C": 1.0, "X": 2.0}
ENTITY_STATIC_FEAT_SIZE = 12


@dataclass(frozen=True)
class JEPATokenSpec:
    n_agents: int
    n_enemies: int
    max_agents: int
    max_enemies: int
    max_actions: int
    ally_state_feat_size: int
    enemy_state_feat_size: int
    dynamic_token_dim: int
    entity_static_feat_size: int
    static_dim: int
    token_dim: int
    ally_has_shields: bool
    enemy_has_shields: bool
    num_unit_types: int
    mode: str = "entity"

    @property
    def entities(self) -> int:
        return self.max_agents + self.max_enemies

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "max_agents": self.max_agents,
            "max_enemies": self.max_enemies,
            "max_actions": self.max_actions,
            "token_dim": self.token_dim,
            "dynamic_token_dim": self.dynamic_token_dim,
            "static_dim": self.static_dim,
            "entity_static_feat_size": self.entity_static_feat_size,
            "ally_state_feat_size": self.ally_state_feat_size,
            "enemy_state_feat_size": self.enemy_state_feat_size,
            "ally_has_shields": self.ally_has_shields,
            "enemy_has_shields": self.enemy_has_shields,
            "num_unit_types": self.num_unit_types,
            "n_actions": self.max_actions,
        }


def spec_from_env(uw, pad_dims=None) -> JEPATokenSpec:
    mi = uw.map_info
    max_agents = int(pad_dims.max_agents) if pad_dims is not None else int(uw.n_agents)
    max_enemies = int(pad_dims.max_enemies) if pad_dims is not None else int(uw.n_enemies)
    max_actions = int(pad_dims.max_actions) if pad_dims is not None else int(uw.n_actions)
    ally_size = int(getattr(uw, "ally_state_feat_size", getattr(uw, "ally_feat_size", 0)))
    enemy_size = int(getattr(uw, "enemy_state_feat_size", getattr(uw, "enemy_feat_size", 0)))
    dynamic = max(ally_size, enemy_size, 1)
    static_dim = 9 + 3 + 32 * 32
    return JEPATokenSpec(
        n_agents=int(uw.n_agents),
        n_enemies=int(uw.n_enemies),
        max_agents=max_agents,
        max_enemies=max_enemies,
        max_actions=max_actions,
        ally_state_feat_size=ally_size,
        enemy_state_feat_size=enemy_size,
        dynamic_token_dim=dynamic,
        entity_static_feat_size=ENTITY_STATIC_FEAT_SIZE,
        static_dim=static_dim,
        token_dim=dynamic + ENTITY_STATIC_FEAT_SIZE,
        ally_has_shields=bool(getattr(mi, "ally_has_shields", False)),
        enemy_has_shields=bool(getattr(mi, "enemy_has_shields", False)),
        num_unit_types=int(getattr(mi, "num_unit_types", 0)),
    )


def static_condition_from_env(uw) -> np.ndarray:
    mi = uw.map_info
    terrain = np.asarray(
        [[TERRAIN_TYPES.get(getattr(cell, "value", str(cell)), 0.0) / 2.0 for cell in row] for row in mi.terrain],
        dtype=np.float32,
    )
    counts = np.asarray([(terrain == value).mean() for value in (0.0, 0.5, 1.0)], dtype=np.float32)
    flat = np.zeros((32 * 32,), dtype=np.float32)
    h, w = min(terrain.shape[0], 32), min(terrain.shape[1], 32)
    flat.reshape(32, 32)[:h, :w] = terrain[:h, :w]
    base = np.asarray(
        [
            float(mi.width) / 64.0,
            float(mi.height) / 64.0,
            float(mi.attack_point[0]) / max(float(mi.width), 1.0),
            float(mi.attack_point[1]) / max(float(mi.height), 1.0),
            float(bool(mi.ally_has_shields)),
            float(bool(mi.enemy_has_shields)),
            float(mi.num_unit_types) / 16.0,
            float(uw.n_agents) / 64.0,
            float(uw.n_enemies) / 256.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, counts, flat]).astype(np.float32)


def _unit_static(unit) -> np.ndarray:
    stats = unit.type.stats
    return np.asarray(
        [
            float(stats.hp) / 1000.0,
            float(stats.shield) / 1000.0,
            float(stats.damage) / 100.0,
            float(stats.cooldown) / 100.0,
            float(stats.speed) / 10.0,
            float(stats.attack_range) / 20.0,
            float(stats.size) / 10.0,
            float(stats.armor) / 10.0,
            float(stats.energy) / 200.0,
            float(stats.attacks) / 10.0,
            1.0 if str(stats.combat_type).endswith("HEALING") else 0.0,
            1.0 if str(stats.plane).endswith("AIR") else 0.0,
        ],
        dtype=np.float32,
    )


def entity_static_from_env(uw, spec: JEPATokenSpec) -> np.ndarray:
    out = np.zeros((spec.entities, spec.entity_static_feat_size), dtype=np.float32)
    for _, unit in sorted(uw.agents.items()):
        idx = int(unit.id_in_faction)
        if 0 <= idx < spec.max_agents:
            out[idx] = _unit_static(unit)
    for _, unit in sorted(uw.enemies.items()):
        idx = spec.max_agents + int(unit.id_in_faction)
        if spec.max_agents <= idx < spec.entities:
            out[idx] = _unit_static(unit)
    return out


def encode_state_vector(state: np.ndarray, spec: JEPATokenSpec, entity_static: np.ndarray):
    features = np.zeros((spec.entities, spec.token_dim), dtype=np.float32)
    mask = np.zeros((spec.entities,), dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    real_ally_rows = min(spec.n_agents, spec.max_agents)
    ally_len = min(state.size, real_ally_rows * spec.ally_state_feat_size)
    if ally_len:
        allies = state[:ally_len].reshape(real_ally_rows, spec.ally_state_feat_size)
        features[:real_ally_rows, : spec.ally_state_feat_size] = allies
        mask[:real_ally_rows] = (np.abs(allies).sum(axis=-1) > 0).astype(np.float32)
    enemy_src = state[ally_len:]
    real_enemy_rows = min(spec.n_enemies, spec.max_enemies, enemy_src.size // max(spec.enemy_state_feat_size, 1))
    enemy_len = real_enemy_rows * spec.enemy_state_feat_size
    if enemy_len:
        enemies = enemy_src[:enemy_len].reshape(real_enemy_rows, spec.enemy_state_feat_size)
        start = spec.max_agents
        features[start : start + real_enemy_rows, : spec.enemy_state_feat_size] = enemies
        mask[start : start + real_enemy_rows] = (np.abs(enemies).sum(axis=-1) > 0).astype(np.float32)
    off = spec.dynamic_token_dim
    features[:, off : off + spec.entity_static_feat_size] = entity_static[:, : spec.entity_static_feat_size]
    slot = np.zeros((spec.entities,), dtype=np.float32)
    slot[: spec.max_agents] = 1.0
    slot[spec.max_agents : spec.max_agents + spec.max_enemies] = 1.0
    return features, mask, slot


def build_jepa_observation(uw, pad_dims=None) -> tuple[dict[str, np.ndarray], JEPATokenSpec]:
    spec = spec_from_env(uw, pad_dims)
    entity_static = entity_static_from_env(uw, spec)
    state = np.asarray(uw.get_state(), dtype=np.float32)
    entity, mask, slot = encode_state_vector(state, spec, entity_static)
    static = static_condition_from_env(uw)
    return {
        "jepa_entity": entity.astype(np.float32),
        "jepa_entity_mask": mask.astype(np.float32),
        "jepa_entity_slot_mask": slot.astype(np.float32),
        "jepa_static_condition": static.astype(np.float32),
    }, spec
