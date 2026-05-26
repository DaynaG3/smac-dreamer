"""Map sampler for Phase 2 same-shape multi-map training."""
import pathlib
import random
from dataclasses import dataclass
from typing import List, Optional

import ruamel.yaml as yaml


@dataclass
class MapEntry:
    name: str
    type: str         # 'builtin' or 'custom'
    path: Optional[str] = None  # required when type='custom'


_VALID_TYPES = ('builtin', 'custom')


def validate_manifest(manifest_path: str) -> dict:
    """Load and validate a Phase 2 map manifest.

    Raises ValueError for:
    - empty map list
    - unknown map type
    - custom entry missing path
    - missing custom map file
    """
    p = pathlib.Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    raw = yaml.YAML(typ='safe').load(p.read_text(encoding='utf-8'))
    if not raw or 'maps' not in raw or not raw['maps']:
        raise ValueError(f"Manifest '{manifest_path}' has no maps.")
    if 'padding' in raw:
        pad = raw['padding']
        for key in ('max_agents', 'max_enemies', 'max_actions', 'max_obs_size'):
            if key not in pad or not isinstance(pad[key], int) or pad[key] <= 0:
                raise ValueError(
                    f"Manifest '{manifest_path}': padding.{key} must be a positive int."
                )
    root = p.parent.parent.parent  # configs/maps -> configs -> project root
    for entry in raw['maps']:
        t = entry.get('type')
        if t not in _VALID_TYPES:
            raise ValueError(
                f"Manifest '{manifest_path}': map '{entry.get('name')}' has "
                f"unknown type {t!r}. Must be one of {_VALID_TYPES}."
            )
        if t == 'custom':
            ep = entry.get('path')
            if not ep:
                raise ValueError(
                    f"Manifest '{manifest_path}': custom map '{entry.get('name')}' "
                    "has no 'path' field."
                )
            abs_path = root / ep
            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Manifest '{manifest_path}': custom map file not found: {abs_path}"
                )
    return raw


class MapSampler:
    """Returns the next map entry on each episode reset.

    Modes:
      fixed         — always returns the first map (Phase 1 compatibility)
      round_robin   — cycles through maps in order
      seeded_random — reproducible random choice

    peek() returns the map that the next next() call would return, without
    advancing the internal index. Used by SMACliteDreamerEnv.__init__ to
    configure the initial env without consuming an episode slot.
    """

    MODES = ('fixed', 'round_robin', 'seeded_random')

    def __init__(self, maps: List[MapEntry], mode: str = 'round_robin', seed: int = 0):
        if mode not in self.MODES:
            raise ValueError(
                f"MapSampler mode must be one of {self.MODES}, got {mode!r}"
            )
        if not maps:
            raise ValueError("MapSampler requires at least one MapEntry.")
        self.maps = list(maps)
        self.mode = mode
        self._idx = 0
        self._rng = random.Random(seed)
        # For seeded_random: pre-generate the first choice so peek() is stable.
        self._next_random: Optional[MapEntry] = (
            self._rng.choice(self.maps) if mode == 'seeded_random' else None
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        mode: str = 'round_robin',
        seed: int = 0,
    ) -> 'MapSampler':
        """Load and validate a manifest, then construct a MapSampler."""
        raw = validate_manifest(manifest_path)
        maps = [
            MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
            for e in raw['maps']
        ]
        return cls(maps=maps, mode=mode, seed=seed)

    def peek(self) -> MapEntry:
        """Return the map that the next next() call would return, without advancing."""
        if self.mode == 'fixed':
            return self.maps[0]
        if self.mode == 'round_robin':
            return self.maps[self._idx]
        return self._next_random  # seeded_random

    def next(self) -> MapEntry:
        """Return the next map and advance the internal state."""
        if self.mode == 'fixed':
            return self.maps[0]
        if self.mode == 'round_robin':
            entry = self.maps[self._idx]
            self._idx = (self._idx + 1) % len(self.maps)
            return entry
        # seeded_random: return pre-generated choice and prepare the next one
        entry = self._next_random
        self._next_random = self._rng.choice(self.maps)
        return entry
