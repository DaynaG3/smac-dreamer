"""Tests for the full-observability sight-range override (smacdreamer.sight_range).

The override sets SMAClite's module constant AGENT_SIGHT_RANGE from the SMACLITE_SIGHT_RANGE env
var. Because that constant gates visibility (and doubles as the distance-normalization divisor),
a silent failure here would corrupt every observation in a full-vis run -- hence targeted tests.

Pure-Python where possible; smaclite-touching tests skip cleanly when the simulator is absent.
"""

import glob
import os
import pathlib

import pytest

from conftest import requires_smaclite

import smacdreamer.sight_range as sr
from smacdreamer.sight_range import ENV_VAR, maybe_override_sight_range


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate each test: clear the env var and the once-per-process log guard."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    sr._applied = False
    yield
    sr._applied = False


@pytest.fixture
def sight_constant():
    """Snapshot/restore smaclite's AGENT_SIGHT_RANGE so tests do not leak global state."""
    from smaclite.env import smaclite as _sm
    original = _sm.AGENT_SIGHT_RANGE
    try:
        yield _sm
    finally:
        _sm.AGENT_SIGHT_RANGE = original


def test_helper_no_env_var_returns_none():
    # No smaclite import happens on this path: unset env var -> pure no-op.
    assert os.environ.get(ENV_VAR) is None
    assert maybe_override_sight_range() is None


@requires_smaclite
def test_helper_sets_constant_from_env(sight_constant, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "24")
    assert maybe_override_sight_range() == 24
    assert sight_constant.AGENT_SIGHT_RANGE == 24


@requires_smaclite
def test_helper_default_leaves_nine(sight_constant):
    # Env var unset (autouse fixture cleared it); constant must be untouched.
    sight_constant.AGENT_SIGHT_RANGE = 9
    assert maybe_override_sight_range() is None
    assert sight_constant.AGENT_SIGHT_RANGE == 9


@requires_smaclite
def test_full_visibility_probe(sight_constant):
    """With radius 24 every agent sees all alive enemies on an r2_2100 map (full visibility)."""
    root = pathlib.Path(__file__).resolve().parent.parent
    maps = sorted(glob.glob(str(root / "configs/maps/r2_2100/configs/train/*.json")))
    if not maps:
        pytest.skip("r2_2100 train maps not present")

    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=maps[0])
    try:
        env.reset()
        if not hasattr(env, "neighbour_finder_enemy") or not hasattr(env, "agents"):
            pytest.skip("smaclite internals differ; visibility probe not applicable")
        agents = list(env.agents.values())            # env.agents is a dict {id: unit}
        alive_enemies = sum(1 for e in env.enemies.values() if e.hp > 0)
        visible_lists = env.neighbour_finder_enemy.query_radius(agents, 24, True)
        for per_agent in visible_lists:
            assert len(per_agent) == alive_enemies    # every agent sees every alive enemy
    finally:
        env.close()
