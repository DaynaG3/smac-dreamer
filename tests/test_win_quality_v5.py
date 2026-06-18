"""Tests for the win_quality_v5 reward (reward-transfer continuation, 2M -> 4M).

Pure-Python reward-function tests (no smaclite/torch) — always run. win_quality_v5 must not
disturb smaclite_default, dense_v3, or ally_ehp_v4 (covered unchanged elsewhere).

The reward = original SMAClite reward
  + Term1 dense allied-EHP preservation (shifted potential, like ally_ehp_v4)
  + Term2 terminal win-EHP bonus  (WINS ONLY)
  + Term3 terminal surviving-allies bonus (WINS ONLY)
  - timeout penalty on truncation.
"""

import pytest

from smacdreamer.envs.reward_registry import resolve, resolved_params, available, RewardContext


# win_quality_v5 default weights
W_EHP_DENSE = 0.25
W_WIN_EHP = 0.50
W_WIN_ALIVE = 0.25
W_TIMEOUT = 0.10


def _ctx(**kw):
    base = dict(
        base_reward=0.0, gamma=1.0,
        ally_ehp_frac=1.0, prev_ally_ehp_frac=1.0,
        ally_alive_frac=1.0, prev_ally_alive_frac=1.0,
        enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0,
        terminated=False, truncated=False, battle_won=False,
    )
    base.update(kw)
    return RewardContext(**base)


# ----------------------------------------------------------------------
# Registry plumbing
# ----------------------------------------------------------------------

def test_registered_and_does_not_shadow_others():
    names = available()
    assert "win_quality_v5" in names
    # existing rewards remain available and untouched
    assert {"smaclite_default", "dense_v3", "ally_ehp_v4"}.issubset(names)


def test_resolved_params_fills_defaults():
    rp = resolved_params("win_quality_v5", {})
    assert rp["w_ally_ehp_dense"] == pytest.approx(W_EHP_DENSE)
    assert rp["w_win_ehp"] == pytest.approx(W_WIN_EHP)
    assert rp["w_win_alive"] == pytest.approx(W_WIN_ALIVE)
    assert rp["w_timeout"] == pytest.approx(W_TIMEOUT)


def test_terms_dict_has_expected_keys():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx())
    assert set(t) == {"original", "ally_ehp_dense", "win_ehp_quality",
                      "win_alive_quality", "timeout", "shaping_total"}


# ----------------------------------------------------------------------
# Term 1: dense allied-EHP preservation (shifted potential, like ally_ehp_v4)
# ----------------------------------------------------------------------

def test_full_health_no_event_gives_zero():
    fn = resolve("win_quality_v5")
    r, t = fn(_ctx())
    assert t["ally_ehp_dense"] == pytest.approx(0.0)
    assert t["shaping_total"] == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_ally_ehp_damage_is_negative_dense_term():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(prev_ally_ehp_frac=1.0, ally_ehp_frac=0.8))
    # w*(gamma*(0.8-1) - (1-1)) = 0.25*(-0.2) = -0.05
    assert t["ally_ehp_dense"] == pytest.approx(-0.05)


def test_dense_term_uses_shifted_potential_and_terminal_zero():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(prev_ally_ehp_frac=0.5, ally_ehp_frac=0.2, terminated=True, battle_won=False))
    # terminal forces phi_next = 0: w*(gamma*0 - (0.5-1)) = 0.25*0.5 = 0.125
    assert t["ally_ehp_dense"] == pytest.approx(W_EHP_DENSE * 0.5)


def test_dense_term_telescopes_to_zero_over_episode():
    fn = resolve("win_quality_v5")
    gamma = 0.9
    fracs = [1.0, 0.9, 0.7, 0.4]
    discounted = 0.0
    for i in range(len(fracs) - 1):
        terminated = (i == len(fracs) - 2)
        _, t = fn(_ctx(prev_ally_ehp_frac=fracs[i], ally_ehp_frac=fracs[i + 1],
                       terminated=terminated, gamma=gamma, battle_won=True))
        discounted += (gamma ** i) * t["ally_ehp_dense"]
    assert discounted == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# Terms 2 & 3: win-quality bonuses — ONLY on a true terminal win
# ----------------------------------------------------------------------

def test_win_ehp_quality_fires_on_terminal_win():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(terminated=True, battle_won=True, ally_ehp_frac=0.6, ally_alive_frac=0.8))
    assert t["win_ehp_quality"] == pytest.approx(W_WIN_EHP * 0.6)
    assert t["win_alive_quality"] == pytest.approx(W_WIN_ALIVE * 0.8)


def test_no_win_bonus_on_terminal_loss():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(terminated=True, battle_won=False, ally_ehp_frac=0.6, ally_alive_frac=0.8))
    assert t["win_ehp_quality"] == pytest.approx(0.0)
    assert t["win_alive_quality"] == pytest.approx(0.0)


def test_no_win_bonus_on_non_terminal_step():
    fn = resolve("win_quality_v5")
    # battle_won True but not terminated -> no bonus (bonus is terminal-only)
    _, t = fn(_ctx(terminated=False, battle_won=True, ally_ehp_frac=0.6, ally_alive_frac=0.8))
    assert t["win_ehp_quality"] == pytest.approx(0.0)
    assert t["win_alive_quality"] == pytest.approx(0.0)


def test_no_win_bonus_on_truncation_even_if_battle_won_flag_set():
    fn = resolve("win_quality_v5")
    # truncation is mutually exclusive with terminated; bonus requires terminated
    _, t = fn(_ctx(terminated=False, truncated=True, battle_won=True,
                   ally_ehp_frac=0.6, ally_alive_frac=0.8))
    assert t["win_ehp_quality"] == pytest.approx(0.0)
    assert t["win_alive_quality"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Timeout penalty
# ----------------------------------------------------------------------

def test_timeout_penalty_only_on_truncation():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(truncated=True))
    assert t["timeout"] == pytest.approx(-W_TIMEOUT)


def test_no_timeout_penalty_on_terminal():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx(terminated=True, battle_won=True))
    assert t["timeout"] == pytest.approx(0.0)


def test_no_timeout_penalty_on_intermediate_step():
    fn = resolve("win_quality_v5")
    _, t = fn(_ctx())
    assert t["timeout"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Composition: reward = base + sum of terms
# ----------------------------------------------------------------------

def test_reward_is_base_plus_shaping_total():
    fn = resolve("win_quality_v5")
    r, t = fn(_ctx(base_reward=3.0, terminated=True, battle_won=True,
                   prev_ally_ehp_frac=0.9, ally_ehp_frac=0.7, ally_alive_frac=0.6))
    assert t["original"] == pytest.approx(3.0)
    parts = t["ally_ehp_dense"] + t["win_ehp_quality"] + t["win_alive_quality"] + t["timeout"]
    assert t["shaping_total"] == pytest.approx(parts)
    assert r == pytest.approx(3.0 + t["shaping_total"])


def test_custom_weights_respected():
    fn = resolve("win_quality_v5", {"w_win_ehp": 1.0, "w_win_alive": 0.0, "w_timeout": 0.5})
    _, t = fn(_ctx(terminated=True, battle_won=True, ally_ehp_frac=0.5, ally_alive_frac=0.5))
    assert t["win_ehp_quality"] == pytest.approx(0.5)   # 1.0 * 0.5
    assert t["win_alive_quality"] == pytest.approx(0.0)  # weight 0
    _, t2 = fn(_ctx(truncated=True))
    assert t2["timeout"] == pytest.approx(-0.5)
