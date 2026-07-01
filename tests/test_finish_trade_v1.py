"""Tests for the finish_trade_v1 reward. Pure Python — no smaclite/torch required.

Covers the exact reward formula from the spec: per-step trade terms, the state-level
no-progress stall penalty (with grace + cap), terminal timeout/win/all-dead anchors, the
behaviour diagnostics, per-episode state reset, and NaN-freedom.
"""

import math

import pytest

from smacdreamer.envs.reward_registry import resolve, resolved_params, available, RewardContext


def _ctx(**kw):
    """RewardContext with sensible mid-fight defaults, overridable per test."""
    base = dict(
        base_reward=0.0, step_idx=1, max_episode_steps=200,
        enemies_alive=3, allies_alive=3,
        enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0,
        ally_ehp_frac=1.0, prev_ally_ehp_frac=1.0,
        ally_alive_frac=1.0, prev_ally_alive_frac=1.0,
    )
    base.update(kw)
    return RewardContext(**base)


def test_registered_and_defaults():
    assert "finish_trade_v1" in available()
    rp = resolved_params("finish_trade_v1", {"w_stall": 0.05})
    assert rp["w_stall"] == 0.05                 # override kept
    assert rp["w_enemy_progress"] == 0.25        # default filled
    assert rp["progress_eps"] == 0.0005
    assert rp["w_all_dead_loss"] == 0.75


def test_base_reward_preserved_when_static():
    fn = resolve("finish_trade_v1")
    r, t = fn(_ctx(base_reward=2.0, step_idx=1))
    # No enemy progress, no ally loss, no terminal -> shaping ~ 0, reward == base.
    assert t["original"] == pytest.approx(2.0)
    assert r == pytest.approx(2.0)
    assert t["shaping_total"] == pytest.approx(0.0)


def test_enemy_progress_and_ally_loss_terms():
    fn = resolve("finish_trade_v1")
    r, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8,
                   prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["enemy_progress"] == pytest.approx(0.25 * 0.2)    # +0.05
    assert t["ally_loss"] == pytest.approx(-0.20 * 0.1)        # -0.02
    assert r == pytest.approx(0.25 * 0.2 - 0.20 * 0.1)


def test_ally_loss_never_positive_on_heal():
    # ally EHP increasing (healer) must not create a positive "loss" term.
    fn = resolve("finish_trade_v1")
    _, t = fn(_ctx(step_idx=2, prev_ally_ehp_frac=0.8, ally_ehp_frac=0.9))
    assert t["ally_loss"] == pytest.approx(0.0)


def test_stall_penalty_after_grace_and_cap():
    fn = resolve("finish_trade_v1", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    # First step resets state. Feed steps with no enemy progress; both sides alive.
    penalties = []
    for s in range(1, 12):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        penalties.append(t["stall_penalty"])
    # streak grows 1..; grace=2 -> first penalty when streak==3.
    # streak 3 -> scale (3-2)/4=0.25 -> -0.025; caps at streak>=6 -> scale 1.0 -> -0.1.
    assert penalties[0] == pytest.approx(0.0)   # streak 1
    assert penalties[1] == pytest.approx(0.0)   # streak 2 (== grace, not > grace)
    assert penalties[2] == pytest.approx(-0.1 * 0.25)  # streak 3
    assert penalties[-1] == pytest.approx(-0.1)         # saturated at cap


def test_stall_resets_on_progress():
    fn = resolve("finish_trade_v1", {"stall_grace": 1, "stall_cap": 10, "w_stall": 0.1})
    for s in range(1, 6):   # build a streak
        fn(_ctx(step_idx=s))
    # A step WITH enemy progress resets the streak -> no stall penalty this step.
    _, t = fn(_ctx(step_idx=6, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.9))
    assert t["stall_penalty"] == pytest.approx(0.0)


def test_terminal_timeout_penalty_scaled():
    fn = resolve("finish_trade_v1")
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=0.4, ally_alive_frac=0.5, enemies_alive=2, allies_alive=2))
    assert t["timeout_enemy"] == pytest.approx(-1.25 * 0.4)
    assert t["timeout_alive"] == pytest.approx(-0.50 * 0.5)
    assert t["timeout_with_allies_alive"] == pytest.approx(1.0)
    assert t["near_win_timeout"] == pytest.approx(0.0)   # enemy_ehp 0.4 > 0.15


def test_near_win_timeout_flag():
    fn = resolve("finish_trade_v1")
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=0.1, ally_alive_frac=0.6, enemies_alive=1, allies_alive=2))
    assert t["timeout_with_allies_alive"] == pytest.approx(1.0)
    assert t["near_win_timeout"] == pytest.approx(1.0)    # enemy_ehp 0.1 <= 0.15


def test_terminal_win_bonuses():
    fn = resolve("finish_trade_v1")
    _, t = fn(_ctx(step_idx=50, max_episode_steps=200, is_last=True, terminated=True,
                   battle_won=True, enemy_ehp_frac=0.0, ally_ehp_frac=0.7,
                   enemies_alive=0, allies_alive=3))
    assert t["win_speed"] == pytest.approx(0.50 * (1.0 - 50 / 200))
    assert t["win_ally_ehp"] == pytest.approx(0.50 * 0.7)
    # Wins are not timeouts / all-dead.
    assert t["timeout_enemy"] == pytest.approx(0.0)
    assert t["all_dead_loss"] == pytest.approx(0.0)


def test_terminal_all_dead_loss():
    fn = resolve("finish_trade_v1")
    _, t = fn(_ctx(step_idx=120, is_last=True, terminated=True, battle_won=False,
                   allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                   enemy_ehp_frac=0.5, enemies_alive=2))
    assert t["all_dead_loss"] == pytest.approx(-0.75)
    assert t["allies_dead_loss"] == pytest.approx(1.0)
    assert t["timeout_with_allies_alive"] == pytest.approx(0.0)


def test_state_resets_between_episodes():
    fn = resolve("finish_trade_v1", {"stall_grace": 1, "stall_cap": 5, "w_stall": 0.1})
    # Episode A: build a big streak, end at terminal.
    for s in range(1, 8):
        fn(_ctx(step_idx=s))
    _, ta = fn(_ctx(step_idx=8, is_last=True, truncated=True, battle_won=False,
                    enemy_ehp_frac=0.5, allies_alive=2, ally_alive_frac=0.5))
    assert ta["no_damage_streak_max"] >= 5
    # Episode B: step_idx resets to 1 -> streak state cleared -> no penalty on step 1.
    _, tb = fn(_ctx(step_idx=1))
    assert tb["stall_penalty"] == pytest.approx(0.0)
    assert tb["no_damage_streak_max"] == pytest.approx(0.0)  # not terminal -> 0 emitted


def test_no_nans_across_terms():
    fn = resolve("finish_trade_v1")
    for ctx in [
        _ctx(step_idx=1),
        _ctx(step_idx=10, enemy_ehp_frac=0.5, prev_enemy_ehp_frac=0.7, ally_ehp_frac=0.6,
             prev_ally_ehp_frac=0.8),
        _ctx(step_idx=200, is_last=True, truncated=True, battle_won=False, enemy_ehp_frac=0.3,
             ally_alive_frac=0.4, allies_alive=1),
        _ctx(step_idx=30, is_last=True, terminated=True, battle_won=True, enemy_ehp_frac=0.0,
             ally_ehp_frac=0.9, enemies_alive=0),
    ]:
        r, t = fn(ctx)
        assert math.isfinite(r)
        assert all(math.isfinite(float(v)) for v in t.values())
