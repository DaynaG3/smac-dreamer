"""Tests for the finish_trade_v2 reward. Pure Python — no smaclite/torch required.

Covers the spec: clamped per-step trade terms, the state-level no-progress stall (grace + cap),
terminal timeout/win anchors, the all-dead loss WITH the unfinished-close penalty, the
diagnostics, per-episode reset, and NaN-safety for missing/zero EHP fractions.
"""

import math

import pytest

from smacdreamer.envs.reward_registry import resolve, resolved_params, available, RewardContext


def _ctx(**kw):
    """RewardContext with mid-fight defaults, overridable per test."""
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
    assert "finish_trade_v2" in available()
    rp = resolved_params("finish_trade_v2", {"w_stall": 0.02})
    assert rp["w_stall"] == 0.02                       # override kept
    assert rp["w_ally_loss"] == 0.32                   # default filled
    assert rp["w_all_dead_loss"] == 1.25
    assert rp["w_unfinished_close_loss"] == 0.35
    assert rp["stall_grace"] == 12


def test_a_enemy_progress_positive():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))
    assert t["enemy_progress"] == pytest.approx(0.22 * 0.2)   # +0.044


def test_enemy_progress_clamped_on_enemy_heal():
    # v2 clamps enemy_progress to >= 0 (v1 allowed negative).
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=0.8, enemy_ehp_frac=1.0))
    assert t["enemy_progress"] == pytest.approx(0.0)


def test_b_ally_loss_negative():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=2, prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["ally_loss"] == pytest.approx(-0.32 * 0.1)       # -0.032


def test_c_stall_only_after_grace():
    fn = resolve("finish_trade_v2", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    pens = []
    for s in range(1, 8):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        pens.append(t["stall_penalty"])
    assert pens[0] == pytest.approx(0.0)     # streak 1
    assert pens[1] == pytest.approx(0.0)     # streak 2 == grace (not >)
    assert pens[2] == pytest.approx(-0.1 * 0.25)   # streak 3 -> (3-2)/4


def test_d_stall_capped():
    fn = resolve("finish_trade_v2", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    last = 0.0
    for s in range(1, 20):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        last = t["stall_penalty"]
    assert last == pytest.approx(-0.1)       # saturates at -w_stall


def test_e_timeout_uses_final_enemy_ehp_and_ally_alive():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=0.4, ally_alive_frac=0.5, enemies_alive=2, allies_alive=2))
    assert t["timeout_enemy"] == pytest.approx(-1.00 * 0.4)
    assert t["timeout_alive"] == pytest.approx(-0.30 * 0.5)
    assert t["timeout_with_allies_alive"] == pytest.approx(1.0)


def test_f_win_uses_speed_and_final_ally_ehp():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=40, max_episode_steps=200, is_last=True, terminated=True,
                   battle_won=True, enemy_ehp_frac=0.0, ally_ehp_frac=0.6,
                   enemies_alive=0, allies_alive=3))
    assert t["win_speed"] == pytest.approx(0.25 * (1.0 - 40 / 200))
    assert t["win_ally_ehp"] == pytest.approx(0.80 * 0.6)
    assert t["all_dead_loss"] == pytest.approx(0.0)
    assert t["unfinished_close_loss"] == pytest.approx(0.0)


def test_g_all_dead_includes_unfinished_close():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=120, is_last=True, terminated=True, battle_won=False,
                   allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                   enemy_ehp_frac=0.2, enemies_alive=1))
    assert t["all_dead_loss"] == pytest.approx(-1.25)
    assert t["unfinished_close_loss"] == pytest.approx(-0.35 * (1.0 - 0.2))   # -0.28
    assert t["allies_dead_loss"] == pytest.approx(1.0)
    assert t["near_win_loss"] == pytest.approx(0.0)   # enemy_ehp 0.2 > 0.15


def test_near_win_loss_flag_on_close_wipeout():
    fn = resolve("finish_trade_v2")
    _, t = fn(_ctx(step_idx=120, is_last=True, terminated=True, battle_won=False,
                   allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                   enemy_ehp_frac=0.1, enemies_alive=1))
    assert t["near_win_loss"] == pytest.approx(1.0)   # enemy_ehp 0.1 <= 0.15
    # And the unfinished-close penalty is near its max for a nearly-finished enemy.
    assert t["unfinished_close_loss"] == pytest.approx(-0.35 * 0.9)


def test_h_no_nans_when_fracs_missing_or_zero():
    fn = resolve("finish_trade_v2")
    # Missing (None) fractions must not produce NaNs (coerced to safe defaults).
    r, t = fn(_ctx(step_idx=5, enemy_ehp_frac=None, prev_enemy_ehp_frac=None,
                   ally_ehp_frac=None, prev_ally_ehp_frac=None, ally_alive_frac=None))
    assert math.isfinite(r)
    assert all(math.isfinite(float(v)) for v in t.values())
    # Zero fractions (fully destroyed) also finite.
    r0, t0 = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                     enemy_ehp_frac=0.0, ally_ehp_frac=0.0, ally_alive_frac=0.0, allies_alive=1))
    assert math.isfinite(r0)
    assert all(math.isfinite(float(v)) for v in t0.values())


def test_state_resets_between_episodes():
    fn = resolve("finish_trade_v2", {"stall_grace": 1, "stall_cap": 5, "w_stall": 0.1})
    for s in range(1, 8):
        fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
    _, ta = fn(_ctx(step_idx=8, is_last=True, truncated=True, battle_won=False,
                    enemy_ehp_frac=0.5, allies_alive=2, ally_alive_frac=0.5))
    assert ta["no_damage_streak_max"] >= 5
    _, tb = fn(_ctx(step_idx=1))       # new episode -> streak state cleared
    assert tb["stall_penalty"] == pytest.approx(0.0)
    assert tb["no_damage_streak_max"] == pytest.approx(0.0)
