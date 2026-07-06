"""Tests for the finish_trade_v3 reward. Pure Python — no smaclite/torch required.

v3 keeps finish_trade_v2's exact structure and only rebalances weights (reward enemy progress
more, penalise ally loss / wipeouts less, tilt the win bonus toward speed). These tests cover the
spec: clamped per-step trade terms, the state-level no-progress stall (grace + cap), terminal
timeout/win anchors, the all-dead loss WITH the unfinished-close penalty, per-episode reset, and
NaN/inf-safety for missing/degenerate EHP fractions.
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
    assert "finish_trade_v3" in available()
    # v1 / v2 remain available and untouched.
    assert {"finish_trade_v1", "finish_trade_v2"}.issubset(available())
    rp = resolved_params("finish_trade_v3", {"w_stall": 0.05})
    assert rp["w_stall"] == 0.05                      # override kept
    # v3 defaults (rebalanced vs v2)
    assert rp["w_enemy_progress"] == 0.28
    assert rp["w_ally_loss"] == 0.22
    assert rp["stall_grace"] == 10
    assert rp["w_timeout_enemy"] == 1.10
    assert rp["w_win_speed"] == 0.35
    assert rp["w_win_ally_ehp"] == 0.60
    assert rp["w_all_dead_loss"] == 0.90
    assert rp["w_unfinished_close_loss"] == 0.20


def test_a_enemy_progress_positive():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))
    assert t["enemy_progress"] == pytest.approx(0.28 * 0.2)   # +0.056


def test_b_ally_loss_negative():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=2, prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["ally_loss"] == pytest.approx(-0.22 * 0.1)       # -0.022


def test_c_enemy_progress_clamped_on_enemy_heal():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=0.8, enemy_ehp_frac=1.0))
    assert t["enemy_progress"] == pytest.approx(0.0)


def test_d_stall_only_after_grace():
    fn = resolve("finish_trade_v3", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    pens = []
    for s in range(1, 8):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        pens.append(t["stall_penalty"])
    assert pens[0] == pytest.approx(0.0)             # streak 1
    assert pens[1] == pytest.approx(0.0)             # streak 2 == grace (not >)
    assert pens[2] == pytest.approx(-0.1 * 0.25)     # streak 3 -> (3-2)/4


def test_e_stall_capped():
    fn = resolve("finish_trade_v3", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    last = 0.0
    for s in range(1, 20):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        last = t["stall_penalty"]
    assert last == pytest.approx(-0.1)               # saturates at -w_stall


def test_f_timeout_uses_final_enemy_ehp_and_ally_alive():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=0.4, ally_alive_frac=0.5, enemies_alive=2, allies_alive=2))
    assert t["timeout_enemy"] == pytest.approx(-1.10 * 0.4)   # -0.44
    assert t["timeout_alive"] == pytest.approx(-0.30 * 0.5)   # -0.15
    assert t["timeout_with_allies_alive"] == pytest.approx(1.0)


def test_g_win_uses_speed_and_final_ally_ehp():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=40, max_episode_steps=200, is_last=True, terminated=True,
                   battle_won=True, enemy_ehp_frac=0.0, ally_ehp_frac=0.6,
                   enemies_alive=0, allies_alive=3))
    assert t["win_speed"] == pytest.approx(0.35 * (1.0 - 40 / 200))   # 0.28
    assert t["win_ally_ehp"] == pytest.approx(0.60 * 0.6)             # 0.36
    assert t["all_dead_loss"] == pytest.approx(0.0)
    assert t["unfinished_close_loss"] == pytest.approx(0.0)


def test_h_all_dead_includes_unfinished_close():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=120, is_last=True, terminated=True, battle_won=False,
                   allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                   enemy_ehp_frac=0.2, enemies_alive=1))
    assert t["all_dead_loss"] == pytest.approx(-0.90)
    assert t["unfinished_close_loss"] == pytest.approx(-0.20 * (1.0 - 0.2))   # -0.16
    assert t["allies_dead_loss"] == pytest.approx(1.0)
    assert t["near_win_loss"] == pytest.approx(0.0)   # enemy_ehp 0.2 > 0.15


def test_near_win_loss_flag_on_close_wipeout():
    fn = resolve("finish_trade_v3")
    _, t = fn(_ctx(step_idx=120, is_last=True, terminated=True, battle_won=False,
                   allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                   enemy_ehp_frac=0.1, enemies_alive=1))
    assert t["near_win_loss"] == pytest.approx(1.0)   # enemy_ehp 0.1 <= 0.15
    assert t["unfinished_close_loss"] == pytest.approx(-0.20 * 0.9)   # -0.18


def test_i_no_nans_when_fracs_missing_zero_nan_or_inf():
    fn = resolve("finish_trade_v3")
    # Missing (None) fractions.
    r, t = fn(_ctx(step_idx=5, enemy_ehp_frac=None, prev_enemy_ehp_frac=None,
                   ally_ehp_frac=None, prev_ally_ehp_frac=None, ally_alive_frac=None))
    assert math.isfinite(r) and all(math.isfinite(float(v)) for v in t.values())
    # NaN / inf fractions must be coerced to safe defaults, never propagate.
    r2, t2 = fn(_ctx(step_idx=6, enemy_ehp_frac=float("nan"), prev_enemy_ehp_frac=float("inf"),
                     ally_ehp_frac=float("-inf"), prev_ally_ehp_frac=float("nan"),
                     ally_alive_frac=float("nan")))
    assert math.isfinite(r2) and all(math.isfinite(float(v)) for v in t2.values())
    # Zero fractions (fully destroyed) at a terminal timeout also finite.
    r0, t0 = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                     enemy_ehp_frac=0.0, ally_ehp_frac=0.0, ally_alive_frac=0.0, allies_alive=1))
    assert math.isfinite(r0) and all(math.isfinite(float(v)) for v in t0.values())


def test_j_state_resets_between_episodes():
    fn = resolve("finish_trade_v3", {"stall_grace": 1, "stall_cap": 5, "w_stall": 0.1})
    for s in range(1, 8):
        fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
    _, ta = fn(_ctx(step_idx=8, is_last=True, truncated=True, battle_won=False,
                    enemy_ehp_frac=0.5, allies_alive=2, ally_alive_frac=0.5))
    assert ta["no_damage_streak_max"] >= 5
    _, tb = fn(_ctx(step_idx=1))       # new episode -> streak state cleared
    assert tb["stall_penalty"] == pytest.approx(0.0)
    assert tb["no_damage_streak_max"] == pytest.approx(0.0)


def test_reward_equals_base_plus_shaping_total():
    fn = resolve("finish_trade_v3")
    r, t = fn(_ctx(step_idx=10, base_reward=2.0, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.7,
                   prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["original"] == pytest.approx(2.0)
    assert r == pytest.approx(2.0 + t["shaping_total"])
    # shaping_total is the sum of the reward-component terms
    comp = (t["enemy_progress"] + t["ally_loss"] + t["stall_penalty"]
            + t["timeout_enemy"] + t["timeout_alive"] + t["win_speed"] + t["win_ally_ehp"]
            + t["all_dead_loss"] + t["unfinished_close_loss"])
    assert t["shaping_total"] == pytest.approx(comp)


def test_v3_rebalance_vs_v2_direction():
    """Sanity: v3 rewards a favourable trade (enemy down, small ally loss) more than v2."""
    v2 = resolve("finish_trade_v2")
    v3 = resolve("finish_trade_v3")
    kw = dict(step_idx=5, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8,
              prev_ally_ehp_frac=1.0, ally_ehp_frac=0.95)
    _, t2 = v2(_ctx(**kw))
    _, t3 = v3(_ctx(**kw))
    net2 = t2["enemy_progress"] + t2["ally_loss"]
    net3 = t3["enemy_progress"] + t3["ally_loss"]
    assert net3 > net2   # v3 makes the same favourable trade more rewarding
