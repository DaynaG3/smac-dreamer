"""Tests for the finish_trade_v4 reward. Pure Python — no smaclite/torch required.

v4 iterates on v3 to hit two failure modes: post-contact timeout/disengagement and high-enemy-EHP
all-dead wipeouts. Key new behaviours vs v3: the stall penalty is gated on FIRST CONTACT and
scaled by remaining enemy EHP; timeout is base+enemy+allies; all-dead is base+large-enemy.
"""

import math

import pytest

from smacdreamer.envs.reward_registry import resolve, resolved_params, available, RewardContext


def _ctx(**kw):
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
    assert "finish_trade_v4" in available()
    assert {"finish_trade_v1", "finish_trade_v2", "finish_trade_v3"}.issubset(available())
    rp = resolved_params("finish_trade_v4", {"w_stall": 0.05})
    assert rp["w_stall"] == 0.05                       # override kept
    assert rp["w_enemy_progress"] == 0.22
    assert rp["w_ally_loss"] == 0.32
    assert rp["stall_grace"] == 8
    assert rp["w_timeout_base"] == 1.25
    assert rp["w_timeout_enemy"] == 1.50
    assert rp["w_timeout_alive"] == 0.75
    assert rp["w_all_dead_base"] == 0.75
    assert rp["w_all_dead_enemy"] == 1.40


def test_a_enemy_progress_positive():
    fn = resolve("finish_trade_v4")
    _, t = fn(_ctx(step_idx=2, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))
    assert t["enemy_progress"] == pytest.approx(0.22 * 0.2)   # +0.044


def test_b_ally_loss_negative():
    fn = resolve("finish_trade_v4")
    _, t = fn(_ctx(step_idx=2, prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["ally_loss"] == pytest.approx(-0.32 * 0.1)       # -0.032


def test_c_no_stall_before_first_contact():
    # Many no-damage steps BEFORE ever dealing damage -> never penalised (early positioning).
    fn = resolve("finish_trade_v4", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    for s in range(1, 30):
        _, t = fn(_ctx(step_idx=s, enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0))
        assert t["stall_penalty"] == pytest.approx(0.0)
        assert t["no_damage_streak_max"] == pytest.approx(0.0)   # streak never starts pre-contact


def test_d_stall_only_after_contact_and_grace():
    fn = resolve("finish_trade_v4", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    # step 1: first contact (enemy 1.0 -> 0.8) sets has_dealt_damage_before, streak resets to 0
    fn(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))
    pens = []
    for s in range(2, 8):   # no further damage; enemy held at 0.8
        _, t = fn(_ctx(step_idx=s, prev_enemy_ehp_frac=0.8, enemy_ehp_frac=0.8))
        pens.append(t["stall_penalty"])
    # streaks: s2->1, s3->2, s4->3(>grace) ... factor = 0.5 + 0.5*0.8 = 0.9
    assert pens[0] == pytest.approx(0.0)          # streak 1
    assert pens[1] == pytest.approx(0.0)          # streak 2 == grace (not >)
    assert pens[2] == pytest.approx(-0.1 * 0.25 * 0.9)   # streak 3 -> (3-2)/4, *0.9


def test_e_stall_scales_and_caps():
    fn = resolve("finish_trade_v4", {"stall_grace": 2, "stall_cap": 4, "w_stall": 0.1})
    fn(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))   # first contact
    last = 0.0
    for s in range(2, 25):
        _, t = fn(_ctx(step_idx=s, prev_enemy_ehp_frac=0.8, enemy_ehp_frac=0.8))
        last = t["stall_penalty"]
    # saturates at -w_stall * 1.0 * (0.5 + 0.5*0.8) = -0.1 * 0.9
    assert last == pytest.approx(-0.1 * 0.9)


def test_stall_scale_depends_on_remaining_enemy():
    # Same streak, healthier enemy -> larger penalty (0.5 + 0.5*enemy_ehp).
    def penalty(enemy_ehp):
        fn = resolve("finish_trade_v4", {"stall_grace": 1, "stall_cap": 2, "w_stall": 0.1})
        fn(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=enemy_ehp))  # contact
        p = 0.0
        for s in range(2, 8):
            _, t = fn(_ctx(step_idx=s, prev_enemy_ehp_frac=enemy_ehp, enemy_ehp_frac=enemy_ehp))
            p = t["stall_penalty"]
        return p
    assert penalty(0.9) < penalty(0.2) < 0.0     # healthier enemy -> more negative


def test_f_timeout_base_enemy_and_allies():
    fn = resolve("finish_trade_v4")
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=0.6, ally_alive_frac=0.5, enemies_alive=2, allies_alive=2))
    assert t["timeout_base"] == pytest.approx(-1.25)
    assert t["timeout_enemy"] == pytest.approx(-1.50 * 0.6)   # -0.90
    assert t["timeout_alive"] == pytest.approx(-0.75 * 0.5)   # -0.375
    assert t["timeout_with_allies_alive"] == pytest.approx(1.0)
    total_timeout = t["timeout_base"] + t["timeout_enemy"] + t["timeout_alive"]
    assert total_timeout == pytest.approx(-2.525)


def test_g_all_dead_high_enemy_worse_than_near_win():
    fn = resolve("finish_trade_v4")
    _, hi = fn(_ctx(step_idx=60, is_last=True, terminated=True, battle_won=False,
                    allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                    enemy_ehp_frac=0.8, enemies_alive=3))
    _, near = fn(_ctx(step_idx=1, is_last=True, terminated=True, battle_won=False,
                      allies_alive=0, ally_alive_frac=0.0, ally_ehp_frac=0.0,
                      enemy_ehp_frac=0.1, enemies_alive=1))
    hi_dead = hi["all_dead_base"] + hi["all_dead_enemy"]
    near_dead = near["all_dead_base"] + near["all_dead_enemy"]
    assert hi["all_dead_base"] == pytest.approx(-0.75)
    assert hi["all_dead_enemy"] == pytest.approx(-1.40 * 0.8)   # -1.12
    assert hi_dead == pytest.approx(-1.87)
    assert near_dead == pytest.approx(-0.75 - 1.40 * 0.1)       # -0.89
    assert hi_dead < near_dead                                  # healthy-enemy wipeout hurts most


def test_h_win_positive_but_modest():
    fn = resolve("finish_trade_v4")
    _, t = fn(_ctx(step_idx=40, max_episode_steps=200, is_last=True, terminated=True,
                   battle_won=True, enemy_ehp_frac=0.0, ally_ehp_frac=0.4,
                   enemies_alive=0, allies_alive=2))
    assert t["win_speed"] == pytest.approx(0.25 * (1.0 - 40 / 200))   # 0.20
    assert t["win_ally_ehp"] == pytest.approx(0.50 * 0.4)             # 0.20
    # a low-ally-EHP win is still net-positive (not over-penalised)
    assert t["win_speed"] + t["win_ally_ehp"] > 0.0
    assert t["all_dead_base"] == pytest.approx(0.0)


def test_has_dealt_damage_before_diagnostic():
    fn = resolve("finish_trade_v4")
    # No contact whole episode -> diagnostic 0 at terminal.
    _, t = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                   enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0, allies_alive=3))
    assert t["has_dealt_damage_before"] == pytest.approx(0.0)
    # With contact -> 1 at terminal.
    fn2 = resolve("finish_trade_v4")
    fn2(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.9))
    _, t2 = fn2(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                     prev_enemy_ehp_frac=0.9, enemy_ehp_frac=0.9, allies_alive=3))
    assert t2["has_dealt_damage_before"] == pytest.approx(1.0)


def test_i_no_nans_when_fracs_missing_nan_or_inf():
    fn = resolve("finish_trade_v4")
    r, t = fn(_ctx(step_idx=5, enemy_ehp_frac=None, prev_enemy_ehp_frac=None,
                   ally_ehp_frac=None, prev_ally_ehp_frac=None, ally_alive_frac=None))
    assert math.isfinite(r) and all(math.isfinite(float(v)) for v in t.values())
    r2, t2 = fn(_ctx(step_idx=6, enemy_ehp_frac=float("nan"), prev_enemy_ehp_frac=float("inf"),
                     ally_ehp_frac=float("-inf"), prev_ally_ehp_frac=float("nan"),
                     ally_alive_frac=float("nan")))
    assert math.isfinite(r2) and all(math.isfinite(float(v)) for v in t2.values())
    r0, t0 = fn(_ctx(step_idx=200, is_last=True, truncated=True, battle_won=False,
                     enemy_ehp_frac=0.0, ally_ehp_frac=0.0, ally_alive_frac=0.0, allies_alive=1))
    assert math.isfinite(r0) and all(math.isfinite(float(v)) for v in t0.values())


def test_j_state_resets_between_episodes():
    fn = resolve("finish_trade_v4", {"stall_grace": 1, "stall_cap": 5, "w_stall": 0.1})
    fn(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.8))    # contact
    for s in range(2, 8):
        fn(_ctx(step_idx=s, prev_enemy_ehp_frac=0.8, enemy_ehp_frac=0.8))
    _, ta = fn(_ctx(step_idx=8, is_last=True, truncated=True, battle_won=False,
                    enemy_ehp_frac=0.8, allies_alive=2, ally_alive_frac=0.5))
    assert ta["no_damage_streak_max"] >= 5
    assert ta["has_dealt_damage_before"] == pytest.approx(1.0)
    # New episode -> streak + first-contact flag cleared.
    _, tb = fn(_ctx(step_idx=1, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=1.0))
    assert tb["stall_penalty"] == pytest.approx(0.0)
    assert tb["no_damage_streak_max"] == pytest.approx(0.0)


def test_reward_equals_base_plus_shaping_total():
    fn = resolve("finish_trade_v4")
    r, t = fn(_ctx(step_idx=10, base_reward=1.5, prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.7,
                   prev_ally_ehp_frac=1.0, ally_ehp_frac=0.9))
    assert t["original"] == pytest.approx(1.5)
    assert r == pytest.approx(1.5 + t["shaping_total"])
    comp = (t["enemy_progress"] + t["ally_loss"] + t["stall_penalty"]
            + t["timeout_base"] + t["timeout_enemy"] + t["timeout_alive"]
            + t["win_speed"] + t["win_ally_ehp"] + t["all_dead_base"] + t["all_dead_enemy"])
    assert t["shaping_total"] == pytest.approx(comp)
