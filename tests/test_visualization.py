"""Tests for the structured-observation policy visualisation helpers.

These cover the torch-free / pygame-free logic (action labels, target-focus metric, episode
summary classification, structured-only validation). They do NOT require a trained checkpoint
or the SMAClite simulator. A CLI ``--help`` smoke check is included for both scripts.
"""

import json
import pathlib
import subprocess
import sys

import pytest

from smacdreamer.visualization import trace

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --- Action labels ---------------------------------------------------------

@pytest.mark.parametrize("action,label", [
    (0, "NOOP"),
    (1, "STOP"),
    (2, "MOVE_N"),
    (3, "MOVE_E"),
    (4, "MOVE_S"),
    (5, "MOVE_W"),
    (6, "ATTACK_0"),
    (9, "ATTACK_3"),
])
def test_action_label(action, label):
    assert trace.action_label(action) == label


def test_action_labels_sequence():
    assert trace.action_labels([0, 6, 7]) == ["NOOP", "ATTACK_0", "ATTACK_1"]


# --- Target focus metric ---------------------------------------------------

def test_target_focus_two_thirds():
    assert trace.target_focus_score([6, 6, 7]) == pytest.approx(2 / 3)


def test_target_focus_one_third():
    assert trace.target_focus_score([6, 7, 8]) == pytest.approx(1 / 3)


def test_target_focus_none_when_no_attacks():
    assert trace.target_focus_score([2, 3, 1]) is None


# --- Structured-only validation --------------------------------------------

def test_assert_structured_passes():
    assert trace.assert_structured_obs_mode({"obs_mode": "structured"}) == "structured"


def test_assert_structured_rejects_flat():
    with pytest.raises(ValueError) as exc:
        trace.assert_structured_obs_mode({"obs_mode": "flat"})
    assert "structured checkpoints only" in str(exc.value)


def test_assert_structured_rejects_missing():
    with pytest.raises(ValueError):
        trace.assert_structured_obs_mode({})


# --- Episode summary + classification --------------------------------------

def _records(execs_per_step, *, enemy_ehp_final, won, allies=3, enemies=2):
    """Build minimal per-step records like the rollout emits."""
    recs = []
    n = len(execs_per_step)
    for i, execs in enumerate(execs_per_step, start=1):
        is_last = i == n
        recs.append({
            "step": i, "map": "m", "seed": 0,
            "executed_actions": execs,
            "action_labels": trace.action_labels(execs),
            "reward": 0.0, "original_reward": 0.0,
            "battle_won": won if is_last else False,
            "enemies_alive": enemies, "allies_alive": allies,
            "enemy_hp_damage_this_step": 0.0,
            "final_enemy_ehp_frac_if_available": (enemy_ehp_final if is_last else None),
            "final_ally_ehp_frac_if_available": (0.4 if is_last else None),
            "target_focus_score": trace.target_focus_score(execs),
        })
    return recs


def test_summary_counts_and_histogram():
    recs = _records([[6, 6, 2], [7, 1, 0]], enemy_ehp_final=0.9, won=False)
    s = trace.summarise_episode(recs, map_name="m", seed=0, battle_won=False)
    assert s["episode_length"] == 2
    assert s["attack_action_count"] == 3      # 6,6,7
    assert s["move_action_count"] == 1        # 2
    assert s["noop_stop_action_count"] == 2   # 1,0
    assert s["per_target_histogram"] == {0: 2, 1: 1}
    assert s["attack_steps"] == 2


def test_summary_low_enemy_damage_flag():
    recs = _records([[6, 6], [6, 6]], enemy_ehp_final=0.8, won=False)
    s = trace.summarise_episode(recs, map_name="m", seed=0, battle_won=False,
                                low_enemy_ehp_threshold=0.75)
    assert s["low_enemy_damage"] is True
    # A win never counts as low_enemy_damage.
    s_win = trace.summarise_episode(recs, map_name="m", seed=0, battle_won=True,
                                    low_enemy_ehp_threshold=0.75)
    assert s_win["low_enemy_damage"] is False


def test_summary_poor_target_focus_flag():
    # 6 attack steps, each spreading across two distinct targets -> focus 0.5 each.
    recs = _records([[6, 7]] * 6, enemy_ehp_final=0.3, won=False)
    s = trace.summarise_episode(recs, map_name="m", seed=0, battle_won=False,
                                poor_focus_threshold=0.6, min_attack_steps_for_focus=5)
    assert s["mean_target_focus_score"] == pytest.approx(0.5)
    assert s["attack_steps"] == 6
    assert s["poor_target_focus"] is True
    # Too few attack steps -> not flagged even with low focus.
    recs_few = _records([[6, 7]] * 3, enemy_ehp_final=0.3, won=False)
    s_few = trace.summarise_episode(recs_few, map_name="m", seed=0, battle_won=False,
                                    poor_focus_threshold=0.6, min_attack_steps_for_focus=5)
    assert s_few["poor_target_focus"] is False


# --- CLI --help smoke checks -----------------------------------------------

@pytest.mark.parametrize("script", ["visualize_episode.py", "visualize_batch.py"])
def test_cli_help(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
