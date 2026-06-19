"""Tests for adaptive hard-map prioritisation (src/smacdreamer/map_priority.py).

Pure-Python: no torch/smaclite. Covers the fixed hard-score formula, the effective-mixture
weights, and the MapPriorityTracker EMA + cadence + logging.
"""

import math

import pytest

from smacdreamer.map_priority import (
    hard_score, compute_hard_scores, effective_sample_weights,
    MapPriorityTracker, HARD_SCORE_WEIGHTS, DEFAULT_HARD_MAP_PROBABILITY,
)


# ----------------------------------------------------------------------
# hard_score formula: 0.60*(1-win) + 0.25*enemy_ehp + 0.15*timeout
# ----------------------------------------------------------------------

def test_hard_score_weights_are_spec():
    assert HARD_SCORE_WEIGHTS == {"win": 0.60, "enemy_ehp": 0.25, "timeout": 0.15}
    assert sum(HARD_SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_hard_score_perfect_easy_map_is_zero():
    # always wins, no surviving enemies, never times out -> not hard at all
    assert hard_score(win_rate=1.0, final_enemy_ehp_frac=0.0, timeout_rate=0.0) == pytest.approx(0.0)


def test_hard_score_worst_map_is_one():
    # never wins, enemies at full health, always times out -> maximally hard
    assert hard_score(win_rate=0.0, final_enemy_ehp_frac=1.0, timeout_rate=1.0) == pytest.approx(1.0)


def test_hard_score_components():
    assert hard_score(0.5, 0.0, 0.0) == pytest.approx(0.60 * 0.5)
    assert hard_score(1.0, 0.4, 0.0) == pytest.approx(0.25 * 0.4)
    assert hard_score(1.0, 0.0, 0.8) == pytest.approx(0.15 * 0.8)


def test_hard_score_clips_out_of_range_inputs():
    assert hard_score(-0.5, 2.0, 5.0) == pytest.approx(0.60 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0)


def test_compute_hard_scores_maps_names():
    stats = {
        "easy": {"win_rate": 1.0, "final_enemy_ehp_frac": 0.0, "timeout_rate": 0.0},
        "hard": {"win_rate": 0.0, "final_enemy_ehp_frac": 0.8, "timeout_rate": 0.5},
    }
    scores = compute_hard_scores(stats)
    assert scores["hard"] > scores["easy"]
    assert scores["easy"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# effective_sample_weights: 75% uniform + 25% hard (default)
# ----------------------------------------------------------------------

def test_effective_weights_sum_to_one():
    scores = {"a": 1.0, "b": 3.0}
    eff = effective_sample_weights(scores, ["a", "b"], hard_map_probability=0.25)
    assert sum(eff.values()) == pytest.approx(1.0)


def test_effective_weights_hard_map_gets_more_mass():
    scores = {"a": 1.0, "b": 3.0}   # b is harder
    eff = effective_sample_weights(scores, ["a", "b"], hard_map_probability=0.25)
    assert eff["b"] > eff["a"]
    # baseline 75% is uniform (0.375 each); hard 25% splits 1:3 -> a:+0.0625, b:+0.1875
    assert eff["a"] == pytest.approx(0.375 + 0.0625)
    assert eff["b"] == pytest.approx(0.375 + 0.1875)


def test_effective_weights_zero_hard_prob_is_uniform():
    eff = effective_sample_weights({"a": 5.0, "b": 0.0}, ["a", "b"], hard_map_probability=0.0)
    assert eff["a"] == pytest.approx(0.5)
    assert eff["b"] == pytest.approx(0.5)


def test_effective_weights_map_without_score_still_gets_baseline():
    # 'c' has no hard score -> only baseline share, but non-zero (coverage preserved)
    eff = effective_sample_weights({"a": 1.0}, ["a", "c"], hard_map_probability=0.25)
    assert eff["c"] > 0.0
    assert eff["a"] > eff["c"]


def test_effective_weights_no_scores_is_uniform():
    eff = effective_sample_weights({}, ["a", "b", "c"], hard_map_probability=0.25)
    for v in eff.values():
        assert v == pytest.approx(1 / 3)


# ----------------------------------------------------------------------
# MapPriorityTracker: EMA, eligibility, cadence, logging
# ----------------------------------------------------------------------

def _tracker(**kw):
    base = dict(id_to_name={0: "easy", 1: "hard"}, id_to_family={0: "fam_a", 1: "fam_b"},
                every=1000, warmup=1000, ema_decay=0.5, min_episodes=2,
                hard_map_probability=0.25)
    base.update(kw)
    return MapPriorityTracker(**base)


def test_tracker_records_and_scores_hard_map_higher():
    t = _tracker()
    for _ in range(5):
        t.record(0, battle_won=True, final_enemy_ehp_frac=0.0, timeout=False, original_return=20.0)
        t.record(1, battle_won=False, final_enemy_ehp_frac=0.9, timeout=True, original_return=1.0)
    scores = t.compute_hard_scores()
    assert scores["hard"] > scores["easy"]


def test_tracker_min_episodes_gate():
    t = _tracker(min_episodes=3)
    t.record(0, battle_won=True, final_enemy_ehp_frac=0.0, timeout=False, original_return=1.0)
    t.record(0, battle_won=True, final_enemy_ehp_frac=0.0, timeout=False, original_return=1.0)
    # only 2 episodes < min_episodes=3 -> not yet eligible
    assert t.compute_hard_scores() == {}


def test_tracker_maybe_compute_respects_warmup_and_cadence():
    t = _tracker(warmup=1000, every=1000, min_episodes=1)
    t.record(1, battle_won=False, final_enemy_ehp_frac=1.0, timeout=True, original_return=0.0)
    assert t.maybe_compute(500) is None          # before warmup
    out = t.maybe_compute(1000)                   # first eligible tick
    assert out is not None and "hard" in out
    assert t.maybe_compute(1500) is None          # before next cadence
    assert t.maybe_compute(2000) is not None      # next tick


def test_tracker_maybe_compute_none_when_no_eligible_maps():
    t = _tracker(warmup=0, every=1000, min_episodes=10)
    t.record(0, battle_won=True, final_enemy_ehp_frac=0.0, timeout=False, original_return=1.0)
    assert t.maybe_compute(0) is None   # eligible set empty -> None (keep baseline)


def test_tracker_logging_metrics_keys():
    t = _tracker(min_episodes=1)
    for _ in range(3):
        t.record(0, battle_won=True, final_enemy_ehp_frac=0.0, timeout=False, original_return=20.0)
        t.record(1, battle_won=False, final_enemy_ehp_frac=0.9, timeout=True, original_return=1.0)
    m = t.logging_metrics()
    for key in ("sampler/hard_map_probability", "sampler/map_sample_weight/max",
                "sampler/map_sample_weight/min", "sampler/map_sample_weight/entropy",
                "sampler/hard_score_mean", "sampler/hard_score_top10"):
        assert key in m, f"missing {key}"
    assert m["sampler/hard_map_probability"] == pytest.approx(0.25)
    # per-family win rate logged when families known
    assert "sampler/family_win_rate/fam_a" in m
    assert "sampler/family_win_rate/fam_b" in m
    # entropy of a 2-map distribution is in [0, ln 2]
    assert 0.0 <= m["sampler/map_sample_weight/entropy"] <= math.log(2) + 1e-9
