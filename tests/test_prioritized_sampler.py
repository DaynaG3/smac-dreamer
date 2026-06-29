"""Tests for the MapSampler 'prioritized_hard_maps' mixture mode.

Mixture: (1 - hard_map_probability) baseline shuffled_round_robin coverage + hard_map_probability
hard-map oversampling (∝ runtime hard scores). Pure-Python (no torch/smaclite).
"""

import collections

import pytest

from smacdreamer.envs.map_sampler import MapSampler, MapEntry


def _entries(names):
    return [MapEntry(name=n, type="builtin", map_id=i) for i, n in enumerate(names)]


def _sampler(names, seed=0, hmp=0.25):
    return MapSampler.from_entries(_entries(names), mode="prioritized_hard_maps",
                                   seed=seed, hard_map_probability=hmp)


def test_mode_registered():
    assert "prioritized_hard_maps" in MapSampler.MODES


def test_peek_matches_next():
    s = _sampler(["a", "b", "c", "d"])
    for _ in range(20):
        assert s.peek().name == s.next().name


def test_baseline_only_covers_every_map_each_cycle():
    # No hard scores set -> behaves as pure shuffled_round_robin (full coverage per cycle).
    names = [f"m{i}" for i in range(6)]
    s = _sampler(names)
    seen = [s.next().name for _ in range(len(names))]
    assert sorted(seen) == sorted(names)   # first cycle visits each map exactly once


def test_hard_scores_bias_sampling_toward_hard_map():
    names = [f"m{i}" for i in range(5)]
    s = _sampler(names, hmp=0.25)
    # m3 is by far the hardest.
    s.set_hard_scores({"m0": 0.01, "m1": 0.01, "m2": 0.01, "m3": 10.0, "m4": 0.01})
    counts = collections.Counter(s.next().name for _ in range(4000))
    # m3 should be sampled clearly more than a uniform 1/5 share.
    assert counts["m3"] > 4000 / 5
    # every map still gets visited (baseline coverage preserved).
    assert all(counts[n] > 0 for n in names)


def test_full_hard_probability_targets_scored_map():
    names = ["a", "b", "c"]
    s = _sampler(names, hmp=1.0)
    s.set_hard_scores({"b": 1.0})   # only b has a positive score
    # With hmp=1.0 and a single scored map, every draw is the hard pick.
    assert all(s.next().name == "b" for _ in range(50))


def test_set_hard_scores_ignores_unknown_names():
    s = _sampler(["a", "b"])
    s.set_hard_scores({"a": 1.0, "ghost": 99.0})   # 'ghost' not in map set
    # Should not raise and only 'a'/'b' are ever sampled.
    names = {s.next().name for _ in range(100)}
    assert names <= {"a", "b"}


def test_determinism_same_seed_same_scores():
    names = [f"m{i}" for i in range(5)]
    scores = {"m0": 0.1, "m1": 0.9, "m2": 0.3, "m3": 2.0, "m4": 0.05}
    a = _sampler(names, seed=7)
    b = _sampler(names, seed=7)
    a.set_hard_scores(scores)
    b.set_hard_scores(scores)
    assert [a.next().name for _ in range(200)] == [b.next().name for _ in range(200)]


def test_set_hard_scores_noop_for_other_modes():
    s = MapSampler.from_entries(_entries(["a", "b"]), mode="shuffled_round_robin", seed=0)
    s.set_hard_scores({"a": 5.0})   # silently ignored
    assert s._hard_scores == {}


def test_coverage_metrics_still_advance():
    names = [f"m{i}" for i in range(4)]
    s = _sampler(names)
    for _ in range(8):
        s.next()
    cm = s.coverage_metrics()
    assert cm["total_train_maps"] == 4
    assert cm["total_unique_maps_seen"] >= 1
