"""Adaptive hard-map prioritisation for the env map sampler (training-driven curriculum).

This is *map-level* prioritisation, NOT transition-level PER: no importance-sampling weights, no
Dreamer world-model loss changes, no TD-error priorities. The env map sampler keeps full baseline
coverage and merely *oversamples* maps the current policy is bad at.

Effective sampling (mode ``prioritized_hard_maps``) is a mixture::

    p(map) = (1 - hard_map_probability) * baseline_uniform   # default 75% shuffled_round_robin
           +      hard_map_probability  * hard_priority       # default 25% ∝ hard_score

The hard score per map is fixed by spec::

    hard_score = 0.60 * (1 - win_rate_ema)
               + 0.25 * final_enemy_ehp_frac_ema
               + 0.15 * timeout_rate_ema

``original_env_return`` is also tracked (for logging / per-family reporting) but is intentionally
NOT part of the hard score. The 75% baseline stream guarantees every map keeps a coverage floor.

Two pure-Python pieces (no torch/smaclite), both unit-testable:
  * ``compute_hard_scores`` — per-map EMA stats -> per-map hard score.
  * ``MapPriorityTracker`` — running per-map EMA of TRAINING-episode outcomes, broadcast cadence,
    and the ``sampler/*`` logging metrics (effective weight max/min/entropy, hard-score mean/top10,
    per-family win rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Fixed hard-score blend (must sum to 1.0 over the three difficulty signals).
HARD_SCORE_WEIGHTS = {"win": 0.60, "enemy_ehp": 0.25, "timeout": 0.15}
DEFAULT_HARD_MAP_PROBABILITY = 0.25


def _clip01(x: float) -> float:
    x = float(x)
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def hard_score(win_rate: float, final_enemy_ehp_frac: float, timeout_rate: float) -> float:
    """The fixed-weight hard-map difficulty score for one map."""
    return (
        HARD_SCORE_WEIGHTS["win"] * (1.0 - _clip01(win_rate))
        + HARD_SCORE_WEIGHTS["enemy_ehp"] * _clip01(final_enemy_ehp_frac)
        + HARD_SCORE_WEIGHTS["timeout"] * _clip01(timeout_rate)
    )


def compute_hard_scores(per_map_stats: Dict[str, dict]) -> Dict[str, float]:
    """``name -> hard_score`` from ``name -> {win_rate, final_enemy_ehp_frac, timeout_rate}``."""
    return {
        n: hard_score(
            s.get("win_rate", 0.0),
            s.get("final_enemy_ehp_frac", 0.0),
            s.get("timeout_rate", 0.0),
        )
        for n, s in per_map_stats.items()
    }


def effective_sample_weights(
    hard_scores: Dict[str, float],
    all_names: List[str],
    *,
    hard_map_probability: float = DEFAULT_HARD_MAP_PROBABILITY,
) -> Dict[str, float]:
    """Mixture probability per map: ``(1-p)*uniform + p*(score/Σscore)``, normalised to sum 1.

    ``hard_scores`` may cover only a subset of ``all_names`` (maps without enough episodes get no
    hard-component mass but still receive the uniform baseline share).
    """
    names = list(all_names)
    n = len(names)
    if n == 0:
        return {}
    p = max(0.0, min(float(hard_map_probability), 1.0))
    ssum = sum(max(float(v), 0.0) for v in hard_scores.values())
    eff: Dict[str, float] = {}
    for name in names:
        base = (1.0 - p) / n
        hard = p * (max(float(hard_scores.get(name, 0.0)), 0.0) / ssum) if ssum > 0 else 0.0
        eff[name] = base + hard
    tot = sum(eff.values()) or 1.0
    return {k: v / tot for k, v in eff.items()}


def _entropy(probs) -> float:
    """Shannon entropy (nats) of a probability vector; robust to zeros."""
    h = 0.0
    for p in probs:
        p = float(p)
        if p > 0.0:
            h -= p * math.log(p)
    return h


@dataclass
class MapPriorityTracker:
    """Running per-map EMA of TRAINING-episode outcomes + broadcast cadence + logging.

    The trainer calls :meth:`record` once per finished training episode (keyed by the env's
    integer ``log_map_id``) and :meth:`maybe_compute` once per step; the latter returns fresh
    ``name -> hard_score`` to broadcast to the workers on the configured cadence (after warm-up),
    or ``None``. :meth:`logging_metrics` produces the ``sampler/*`` scalars.
    """

    id_to_name: Dict[int, str]
    id_to_family: Optional[Dict[int, str]] = None
    every: int = 100_000               # env steps between hard-score broadcasts
    warmup: int = 100_000              # no broadcast before this many env steps
    ema_decay: float = 0.98            # per-episode EMA smoothing (higher = slower)
    min_episodes: int = 5              # episodes before a map contributes to the hard component
    hard_map_probability: float = DEFAULT_HARD_MAP_PROBABILITY
    _stats: Dict[int, dict] = field(default_factory=dict, init=False, repr=False)
    _next_update: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.id_to_name = {int(k): str(v) for k, v in dict(self.id_to_name).items()}
        self.id_to_family = ({int(k): str(v) for k, v in dict(self.id_to_family).items()}
                             if self.id_to_family else {})
        self.every = max(1, int(self.every))
        self.warmup = max(0, int(self.warmup))
        self._next_update = self.warmup

    # -- ingest -------------------------------------------------------------
    def record(self, map_id: int, *, battle_won: bool, final_enemy_ehp_frac: float,
               timeout: bool, original_return: float) -> None:
        """Fold one finished training episode into the per-map EMA."""
        mid = int(map_id)
        win = 1.0 if battle_won else 0.0
        to = 1.0 if timeout else 0.0
        enemy = _clip01(final_enemy_ehp_frac)
        ret = float(original_return)
        st = self._stats.get(mid)
        if st is None:
            self._stats[mid] = {"win_rate": win, "final_enemy_ehp_frac": enemy,
                                "timeout_rate": to, "original_return": ret, "n": 1}
            return
        d = float(self.ema_decay)
        st["win_rate"] = d * st["win_rate"] + (1 - d) * win
        st["final_enemy_ehp_frac"] = d * st["final_enemy_ehp_frac"] + (1 - d) * enemy
        st["timeout_rate"] = d * st["timeout_rate"] + (1 - d) * to
        st["original_return"] = d * st["original_return"] + (1 - d) * ret
        st["n"] += 1

    # -- hard scores --------------------------------------------------------
    def _eligible_stats(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for mid, st in self._stats.items():
            if st["n"] < self.min_episodes:
                continue
            name = self.id_to_name.get(mid)
            if name is None:
                continue
            out[name] = dict(st)
        return out

    def compute_hard_scores(self) -> Dict[str, float]:
        """Current ``name -> hard_score`` for eligible maps (empty until enough data)."""
        return compute_hard_scores(self._eligible_stats())

    def maybe_compute(self, step: int) -> Optional[Dict[str, float]]:
        """Return fresh hard scores on the cadence (post warm-up), else ``None``.

        Advances the schedule past ``step`` so a missed tick fires at most once. Returns ``None``
        when no map is eligible yet (sampler keeps its current scores / pure baseline).
        """
        step = int(step)
        if step < self._next_update:
            return None
        while step >= self._next_update:
            self._next_update += self.every
        scores = self.compute_hard_scores()
        return scores or None

    # -- logging ------------------------------------------------------------
    def logging_metrics(self) -> Dict[str, float]:
        """The ``sampler/*`` scalars for W&B/TensorBoard (computed in the main process)."""
        all_names = list(self.id_to_name.values())
        scores = self.compute_hard_scores()
        eff = effective_sample_weights(
            scores, all_names, hard_map_probability=self.hard_map_probability)
        metrics: Dict[str, float] = {
            "sampler/hard_map_probability": float(self.hard_map_probability),
        }
        if eff:
            ev = list(eff.values())
            metrics["sampler/map_sample_weight/max"] = max(ev)
            metrics["sampler/map_sample_weight/min"] = min(ev)
            metrics["sampler/map_sample_weight/entropy"] = _entropy(ev)
        sv = sorted(scores.values(), reverse=True)
        if sv:
            metrics["sampler/hard_score_mean"] = sum(sv) / len(sv)
            top = sv[:min(10, len(sv))]
            metrics["sampler/hard_score_top10"] = sum(top) / len(top)
        # Per-family win rate (episode-weighted EMA mean over the family's maps), if families known.
        if self.id_to_family:
            fam_win: Dict[str, list] = {}
            for mid, st in self._stats.items():
                fam = self.id_to_family.get(mid)
                if fam is None:
                    continue
                fam_win.setdefault(fam, []).append(st["win_rate"])
            for fam, vals in fam_win.items():
                metrics[f"sampler/family_win_rate/{fam}"] = sum(vals) / len(vals)
        return metrics


__all__ = [
    "hard_score", "compute_hard_scores", "effective_sample_weights",
    "MapPriorityTracker", "HARD_SCORE_WEIGHTS", "DEFAULT_HARD_MAP_PROBABILITY",
]
