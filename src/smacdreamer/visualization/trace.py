"""Pure-Python trace helpers: action labels, target-focus metric, episode summary.

Deliberately torch-free and pygame-free so the label/metric logic can be unit-tested in
the JAX-free test environment (only numpy is required, and even that is optional here).

Action label convention (per the visualisation spec)::

    0       -> NOOP
    1       -> STOP
    2       -> MOVE_N
    3       -> MOVE_E
    4       -> MOVE_S
    5       -> MOVE_W
    >= 6    -> ATTACK_<action - 6>

Note: the >= 6 actions are SMAClite *target* slots. We label them ``ATTACK_<n>`` because
allied damage units attack enemy slot ``n``; for healer/ambiguous target maps the same slot
index is still the target id, so the index is the load-bearing part of the label.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence


# Explicit per the spec. Intentionally does NOT mirror SMAClite's internal Direction enum
# ordering — the spec fixes these four labels and the tests assert them.
_MOVE_LABELS = {2: "MOVE_N", 3: "MOVE_E", 4: "MOVE_S", 5: "MOVE_W"}


def action_label(action: int) -> str:
    """Map a single executed action index to a human-readable label."""
    a = int(action)
    if a == 0:
        return "NOOP"
    if a == 1:
        return "STOP"
    if a in _MOVE_LABELS:
        return _MOVE_LABELS[a]
    if a >= 6:
        return f"ATTACK_{a - 6}"
    return f"UNKNOWN_{a}"


def action_labels(actions: Sequence[int]) -> list:
    """Map a sequence of executed action indices to labels."""
    return [action_label(a) for a in actions]


def target_focus_score(executed_actions: Sequence[int]) -> Optional[float]:
    """Fraction of attack actions this step that hit the single most-common target.

    Returns ``None`` when no attack (>=6) actions were executed this step (so it is excluded
    from the episode-level mean rather than counted as 0). Example: ``[6, 6, 7] -> 2/3``.
    """
    target_ids = [int(a) - 6 for a in executed_actions if int(a) >= 6]
    if not target_ids:
        return None
    counts = Counter(target_ids)
    most_common_target_count = max(counts.values())
    return most_common_target_count / len(target_ids)


def classify_episode(
    summary: dict,
    *,
    low_enemy_ehp_threshold: float = 0.75,
    poor_focus_threshold: float = 0.5,
    min_attack_steps_for_focus: int = 5,
) -> dict:
    """Return ``{low_enemy_damage, poor_target_focus}`` booleans for a finished episode.

    * low_enemy_damage  : lost AND final enemy EHP fraction >= ``low_enemy_ehp_threshold``
                          (the agent failed to meaningfully damage the enemy team).
    * poor_target_focus : lost AND mean per-step target-focus < ``poor_focus_threshold`` AND
                          at least ``min_attack_steps_for_focus`` steps had an attack action
                          (the agent attacked enough to judge, but spread fire incoherently).
    """
    lost = not bool(summary.get("battle_won", False))
    final_enemy_ehp = summary.get("final_enemy_ehp_frac")
    mean_focus = summary.get("mean_target_focus_score")
    attack_steps = int(summary.get("attack_steps", 0))

    low_enemy_damage = bool(
        lost and final_enemy_ehp is not None and final_enemy_ehp >= low_enemy_ehp_threshold
    )
    poor_target_focus = bool(
        lost
        and mean_focus is not None
        and mean_focus < poor_focus_threshold
        and attack_steps >= min_attack_steps_for_focus
    )
    return {"low_enemy_damage": low_enemy_damage, "poor_target_focus": poor_target_focus}


def summarise_episode(
    records: Sequence[dict],
    *,
    map_name: str,
    seed: int,
    battle_won: bool,
    low_enemy_ehp_threshold: float = 0.75,
    poor_focus_threshold: float = 0.5,
    min_attack_steps_for_focus: int = 5,
) -> dict:
    """Compute the per-episode summary from the per-step JSONL records.

    ``records`` are the per-step dicts emitted by the rollout (one per env step). The summary
    aggregates executed-action statistics, return totals, final HP fractions, and the
    low-enemy-damage / poor-target-focus classification flags.
    """
    n_steps = len(records)
    total_original = sum(float(r.get("original_reward", 0.0)) for r in records)
    total_shaped = sum(float(r.get("reward", 0.0)) for r in records)
    total_enemy_dmg = sum(float(r.get("enemy_hp_damage_this_step", 0.0)) for r in records)

    attack_count = move_count = noop_stop_count = 0
    target_hist: Counter = Counter()
    focus_values = []
    attack_steps = 0
    for r in records:
        acts = [int(a) for a in r.get("executed_actions", [])]
        for a in acts:
            if a <= 1:
                noop_stop_count += 1
            elif a <= 5:
                move_count += 1
            else:
                attack_count += 1
                target_hist[a - 6] += 1
        focus = r.get("target_focus_score")
        if focus is not None:
            focus_values.append(float(focus))
            attack_steps += 1

    mean_focus = (sum(focus_values) / len(focus_values)) if focus_values else None

    # Final HP fractions / alive counts from the last step (env exposes the episode-end
    # values only on the terminal step; mid-episode they carry the zero sentinel).
    last = records[-1] if records else {}
    final_enemy_ehp = last.get("final_enemy_ehp_frac_if_available")
    final_ally_ehp = last.get("final_ally_ehp_frac_if_available")
    enemies_alive_end = last.get("enemies_alive")
    allies_alive_end = last.get("allies_alive")

    summary = {
        "map": map_name,
        "seed": int(seed),
        "battle_won": bool(battle_won),
        "episode_length": n_steps,
        "total_original_return": total_original,
        "total_shaped_reward": total_shaped,
        "total_enemy_hp_damage": total_enemy_dmg,
        "final_enemy_ehp_frac": final_enemy_ehp,
        "final_ally_ehp_frac": final_ally_ehp,
        "enemies_alive_at_end": enemies_alive_end,
        "allies_alive_at_end": allies_alive_end,
        "attack_action_count": attack_count,
        "move_action_count": move_count,
        "noop_stop_action_count": noop_stop_count,
        "per_target_histogram": dict(sorted(target_hist.items())),
        "mean_target_focus_score": mean_focus,
        "attack_steps": attack_steps,
    }
    summary.update(
        classify_episode(
            summary,
            low_enemy_ehp_threshold=low_enemy_ehp_threshold,
            poor_focus_threshold=poor_focus_threshold,
            min_attack_steps_for_focus=min_attack_steps_for_focus,
        )
    )
    return summary


def assert_structured_obs_mode(run_meta: dict, *, source: str = "run_meta.json") -> str:
    """Validate that the checkpoint was trained with structured observations.

    Returns the resolved ``obs_mode`` on success; raises ``ValueError`` with a clear message
    otherwise. This visualiser intentionally supports structured checkpoints only.
    """
    obs_mode = str((run_meta or {}).get("obs_mode", "")).strip()
    if obs_mode == "structured":
        return obs_mode
    raise ValueError(
        f"This visualizer currently supports structured checkpoints only "
        f"(obs_mode == 'structured'), but {source} reports obs_mode={obs_mode!r}. "
        "Flat-observation support is not implemented yet."
    )


__all__ = [
    "action_label",
    "action_labels",
    "target_focus_score",
    "classify_episode",
    "summarise_episode",
    "assert_structured_obs_mode",
]
