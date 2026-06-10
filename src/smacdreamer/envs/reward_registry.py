"""Swappable reward functions for the centralised SMAClite controller.

A small name -> callable registry resolved from config, so the reward can be changed
without editing env or repo internals. Each callable takes a ``RewardContext`` (the base
SMAClite reward + the per-step combat stats the env already tracks) and returns
``(reward: float, terms: dict[str, float])`` — the scalar reward used for training plus a
per-term breakdown the env logs under ``log_reward_term_*`` keys.

Built-ins:
  smaclite_default — returns the base SMAClite reward unchanged (no shaping).
  v2_shaping       — the existing RewardShapingConfig v2 math (terminal + per-step combat).
  dense_v3         — denser default for multimap: potential-based (in raw-reward space)
                     enemy-HP-destroyed + ally-survival shaping, plus a terminal win/loss
                     bonus, plus an OPTIONAL non-potential positioning bonus (off by default).

IMPORTANT — invariance is approximate in this pipeline. ``dense_v3``'s potential terms are
Φ-differences (γΦ(s')−Φ(s)) and would be policy-invariant in RAW reward space. But DreamerV3
applies symlog/twohot + a running return normalizer before the critic, so a nonlinear
transform of a potential difference is no longer a potential difference and strict
policy-invariance does NOT hold exactly here. The shaping γ MUST equal the agent's discount;
even then invariance is only approximate. Eval-on-original-return is the guard against
objective distortion. The optional positioning term is explicitly NOT potential-based.

No JAX / torch / repo imports — pure Python. Reusable from the env and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ----------------------------------------------------------------------
# Context passed to every reward callable
# ----------------------------------------------------------------------

@dataclass
class RewardContext:
    """Everything a reward callable may need, sourced from values the env already computes.

    Potentials need both the current and previous normalised quantities; the env supplies
    fractions in [0, 1] so the shaping is map-size invariant.
    """
    base_reward: float                 # raw SMAClite reward this step (already [0,20]-scaled)

    # Per-step combat deltas (already computed in SMACliteDreamerEnv.step)
    kill_delta: int = 0                # enemies killed this step
    ally_deaths: int = 0              # allies that died this step
    enemy_hp_damage: float = 0.0       # absolute enemy HP removed this step

    # Normalised state quantities for potentials (fractions in [0, 1])
    enemy_hp_frac: float = 1.0         # current enemy HP / initial enemy HP  (Φ_hp = this)
    prev_enemy_hp_frac: float = 1.0    # previous step's enemy_hp_frac
    ally_alive_frac: float = 1.0       # current allies alive / initial allies (Φ_ally)
    prev_ally_alive_frac: float = 1.0  # previous step's ally_alive_frac

    # Counts (for non-potential bonuses / diagnostics)
    allies_alive: int = 0
    enemies_alive: int = 0

    # Episode framing
    is_last: bool = False
    battle_won: bool = False
    step_idx: int = 0
    max_episode_steps: int = 200

    # Discount — shaping γ MUST equal the agent's discount for raw-space telescoping.
    gamma: float = 0.997


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

# name -> factory(params: dict) -> callable(ctx) -> (reward, terms)
_REGISTRY: dict[str, Callable[[dict], Callable[[RewardContext], tuple]]] = {}


def register(name: str):
    def deco(factory):
        if name in _REGISTRY:
            raise ValueError(f"reward '{name}' already registered")
        _REGISTRY[name] = factory
        return factory
    return deco


def available() -> list:
    return sorted(_REGISTRY)


def resolve(name: str, params: Optional[dict] = None) -> Callable[[RewardContext], tuple]:
    """Return a ready-to-call reward function for ``name`` bound to ``params``."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown reward '{name}'. Available: {available()}")
    return _REGISTRY[name](params or {})


def resolved_params(name: str, params: Optional[dict] = None) -> dict:
    """Return the FULLY-resolved params for ``name`` (defaults filled in).

    Used for run-config logging and the reward hash so identical effective configs hash
    identically regardless of which fields the user left implicit.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown reward '{name}'. Available: {available()}")
    # Each factory exposes its defaults via a ``.defaults`` attribute set by @with_defaults.
    factory = _REGISTRY[name]
    defaults = dict(getattr(factory, "defaults", {}))
    defaults.update(params or {})
    return defaults


def _with_defaults(defaults: dict):
    """Attach a defaults dict to a factory so resolved_params can introspect it."""
    def deco(factory):
        factory.defaults = defaults
        return factory
    return deco


# ----------------------------------------------------------------------
# Built-in reward functions
# ----------------------------------------------------------------------

@register("smaclite_default")
@_with_defaults({})
def _make_smaclite_default(params: dict):
    def fn(ctx: RewardContext):
        return float(ctx.base_reward), {"original": float(ctx.base_reward)}
    return fn


@register("v2_shaping")
@_with_defaults({
    "win_bonus": 0.0, "loss_penalty": 0.0, "enemy_kill_bonus": 0.0,
    "ally_death_penalty": 0.0, "ally_survival_bonus": 0.0, "step_penalty": 0.0,
    "damage_delta_scale": 0.0,
})
def _make_v2_shaping(params: dict):
    """The existing v2 shaping math, expressed against RewardContext.

    Mirrors RewardShapingConfig semantics: terminal win/loss applied once on is_last;
    per-step kill/death/survival/step-penalty/damage. base_reward is always included.
    """
    d = _make_v2_shaping.defaults
    p = {**d, **params}

    def fn(ctx: RewardContext):
        win = p["win_bonus"] if (ctx.is_last and ctx.battle_won) else 0.0
        loss = p["loss_penalty"] if (ctx.is_last and not ctx.battle_won) else 0.0
        kill = ctx.kill_delta * p["enemy_kill_bonus"]
        death = ctx.ally_deaths * p["ally_death_penalty"]
        survival = ctx.allies_alive * p["ally_survival_bonus"]
        step_pen = p["step_penalty"]
        damage = ctx.enemy_hp_damage * p["damage_delta_scale"]
        shaping = win + loss + kill + death + survival - step_pen + damage
        reward = float(ctx.base_reward) + float(shaping)
        terms = {
            "original": float(ctx.base_reward),
            "win": float(win), "loss": float(loss), "kill": float(kill),
            "death": float(death), "survival": float(survival),
            "step_penalty": float(-step_pen), "damage": float(damage),
            "shaping_total": float(shaping),
        }
        return reward, terms
    return fn


@register("dense_v3")
@_with_defaults({
    # terminal anchor (reference scale)
    "w_win": 1.0,
    "w_loss": 1.0,          # magnitude of terminal loss penalty (applied as -w_loss)
    # potential-based densifying terms (small by default so shaping stays subordinate)
    "w_hp": 0.1,            # weight on γΦ_hp(s') − Φ_hp(s)  (enemy HP destroyed)
    "w_ally": 0.1,          # weight on γΦ_ally(s') − Φ_ally(s)  (ally survival)
    # OPTIONAL non-potential bonus bucket (off by default; NOT policy-invariant)
    "nonpotential": {"positioning_weight": 0.0},
})
def _make_dense_v3(params: dict):
    """Denser default: potential-based (raw space) HP + ally-survival, terminal win/loss.

    Φ_hp   = enemy_hp_frac        (1.0 full enemy HP -> 0.0 all destroyed); destroying enemy
             HP INCREASES progress, so we use the NEGATIVE potential -Φ_hp so that a drop in
             enemy HP yields positive shaping. Term = w_hp * (γ(−Φ_hp(s')) − (−Φ_hp(s)))
                                                     = w_hp * (Φ_hp(s) − γΦ_hp(s')).
    Φ_ally = ally_alive_frac; losing allies should not be rewarded, so the ally potential is
             +Φ_ally: term = w_ally * (γΦ_ally(s') − Φ_ally(s)) (<= 0 when allies die).
    Terminal win/loss is added once on is_last (terminal-only, as in standard SMAC).
    Positioning (optional) is a plain per-step bonus, NOT potential-based — off by default.
    """
    d = _make_dense_v3.defaults
    p = {**d, **params}
    nonp = {**d["nonpotential"], **(params.get("nonpotential") or {})}
    g = None  # gamma is taken from ctx (must equal the agent discount)

    def fn(ctx: RewardContext):
        gamma = ctx.gamma
        # HP-destroyed potential (negative potential of remaining enemy HP fraction).
        hp_term = p["w_hp"] * (ctx.prev_enemy_hp_frac - gamma * ctx.enemy_hp_frac)
        # Ally-survival potential.
        ally_term = p["w_ally"] * (gamma * ctx.ally_alive_frac - ctx.prev_ally_alive_frac)
        # Terminal win/loss (once).
        win_term = p["w_win"] if (ctx.is_last and ctx.battle_won) else 0.0
        loss_term = -p["w_loss"] if (ctx.is_last and not ctx.battle_won) else 0.0
        # Optional non-potential positioning bonus (off by default).
        pos_w = float(nonp.get("positioning_weight", 0.0))
        positioning_term = 0.0
        if pos_w != 0.0:
            # Simple proxy: per-step credit proportional to fraction of allies still alive.
            positioning_term = pos_w * (ctx.allies_alive / max(1, ctx.allies_alive + ctx.ally_deaths))

        shaping = hp_term + ally_term + win_term + loss_term + positioning_term
        reward = float(ctx.base_reward) + float(shaping)
        terms = {
            "original": float(ctx.base_reward),
            "hp": float(hp_term),
            "ally": float(ally_term),
            "win": float(win_term + loss_term),   # combined terminal signal
            "positioning": float(positioning_term),
            "shaping_total": float(shaping),
        }
        return reward, terms
    return fn


__all__ = [
    "RewardContext", "register", "resolve", "resolved_params", "available",
]
