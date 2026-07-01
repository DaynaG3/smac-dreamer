"""Checkpoint/agent reconstruction and the deterministic episode driver.

Reuses the *exact* model-reconstruction path from ``scripts/evaluate_multimap.py`` (read
``run_meta.json`` beside the checkpoint -> obs_mode/dims/padding -> build a probe env for the
obs/action spaces -> ``Dreamer`` -> ``_propagate_device`` -> load ``agent_state_dict`` ->
``eval()``). It then drives one deterministic episode on a single-map SMAClite env, capturing
RGB frames and per-step trace records keyed off the env's *executed* actions
(``get_debug_context()["last_executed_action"]``), never the raw policy output.

This module imports torch and the training-side config builder, so it is intentionally NOT
imported by the pure-logic tests. Use ``smacdreamer.visualization.trace`` for those.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# --- Path setup: src + vendored agent/simulator + scripts (for the config builder) -------
ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
    str(ROOT / "scripts"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smacdreamer.visualization import render as _render
from smacdreamer.visualization import trace as _trace


# ---------------------------------------------------------------------------
# Config / checkpoint / run-meta resolution
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load the YAML config with OmegaConf (relative paths resolve against the repo root)."""
    from omegaconf import OmegaConf

    p = pathlib.Path(config_path)
    if not p.is_absolute():
        p = ROOT / config_path
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return OmegaConf.load(str(p))


def resolve_checkpoint(checkpoint_path) -> pathlib.Path:
    """Resolve a checkpoint path (arbitrary folder supported)."""
    p = pathlib.Path(checkpoint_path)
    if not p.is_absolute():
        # try as given (cwd) then relative to repo root
        if not p.exists():
            p = ROOT / checkpoint_path
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return p.resolve()


def resolve_run_meta(checkpoint: pathlib.Path, run_meta_arg=None) -> Optional[pathlib.Path]:
    """Resolve the run_meta.json path: explicit ``--run-meta`` or beside the checkpoint."""
    if run_meta_arg:
        p = pathlib.Path(run_meta_arg)
        if not p.is_absolute() and not p.exists():
            p = ROOT / run_meta_arg
        if not p.exists():
            raise FileNotFoundError(f"--run-meta not found: {run_meta_arg}")
        return p.resolve()
    candidate = pathlib.Path(checkpoint).resolve().parent / "run_meta.json"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Loaded-agent context
# ---------------------------------------------------------------------------

@dataclass
class VizContext:
    """Everything a rollout needs: the loaded agent, dims, padding, and resolved knobs."""

    agent: object
    config: object               # OmegaConf DictConfig (the user --config)
    run_meta: dict
    obs_mode: str
    pad_dims: object             # PaddingDims
    device: str
    gamma: float
    max_episode_steps: int
    obs_space: object
    act_space: object


def load_agent(config_path, checkpoint, run_meta=None, device=None,
               max_episode_steps=None, probe_entry=None, probe_entries=None) -> VizContext:
    """Reconstruct the Dreamer agent exactly like ``evaluate_multimap.py``.

    A probe env (single map, fixed sampler, original reward) supplies the obs/action spaces.
    Pass ``probe_entry`` (or ``probe_entries`` and the first is used) — typically the first
    map you intend to visualise; the spaces depend only on ``pad_dims`` so any map works.
    """
    import torch

    from train_r2dreamer_smaclite_debug import make_config as _make_debug_config
    from train_r2dreamer_smaclite_multimap import _propagate_device
    from dreamer import Dreamer
    from smacdreamer.envs.padding import PaddingDims
    from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env

    cfg = load_config(config_path)
    ckpt_path = resolve_checkpoint(checkpoint)
    meta_path = resolve_run_meta(ckpt_path, run_meta)
    run_meta_dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else {}

    # obs_mode: prefer run_meta (the truth for this checkpoint), fall back to config.
    obs_mode = str(run_meta_dict.get(
        "obs_mode", cfg.observation.mode if cfg.get("observation") else "flat"))
    # Hard requirement: this visualiser supports structured checkpoints only.
    _trace.assert_structured_obs_mode(
        {"obs_mode": obs_mode},
        source=(str(meta_path) if meta_path else "--config (no run_meta.json found)"),
    )

    # Model dims: run_meta first, then config (matches evaluate_multimap).
    def _dim(key):
        v = run_meta_dict.get(key, cfg.get(key))
        if v is None:
            raise ValueError(
                f"cannot resolve model dim {key!r}: not in run_meta.json and not in --config. "
                "Provide a run_meta.json beside the checkpoint or a config carrying it."
            )
        return int(v)

    units, deter = _dim("units"), _dim("deter")
    batch_size, batch_length = _dim("batch_size"), _dim("batch_length")
    imag_horizon = _dim("imag_horizon")

    # Padding MUST come from run_meta for an arbitrary checkpoint folder (structured eval
    # contract). Fall back to config.padding only if explicitly present; never guess.
    pad = run_meta_dict.get("padding")
    if not pad and cfg.get("padding"):
        from omegaconf import OmegaConf
        pad = OmegaConf.to_container(cfg.padding, resolve=True)
    if not pad:
        raise ValueError(
            "Structured visualisation needs padding dims (max_agents/max_enemies/max_actions/"
            "max_obs_size). They were not found in run_meta.json (beside the checkpoint) or in "
            "the config. Point --run-meta at the training run's run_meta.json."
        )
    pad_dims = PaddingDims(
        max_agents=int(pad["max_agents"]), max_enemies=int(pad["max_enemies"]),
        max_actions=int(pad["max_actions"]), max_obs_size=int(pad["max_obs_size"]),
    )

    device = str(device) if device else str(cfg.get("device", "cpu"))
    gamma = float(cfg.get("gamma", 0.997))
    mes = int(max_episode_steps if max_episode_steps is not None
              else cfg.get("max_episode_steps", 200))

    entry = probe_entry or (probe_entries[0] if probe_entries else None)
    if entry is None:
        raise ValueError("load_agent: a probe map entry is required to read obs/action spaces")

    probe = make_smaclite_multimap_env(
        [entry], pad_dims, "fixed", 0, 0, "smaclite_default", {},
        gamma, mes, obs_mode,
    )
    obs_space, act_space = probe.observation_space, probe.action_space
    try:
        probe.close()
    except Exception:
        pass

    config = _make_debug_config(argparse.Namespace(
        steps=1, batch_size=batch_size, batch_length=batch_length,
        units=units, deter=deter, imag_horizon=imag_horizon,
    ))
    _propagate_device(config, device)

    # Rebuild the P0.2 predicted-mask heads (avail_head/alive_head + frozen copies) when the
    # checkpoint was trained with action masking, so the reconstructed model matches the saved
    # state_dict. Mirrors the injection in train_r2dreamer_smaclite_multimap.py; run_meta wins
    # over --config so an arbitrary checkpoint folder reconstructs correctly.
    action_masking = bool(run_meta_dict.get("action_masking", cfg.get("action_masking", False)))
    if action_masking and obs_mode != "structured":
        raise ValueError("action_masking requires observation.mode: structured")
    config.model.action_masking = action_masking
    config.model.mask_threshold = float(
        run_meta_dict.get("mask_threshold", cfg.get("mask_threshold", 0.7)))

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"[visualize] loaded checkpoint {ckpt_path} "
          f"(obs_mode={obs_mode} units={units} deter={deter} device={device})")

    return VizContext(
        agent=agent, config=cfg, run_meta=run_meta_dict, obs_mode=obs_mode,
        pad_dims=pad_dims, device=device, gamma=gamma, max_episode_steps=mes,
        obs_space=obs_space, act_space=act_space,
    )


# ---------------------------------------------------------------------------
# Map resolution
# ---------------------------------------------------------------------------

def resolve_map_entries(cfg, split="validation", maps_dir=None, map_name=None) -> list:
    """Resolve the list of ``MapEntry`` to visualise.

    Priority: ``maps_dir`` (arbitrary folder) overrides the config split. ``map_name`` filters
    to a single named map. The split must be a key under ``cfg.maps`` (validation / blind_iid /
    blind_compositional).
    """
    from smacdreamer.envs.map_discovery import scan_folder_entries

    if maps_dir:
        entries = scan_folder_entries(str(maps_dir))
        source = str(maps_dir)
    else:
        if not cfg.get("maps"):
            raise ValueError(
                "config has no 'maps' block; pass --maps-dir to point at a folder of maps.")
        folder = cfg.maps.get(split)
        if not folder:
            raise ValueError(
                f"--split {split!r} not in cfg.maps; available: {list(cfg.maps.keys())}")
        entries = scan_folder_entries(str(folder))
        source = str(folder)

    if map_name:
        matches = [e for e in entries if e.name == map_name]
        if not matches:
            raise ValueError(
                f"--map-name {map_name!r} not found in {source}. "
                f"Available ({len(entries)}): {[e.name for e in entries][:20]}"
                + (" ..." if len(entries) > 20 else ""))
        return matches
    return entries


def resolve_seeds(cfg, seeds_arg=None) -> list:
    """Resolve eval seeds: ``--seeds`` CLI > cfg.eval/validation seeds > [0]."""
    if seeds_arg:
        return [int(s) for s in str(seeds_arg).split(",") if str(s).strip() != ""]
    from omegaconf import OmegaConf
    for blk in ("eval", "validation"):
        node = cfg.get(blk) if cfg.get(blk) else None
        if node is not None:
            for key in ("fixed_seeds", "seeds"):
                val = node.get(key) if node.get(key) is not None else None
                if val:
                    return [int(s) for s in OmegaConf.to_container(val, resolve=True)]
    return [0]


# ---------------------------------------------------------------------------
# Single-map env construction
# ---------------------------------------------------------------------------

def make_single_map_env(ctx: VizContext, entry, *, capture=False,
                        reward_name="smaclite_default", reward_params=None):
    """Construct one direct (in-process) single-map SMAClite env compatible with R2-Dreamer.

    A ``fixed`` sampler over the single entry means the map never switches, so we can safely
    flip the underlying SMAClite renderer to ``rgb_array`` once for frame capture.
    """
    from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env

    env = make_smaclite_multimap_env(
        [entry], ctx.pad_dims, "fixed", 0, 0, reward_name, reward_params or {},
        ctx.gamma, ctx.max_episode_steps, ctx.obs_mode,
    )
    if capture:
        _render.enable_rgb_render(env)
    return env


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    records: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    frames: List[object] = field(default_factory=list)
    battle_won: bool = False


def _step_record(*, step, map_name, seed, executed, info, is_last) -> dict:
    """Build one per-step JSONL record from executed actions + the env info dict."""
    executed = [int(a) for a in executed]
    focus = _trace.target_focus_score(executed)
    rec = {
        "step": int(step),
        "map": map_name,
        "seed": int(seed),
        "executed_actions": executed,
        "action_labels": _trace.action_labels(executed),
        "reward": float(info.get("log_shaped_reward", 0.0)),
        "original_reward": float(info.get("log_reward_original",
                                          info.get("log_original_env_reward", 0.0))),
        "battle_won": bool(info.get("battle_won", False)),
        "enemies_alive": int(float(info.get("log_enemies_alive", 0.0))),
        "allies_alive": int(float(info.get("log_allies_alive", 0.0))),
        "enemy_hp_damage_this_step": float(info.get("log_enemy_hp_damage_this_step", 0.0)),
        # Episode-end EHP fractions are only meaningful on the terminal step.
        "final_enemy_ehp_frac_if_available": (
            float(info.get("log_final_enemy_ehp_frac", 0.0)) if is_last else None),
        "final_ally_ehp_frac_if_available": (
            float(info.get("log_final_ally_ehp_frac", 0.0)) if is_last else None),
        "target_focus_score": focus,
    }
    return rec


def run_episode(
    ctx: VizContext,
    env,
    *,
    map_name: str,
    seed: int,
    capture_frames: bool = False,
    overlay: bool = True,
    scale: int = 1,
    interactive_window=None,
    low_enemy_ehp_threshold: float = 0.75,
    poor_focus_threshold: float = 0.5,
    min_attack_steps_for_focus: int = 5,
) -> EpisodeResult:
    """Drive one deterministic greedy episode; collect per-step records (+ frames).

    Action selection mirrors ``evaluation.evaluate_episode`` (agent eval mode). Executed
    actions come from the env debug context, so the trace reflects what SMAClite actually ran.
    """
    import torch
    from tensordict import TensorDict

    device = ctx.device
    agent = ctx.agent
    result = EpisodeResult()

    def _frame(step, executed, info, is_last, battle_won):
        if not (capture_frames or interactive_window is not None):
            return
        frame = _render.capture_frame(env, scale=scale)
        if overlay:
            focus = _trace.target_focus_score([int(a) for a in executed]) if executed else None
            lines = _render.build_overlay_lines(
                map_name=map_name, seed=seed, step=step,
                action_labels=_trace.action_labels([int(a) for a in executed]),
                enemies_alive=int(float(info.get("log_enemies_alive", 0.0))),
                allies_alive=int(float(info.get("log_allies_alive", 0.0))),
                enemy_hp_damage_this_step=float(info.get("log_enemy_hp_damage_this_step", 0.0)),
                target_focus_score=focus, done=is_last, battle_won=battle_won,
            )
            frame = _render.draw_overlay(frame, lines)
        if capture_frames:
            result.frames.append(frame)
        if interactive_window is not None:
            interactive_window.show(frame)

    with torch.no_grad():
        obs = env.reset(seed=int(seed))
        state = agent.get_initial_state(1)
        act = state["prev_action"].clone()
        # Initial frame (pre-action). No executed actions yet; reset info is on obs (log_* keys).
        _frame(step=0, executed=[], info=obs, is_last=False, battle_won=False)

        done = False
        length = 0
        last_info: dict = {}
        battle_won = False
        while not done and length <= int(ctx.max_episode_steps) + 1:
            td = TensorDict(
                {k: torch.as_tensor(v).unsqueeze(0).to(device)
                 for k, v in obs.items()},
                batch_size=(1,),
            )
            td["action"] = act
            act, state = agent.act(td, state, eval=True)
            a = act.detach().cpu().numpy().reshape(-1)
            obs, reward, done, info = env.step(a)
            length += 1
            executed = env.get_debug_context().get("last_executed_action", [])
            battle_won = bool(info.get("battle_won", False))
            rec = _step_record(step=length, map_name=map_name, seed=seed,
                               executed=executed, info=info, is_last=done)
            result.records.append(rec)
            last_info = info
            _frame(step=length, executed=executed, info=info, is_last=done,
                   battle_won=battle_won)

    result.battle_won = bool(last_info.get("battle_won", False))
    result.summary = _trace.summarise_episode(
        result.records, map_name=map_name, seed=seed, battle_won=result.battle_won,
        low_enemy_ehp_threshold=low_enemy_ehp_threshold,
        poor_focus_threshold=poor_focus_threshold,
        min_attack_steps_for_focus=min_attack_steps_for_focus,
    )
    return result


__all__ = [
    "ROOT", "VizContext", "EpisodeResult",
    "load_config", "resolve_checkpoint", "resolve_run_meta", "load_agent",
    "resolve_map_entries", "resolve_seeds", "make_single_map_env", "run_episode",
]
