"""Reward-transfer checkpoint loading for continuation training.

When continuing training with a CHANGED reward function, the reward-prediction head, the
critic/value head, the slow target critic, and the return-normalisation statistics are no
longer valid and must be re-initialised. Everything that is reward-agnostic — the encoder,
RSSM/world-model dynamics, the actor, the continuation head, the action-mask heads, and the
latent projector — is transferred from the checkpoint.

This module reads ``agent_state_dict`` from either ``best_val_macro_winrate.pt`` or
``latest.pt`` (any absolute path, including off-repo), filters keys explicitly, validates the
retained set (failing loudly on incompatible/missing layers), and reports exactly what was
loaded vs intentionally reset. Optimizer/scheduler/scaler/return-EMA state is NEVER restored.
"""

from __future__ import annotations

import pathlib

import torch

# Reward-agnostic trainable modules transferred verbatim (top-level state-dict prefixes).
RETAIN_PREFIXES = ("encoder", "rssm", "actor", "cont", "avail_head", "alive_head", "prj")
# Reward/value-dependent modules intentionally left newly-initialised.
RESET_PREFIXES = ("reward", "value", "_slow_value", "return_ema")
# Frozen mirrors share storage with their trainable source and are rebuilt by
# Dreamer.clone_and_freeze() after the partial load, so they are never loaded here.
_FROZEN_MARKER = "_frozen_"


def _top(key: str) -> str:
    return key.split(".", 1)[0]


def validate_resume_args(resume_mode, step_offset, resume_path) -> None:
    """Guard against silently starting a fresh run with continuation settings.

    ``transfer_reward`` and ``weights_only`` load weights from a checkpoint, so they REQUIRE
    ``--resume``. A non-zero ``step_offset`` only makes sense when continuing an existing run, so
    it also requires ``--resume``. Raises ``ValueError`` (clear message) if either is violated.
    """
    if resume_mode in ("transfer_reward", "weights_only") and not resume_path:
        raise ValueError(
            f"resume_mode={resume_mode!r} requires --resume /absolute/path/to/checkpoint.pt"
        )
    if int(step_offset) > 0 and not resume_path:
        raise ValueError(
            f"step_offset={step_offset} was set, but --resume was not provided. "
            "Refusing to start a continuation run from scratch."
        )


def read_agent_state_dict(checkpoint_path) -> dict:
    """Load a checkpoint (absolute path, possibly off-repo) and return its ``agent_state_dict``.

    Accepts both the ``latest.pt`` (full payload) and ``best_val_macro_winrate.pt`` formats —
    both carry ``agent_state_dict``. Fails clearly if the path is missing or the key is absent.
    """
    path = pathlib.Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path} (transfer needs an existing checkpoint; on Kubeflow "
            "this is the persistent-volume path, e.g. /mnt/pvc/checkpoints/r2_650/best_val_macro_winrate.pt)")
    ckpt = torch.load(str(path), map_location="cpu")
    if "agent_state_dict" not in ckpt:
        raise KeyError(f"{path} has no 'agent_state_dict' (keys: {sorted(ckpt)[:8]})")
    return ckpt["agent_state_dict"]


def transfer_reward_load(agent, checkpoint_path, *, verbose: bool = True) -> dict:
    """Load only the reward-agnostic modules from ``checkpoint_path`` into ``agent``.

    Reward/critic/slow-critic/return-EMA stay newly-initialised. Frozen mirrors are rebuilt via
    ``agent.clone_and_freeze()``. Optimizer/scheduler/scaler state is NOT touched. Raises if a
    retained layer is missing from the checkpoint or shape-incompatible.
    """
    src = read_agent_state_dict(checkpoint_path)
    agent_sd = agent.state_dict()

    retained, skipped, incompatible, unexpected = {}, [], [], []
    for k, v in src.items():
        tp = _top(k)
        if tp in RETAIN_PREFIXES and not k.startswith(_FROZEN_MARKER):
            if k not in agent_sd:
                unexpected.append(k)
            elif tuple(agent_sd[k].shape) != tuple(v.shape):
                incompatible.append((k, tuple(v.shape), tuple(agent_sd[k].shape)))
            else:
                retained[k] = v
        else:
            skipped.append(k)   # reset modules + frozen mirrors

    if incompatible:
        lines = "\n".join(f"    {k}: checkpoint {a} vs model {b}" for k, a, b in incompatible)
        raise RuntimeError(
            "transfer_reward: incompatible retained layer(s) — the world model/actor architecture "
            f"does not match the checkpoint:\n{lines}\n"
            "Architectures must match for transfer (only reward/value modules may differ).")

    expected_retained = [k for k in agent_sd
                         if _top(k) in RETAIN_PREFIXES and not k.startswith(_FROZEN_MARKER)]
    missing_retained = [k for k in expected_retained if k not in retained]
    if missing_retained:
        raise RuntimeError(
            f"transfer_reward: {len(missing_retained)} retained key(s) absent from the checkpoint "
            f"(cannot transfer a partial world model): {missing_retained[:8]}"
            + (" ..." if len(missing_retained) > 8 else ""))

    # Explicit partial load (NOT a silent strict=False): we validated the retained set above and
    # report the skipped groups below.
    result = agent.load_state_dict(retained, strict=False)
    unexpected_to_model = [k for k in getattr(result, "unexpected_keys", [])]
    # Rebuild frozen mirrors from the now-loaded trainable weights; reset heads stay fresh.
    if hasattr(agent, "clone_and_freeze"):
        agent.clone_and_freeze()

    if verbose:
        print(f"[transfer_reward] checkpoint: {checkpoint_path}")
        print(f"[transfer_reward] loaded {len(retained)} retained keys "
              f"(modules: {', '.join(RETAIN_PREFIXES)})")
        print(f"[transfer_reward] intentionally RESET (newly initialised): reward head, "
              f"critic/value head, slow target critic, return-EMA")
        print(f"[transfer_reward] skipped {len(skipped)} keys (reset modules + frozen mirrors)")
        print(f"[transfer_reward] NOT restored: optimizer / scheduler / AMP scaler / return stats")
        if unexpected:
            print(f"[transfer_reward] WARNING: {len(unexpected)} checkpoint key(s) not present in "
                  f"the model (ignored): {unexpected[:6]}")
        if unexpected_to_model:
            print(f"[transfer_reward] WARNING: {len(unexpected_to_model)} loaded key(s) unexpected "
                  f"by the model: {unexpected_to_model[:6]}")
    return {
        "loaded": len(retained), "skipped": len(skipped),
        "unexpected_source": unexpected, "missing_retained": missing_retained,
        "incompatible": incompatible,
    }


def load_weights_only(agent, checkpoint_path, *, verbose: bool = True) -> dict:
    """Load the complete ``agent_state_dict`` but restore no optimizer/training state."""
    src = read_agent_state_dict(checkpoint_path)
    result = agent.load_state_dict(src, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if hasattr(agent, "clone_and_freeze"):
        agent.clone_and_freeze()
    if verbose:
        print(f"[weights_only] loaded full agent_state_dict from {checkpoint_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)}); no optimizer/training state")
    return {"missing": missing, "unexpected": unexpected}


__all__ = [
    "RETAIN_PREFIXES", "RESET_PREFIXES", "read_agent_state_dict",
    "transfer_reward_load", "load_weights_only", "validate_resume_args",
]