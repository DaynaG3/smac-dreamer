#!/usr/bin/env python3
"""
Qualitative sanity-check for a learned world model on saved SMAC/SMACLite episodes.

What this script produces per episode:
  1. NPZ/data-format inspection so you can see how the episode was collected/stored.
  2. Decoder reconstruction:       x_t -> encoder -> decoder -> xhat_t
  3. Teacher-forced one-step pred: x_t, a_t -> zhat_{t+1} -> decoder -> xhat_{t+1}
  4. Open-loop rollout:           x_0, a_0...a_T -> zhat_1...zhat_T -> decoder
  5. CSV metrics + PNG plots for visual comparison.

Important:
  This script can auto-run only if your checkpoint contains actual torch modules
  or a model object with common method names such as encode/decode/predict_next.

  If your checkpoint only contains state_dicts, pass a small factory with:

      --factory path.to.module:function_name

  The factory should return either:
      - a torch.nn.Module with encode/decode/predict_next-like methods, or
      - a dict with keys: encoder, decoder, dynamics/predictor/model

Example:
  python tools/sanity_decode_rollout.py \
    --checkpoint runs/rnn_seqmem_exp33_dreamer_7ep_v2_clean/checkpoint.pt \
    --train-episode data/r2_general_2100_full/train/shard_02/r2g_train_0043.npz \
    --val-episode data/r2_general_2100_full/validation/shard_03/r2g_validation_1284.npz \
    --out-dir sanity_outputs/exp33_decode_rollout \
    --device cuda \
    --max-steps 120 \
    --flatten-state \
    --flatten-action

If auto key inference chooses the wrong arrays, override with:
    --state-key YOUR_STATE_KEY --action-key YOUR_ACTION_KEY
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch. Install/import torch first.") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires matplotlib for plots. Install/import matplotlib first.") from e


# --------------------------------------------------------------------------------------
# Candidate names. Override from CLI if auto inference picks the wrong one.
# --------------------------------------------------------------------------------------

STATE_KEY_CANDIDATES = [
    "states",
    "state",
    "global_states",
    "global_state",
    "obs",
    "obss",
    "observations",
    "observation",
    "features",
    "x",
]

ACTION_KEY_CANDIDATES = [
    "actions",
    "action",
    "acts",
    "a",
]

REWARD_KEY_CANDIDATES = ["rewards", "reward", "r"]
DONE_KEY_CANDIDATES = ["dones", "done", "terminated", "terminals", "terminal", "is_done"]
META_KEY_HINTS = [
    "map",
    "config",
    "seed",
    "episode",
    "return",
    "win",
    "collector",
    "env",
    "scenario",
    "difficulty",
]


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: Union[str, Path], text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def safe_name(path: Union[str, Path]) -> str:
    p = Path(path)
    return p.stem.replace(" ", "_").replace("/", "_")


def maybe_scalar(x: Any) -> str:
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            return repr(x.item())
        if isinstance(x, np.ndarray) and x.size == 1:
            return repr(x.reshape(-1)[0].item())
    except Exception:
        pass
    return ""


def array_stats(arr: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.dtype.kind in "biufc" and arr.size > 0:
        flat = arr.reshape(-1)
        finite = np.isfinite(flat) if arr.dtype.kind in "fc" else np.ones_like(flat, dtype=bool)
        if finite.any():
            vals = flat[finite]
            out.update(
                {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "nan_count": int(np.isnan(flat).sum()) if arr.dtype.kind in "fc" else 0,
                    "inf_count": int(np.isinf(flat).sum()) if arr.dtype.kind in "fc" else 0,
                }
            )
    else:
        out["scalar_preview"] = maybe_scalar(arr)
    return out


def describe_npz(npz_path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], str]:
    loaded = np.load(npz_path, allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}

    lines: List[str] = []
    lines.append(f"File: {npz_path}")
    lines.append(f"Keys ({len(data)}): {list(data.keys())}")
    lines.append("")
    lines.append("Array summary:")

    for k, arr in data.items():
        stats = array_stats(arr)
        lines.append(f"  - {k}")
        lines.append(f"      shape: {stats.get('shape')}")
        lines.append(f"      dtype: {stats.get('dtype')}")
        if "min" in stats:
            lines.append(
                "      stats: "
                f"min={stats['min']:.6g}, max={stats['max']:.6g}, "
                f"mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
                f"nan={stats['nan_count']}, inf={stats['inf_count']}"
            )
        if stats.get("scalar_preview"):
            lines.append(f"      scalar_preview: {stats['scalar_preview']}")

        # Tiny content preview for metadata-ish arrays only.
        lower = k.lower()
        if any(h in lower for h in META_KEY_HINTS) or arr.dtype.kind in "OUS":
            try:
                preview = arr.item() if arr.shape == () else arr.reshape(-1)[:5].tolist()
                lines.append(f"      preview: {repr(preview)[:500]}")
            except Exception:
                pass

    lines.append("")
    lines.append("Likely collection/metadata fields:")
    found_meta = False
    for k, arr in data.items():
        lower = k.lower()
        if any(h in lower for h in META_KEY_HINTS):
            found_meta = True
            try:
                preview = arr.item() if arr.shape == () else arr.reshape(-1)[:10].tolist()
            except Exception:
                preview = "<unpreviewable>"
            lines.append(f"  - {k}: {repr(preview)[:800]}")
    if not found_meta:
        lines.append("  - No obvious metadata keys found. This file may only contain preprocessed tensors.")

    return data, "\n".join(lines) + "\n"


def choose_key(data: Mapping[str, np.ndarray], candidates: Sequence[str], override: str, kind: str) -> str:
    if override != "auto":
        if override not in data:
            raise KeyError(f"Requested --{kind}-key={override!r}, but keys are {list(data.keys())}")
        return override

    keys_lower = {k.lower(): k for k in data.keys()}
    for cand in candidates:
        if cand.lower() in keys_lower:
            return keys_lower[cand.lower()]

    # Fallback heuristic: choose the longest numeric time-major array for state,
    # and an integer-ish/numeric time-major array for action.
    numeric = []
    for k, arr in data.items():
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim >= 1 and arr.dtype.kind in "biufc" and arr.shape[0] >= 2:
            numeric.append((k, arr))

    if kind == "state" and numeric:
        # Prefer higher-dimensional / wider arrays.
        numeric.sort(key=lambda kv: (kv[1].ndim, int(np.prod(kv[1].shape[1:])) if kv[1].ndim > 1 else 1), reverse=True)
        return numeric[0][0]

    if kind == "action" and numeric:
        # Prefer action-like narrow arrays, often integer/discrete.
        scored = []
        for k, arr in numeric:
            width = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            int_bonus = 1 if arr.dtype.kind in "biu" else 0
            name_bonus = 1 if "act" in k.lower() or k.lower() == "a" else 0
            scored.append((name_bonus, int_bonus, -width, k))
        scored.sort(reverse=True)
        return scored[0][-1]

    raise KeyError(f"Could not infer {kind} key from keys: {list(data.keys())}. Pass --{kind}-key explicitly.")


def flatten_time_major(arr: np.ndarray) -> np.ndarray:
    if arr.ndim <= 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def align_state_action_lengths(states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Make states [T,...], actions [T-1,...] or [T,...] usable."""
    lines = []
    t_state = states.shape[0]
    t_act = actions.shape[0]
    if t_act == t_state:
        lines.append(f"actions length equals states length ({t_act}); using actions[:-1] for transition x_t -> x_(t+1).")
        actions = actions[:-1]
    elif t_act == t_state - 1:
        lines.append(f"actions length is states length - 1 ({t_act}); using actions as-is.")
    elif t_act > t_state - 1:
        lines.append(f"actions longer than needed ({t_act} vs {t_state-1}); truncating to states length - 1.")
        actions = actions[: t_state - 1]
    else:
        lines.append(
            f"WARNING: actions shorter than states transitions ({t_act} vs {t_state-1}); truncating states to {t_act+1}."
        )
        states = states[: t_act + 1]
    return states, actions, "\n".join(lines)


def one_hot_if_requested(actions: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return actions
    if actions.dtype.kind not in "biu":
        raise ValueError("--one-hot-actions was set, but action array is not integer typed.")
    a = actions.astype(np.int64)
    if np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"Action values must be in [0, {n}); got min={a.min()}, max={a.max()}.")
    return np.eye(n, dtype=np.float32)[a]


def to_torch(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if x.dtype.kind in "biu":
        # Most world models expect float actions/features. Keep as float unless your factory handles int actions.
        return torch.as_tensor(x, device=device).to(dtype)
    return torch.as_tensor(x, device=device).to(dtype)


def detach_cpu_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected tensor/ndarray, got {type(x)}")


def batchify(x: torch.Tensor) -> torch.Tensor:
    # Most modules expect [B, ...]. One state/action at a time gets batch dim 1.
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x.unsqueeze(0)


def unbatch(x: Any) -> Any:
    if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    return x


def first_tensor_like(obj: Any) -> Any:
    """Extract a useful tensor from common tuple/list/dict outputs."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj)
    if isinstance(obj, Mapping):
        preferred = [
            "z",
            "latent",
            "embedding",
            "emb",
            "h",
            "state",
            "state_pred",
            "next_state",
            "pred",
            "prediction",
            "x",
            "recon",
            "reconstruction",
            "obs",
            "observation",
            "mean",
        ]
        for k in preferred:
            if k in obj:
                return first_tensor_like(obj[k])
        for v in obj.values():
            try:
                return first_tensor_like(v)
            except Exception:
                continue
    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                return first_tensor_like(v)
            except Exception:
                continue
    raise TypeError(f"Could not extract tensor from object of type {type(obj)}")


def load_factory(factory_spec: str) -> Callable[..., Any]:
    """Load 'module:function' or '/path/to/file.py:function'."""
    if ":" not in factory_spec:
        raise ValueError("--factory must look like module:function or /path/to/file.py:function")
    module_spec, fn_name = factory_spec.split(":", 1)

    if module_spec.endswith(".py") or os.path.exists(module_spec):
        import importlib.util

        path = Path(module_spec).resolve()
        module_name = path.stem + "_sanity_factory"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import factory file {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_spec)

    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Factory {factory_spec} is not callable")
    return fn


# --------------------------------------------------------------------------------------
# Model bundle / adapter
# --------------------------------------------------------------------------------------


@dataclass
class AutoBundle:
    model: Optional[torch.nn.Module] = None
    encoder: Optional[torch.nn.Module] = None
    decoder: Optional[torch.nn.Module] = None
    dynamics: Optional[torch.nn.Module] = None
    predictor: Optional[torch.nn.Module] = None
    device: torch.device = torch.device("cpu")
    verbose: bool = True

    def eval(self) -> "AutoBundle":
        for m in [self.model, self.encoder, self.decoder, self.dynamics, self.predictor]:
            if isinstance(m, torch.nn.Module):
                m.to(self.device)
                m.eval()
        return self

    def _call_method_candidates(self, obj: Any, names: Sequence[str], args: Sequence[Any]) -> Any:
        last_err: Optional[Exception] = None
        for name in names:
            if obj is None or not hasattr(obj, name):
                continue
            fn = getattr(obj, name)
            if not callable(fn):
                continue
            try:
                return fn(*args)
            except TypeError as e:
                last_err = e
                # Try with batchified tensor args if not already batchified.
                try:
                    bargs = [batchify(a) if torch.is_tensor(a) and (a.ndim == 1) else a for a in args]
                    return fn(*bargs)
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Tried methods {names}, but all failed. Last error: {last_err}") from last_err
        raise AttributeError(f"No callable methods among {names} on {type(obj)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # Prefer explicit encoder module.
        if self.encoder is not None:
            out = self.encoder(batchify(x))
            return unbatch(first_tensor_like(out)).to(self.device)

        if self.model is not None:
            out = self._call_method_candidates(
                self.model,
                ["encode", "encode_obs", "encode_state", "embed", "embed_obs", "encoder_forward"],
                [x],
            )
            return unbatch(first_tensor_like(out)).to(self.device)

            # Unreachable, but keeps static checkers happy.
        raise RuntimeError("No encoder/model available. Use --factory or save encoder in checkpoint.")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        if self.decoder is not None:
            out = self.decoder(batchify(z))
            return unbatch(first_tensor_like(out)).to(self.device)

        if self.model is not None:
            out = self._call_method_candidates(
                self.model,
                ["decode", "decode_state", "decode_obs", "reconstruct", "decoder_forward"],
                [z],
            )
            return unbatch(first_tensor_like(out)).to(self.device)

        raise RuntimeError("No decoder/model available. Use --factory or save decoder in checkpoint.")

    def predict_next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device)
        a = a.to(self.device)
        if self.dynamics is not None:
            # Dynamics usually takes (z, a).
            out = self.dynamics(batchify(z), batchify(a))
            return unbatch(first_tensor_like(out)).to(self.device)

        if self.predictor is not None:
            # Predictor usually takes (z, a). If your predictor needs context windows,
            # use --factory and wrap that logic inside the returned object.
            out = self.predictor(batchify(z), batchify(a))
            return unbatch(first_tensor_like(out)).to(self.device)

        if self.model is not None:
            out = self._call_method_candidates(
                self.model,
                [
                    "predict_next",
                    "predict_next_latent",
                    "transition",
                    "dynamics_step",
                    "imagine_step",
                    "prior_step",
                    "step",
                    "predictor_forward",
                ],
                [z, a],
            )
            return unbatch(first_tensor_like(out)).to(self.device)

        raise RuntimeError("No dynamics/predictor/model available. Use --factory or save predictor in checkpoint.")


def normalize_bundle(obj: Any, device: torch.device, verbose: bool = True) -> AutoBundle:
    if isinstance(obj, AutoBundle):
        obj.device = device
        return obj.eval()

    if isinstance(obj, torch.nn.Module):
        return AutoBundle(model=obj, device=device, verbose=verbose).eval()

    if isinstance(obj, Mapping):
        lower_to_key = {str(k).lower(): k for k in obj.keys()}

        def get_any(names: Sequence[str]) -> Optional[Any]:
            for n in names:
                if n in obj:
                    return obj[n]
                if n.lower() in lower_to_key:
                    return obj[lower_to_key[n.lower()]]
            return None

        # If a dict has a model module, use it.
        model = get_any(["model", "world_model", "wm", "net", "network"])
        encoder = get_any(["encoder", "enc", "online_encoder"])
        decoder = get_any(["decoder", "dec", "state_decoder", "obs_decoder"])
        dynamics = get_any(["dynamics", "transition", "rssm", "seqmem", "rnn", "world_dynamics"])
        predictor = get_any(["predictor", "pred", "jepa_predictor"])

        modules = [model, encoder, decoder, dynamics, predictor]
        if any(isinstance(m, torch.nn.Module) for m in modules):
            return AutoBundle(
                model=model if isinstance(model, torch.nn.Module) else None,
                encoder=encoder if isinstance(encoder, torch.nn.Module) else None,
                decoder=decoder if isinstance(decoder, torch.nn.Module) else None,
                dynamics=dynamics if isinstance(dynamics, torch.nn.Module) else None,
                predictor=predictor if isinstance(predictor, torch.nn.Module) else None,
                device=device,
                verbose=verbose,
            ).eval()

    raise TypeError(
        "Could not turn factory/checkpoint object into a runnable model bundle. "
        "Return a torch.nn.Module, AutoBundle, or dict with modules named encoder/decoder/dynamics/predictor/model."
    )


def summarize_checkpoint_object(obj: Any) -> str:
    lines: List[str] = []
    lines.append(f"Checkpoint object type: {type(obj)}")
    if isinstance(obj, Mapping):
        lines.append(f"Top-level keys ({len(obj)}): {list(obj.keys())[:100]}")
        for k, v in obj.items():
            if isinstance(v, torch.nn.Module):
                lines.append(f"  - {k}: torch module {type(v)}")
            elif torch.is_tensor(v):
                lines.append(f"  - {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, Mapping):
                # State dicts can be huge; show only first few.
                subkeys = list(v.keys())[:20]
                lines.append(f"  - {k}: dict with {len(v)} keys; first keys={subkeys}")
            else:
                lines.append(f"  - {k}: {type(v)}")
    elif isinstance(obj, torch.nn.Module):
        lines.append(str(obj))
        methods = [m for m in dir(obj) if not m.startswith("_")]
        useful = [m for m in methods if any(s in m.lower() for s in ["enc", "dec", "pred", "roll", "step", "dyn", "trans", "imagine"])]
        lines.append(f"Useful-looking methods: {useful[:100]}")
    return "\n".join(lines) + "\n"


def load_bundle_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    factory: str = "",
    out_dir: Optional[Path] = None,
) -> AutoBundle:
    if factory:
        fn = load_factory(factory)
        # Support factories with either (checkpoint_path, device), (checkpoint_path), or no args.
        try:
            obj = fn(str(checkpoint_path), device)
        except TypeError:
            try:
                obj = fn(str(checkpoint_path))
            except TypeError:
                obj = fn()
        return normalize_bundle(obj, device=device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if out_dir is not None:
        write_text(out_dir / "checkpoint_summary.txt", summarize_checkpoint_object(ckpt))

    try:
        return normalize_bundle(ckpt, device=device)
    except Exception as e:
        summary = summarize_checkpoint_object(ckpt)
        msg = f"""
Could not auto-build a runnable model from the checkpoint.

This usually means checkpoint.pt contains only state_dicts, not actual model objects.
You need to provide --factory module:function that builds your exact exp33 model architecture,
loads the state_dict, and returns a runnable object/dict.

Checkpoint summary:
{summary}

Original error:
{repr(e)}
"""
        raise RuntimeError(textwrap.dedent(msg)) from e


# --------------------------------------------------------------------------------------
# Sanity-check core
# --------------------------------------------------------------------------------------


@dataclass
class EpisodeRunResult:
    episode_name: str
    state_key: str
    action_key: str
    states_model_np: np.ndarray
    actions_model_np: np.ndarray
    recon_np: np.ndarray
    one_step_np: np.ndarray
    rollout_np: np.ndarray
    metrics: Dict[str, Any]


def mse_by_step(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    n = min(p.shape[0], t.shape[0])
    return np.mean((p[:n] - t[:n]) ** 2, axis=1)


def mae_by_step(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    n = min(p.shape[0], t.shape[0])
    return np.mean(np.abs(p[:n] - t[:n]), axis=1)


def mse_by_feature(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    n = min(p.shape[0], t.shape[0])
    return np.mean((p[:n] - t[:n]) ** 2, axis=0)


def run_episode_sanity(
    bundle: AutoBundle,
    episode_path: Union[str, Path],
    out_dir: Path,
    state_key_override: str,
    action_key_override: str,
    max_steps: int,
    flatten_state: bool,
    flatten_action: bool,
    one_hot_actions: int,
    device: torch.device,
) -> EpisodeRunResult:
    ep_name = safe_name(episode_path)
    ep_dir = ensure_dir(out_dir / ep_name)

    data, desc = describe_npz(episode_path)
    write_text(ep_dir / "episode_format.txt", desc)

    state_key = choose_key(data, STATE_KEY_CANDIDATES, state_key_override, "state")
    action_key = choose_key(data, ACTION_KEY_CANDIDATES, action_key_override, "action")

    states_raw = np.asarray(data[state_key])
    actions_raw = np.asarray(data[action_key])

    states_raw, actions_raw, align_msg = align_state_action_lengths(states_raw, actions_raw)
    if max_steps > 0:
        keep_states = min(states_raw.shape[0], max_steps)
        states_raw = states_raw[:keep_states]
        actions_raw = actions_raw[: max(0, keep_states - 1)]

    actions_raw = one_hot_if_requested(actions_raw, one_hot_actions)

    states_model = flatten_time_major(states_raw) if flatten_state else states_raw
    actions_model = flatten_time_major(actions_raw) if flatten_action else actions_raw

    write_text(
        ep_dir / "chosen_keys.txt",
        "\n".join(
            [
                f"episode_path: {episode_path}",
                f"state_key: {state_key}",
                f"action_key: {action_key}",
                f"states_raw_shape: {states_raw.shape}",
                f"actions_raw_shape: {actions_raw.shape}",
                f"states_model_shape: {states_model.shape}",
                f"actions_model_shape: {actions_model.shape}",
                f"flatten_state: {flatten_state}",
                f"flatten_action: {flatten_action}",
                f"one_hot_actions: {one_hot_actions}",
                "",
                align_msg,
            ]
        )
        + "\n",
    )

    states_t = to_torch(states_model, device=device, dtype=torch.float32)
    actions_t = to_torch(actions_model, device=device, dtype=torch.float32)

    z_true: List[torch.Tensor] = []
    recon: List[torch.Tensor] = []
    one_step: List[torch.Tensor] = []
    rollout: List[torch.Tensor] = []

    with torch.no_grad():
        for t in range(states_t.shape[0]):
            z = bundle.encode(states_t[t])
            x_rec = bundle.decode(z)
            z_true.append(z)
            recon.append(x_rec)

        for t in range(actions_t.shape[0]):
            z_pred_next = bundle.predict_next(z_true[t], actions_t[t])
            x_pred_next = bundle.decode(z_pred_next)
            one_step.append(x_pred_next)

        z_roll = z_true[0]
        for t in range(actions_t.shape[0]):
            z_roll = bundle.predict_next(z_roll, actions_t[t])
            x_roll = bundle.decode(z_roll)
            rollout.append(x_roll)

    recon_np = detach_cpu_np(torch.stack([r.reshape(-1) for r in recon], dim=0))
    one_step_np = detach_cpu_np(torch.stack([p.reshape(-1) for p in one_step], dim=0))
    rollout_np = detach_cpu_np(torch.stack([p.reshape(-1) for p in rollout], dim=0))
    states_model_np = states_model.reshape(states_model.shape[0], -1).astype(np.float32, copy=False)
    actions_model_np = actions_model.reshape(actions_model.shape[0], -1).astype(np.float32, copy=False)

    gt_recon = states_model_np
    gt_next = states_model_np[1 : 1 + one_step_np.shape[0]]

    # Align prediction width if decoder output has slightly different shape.
    def width_align(a: np.ndarray, b: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray, str]:
        if a.shape[1] == b.shape[1]:
            return a, b, ""
        w = min(a.shape[1], b.shape[1])
        msg = f"WARNING: width mismatch for {label}: pred width={a.shape[1]}, target width={b.shape[1]}; comparing first {w} dims."
        return a[:, :w], b[:, :w], msg

    recon_cmp, gt_recon_cmp, msg1 = width_align(recon_np, gt_recon, "reconstruction")
    one_cmp, gt_next_cmp1, msg2 = width_align(one_step_np, gt_next, "one-step")
    roll_cmp, gt_next_cmp2, msg3 = width_align(rollout_np, gt_next, "rollout")
    write_text(ep_dir / "shape_warnings.txt", "\n".join([m for m in [msg1, msg2, msg3] if m]) + "\n")

    metrics: Dict[str, Any] = {
        "episode": ep_name,
        "state_key": state_key,
        "action_key": action_key,
        "num_states": int(states_model_np.shape[0]),
        "num_actions": int(actions_model_np.shape[0]),
        "state_dim_compared": int(gt_recon_cmp.shape[1]),
        "recon_mse_mean": float(np.mean(mse_by_step(recon_cmp, gt_recon_cmp))),
        "recon_mae_mean": float(np.mean(mae_by_step(recon_cmp, gt_recon_cmp))),
        "one_step_mse_mean": float(np.mean(mse_by_step(one_cmp, gt_next_cmp1))),
        "one_step_mae_mean": float(np.mean(mae_by_step(one_cmp, gt_next_cmp1))),
        "rollout_mse_mean": float(np.mean(mse_by_step(roll_cmp, gt_next_cmp2))),
        "rollout_mae_mean": float(np.mean(mae_by_step(roll_cmp, gt_next_cmp2))),
        "rollout_mse_first": float(mse_by_step(roll_cmp, gt_next_cmp2)[0]) if roll_cmp.shape[0] else math.nan,
        "rollout_mse_last": float(mse_by_step(roll_cmp, gt_next_cmp2)[-1]) if roll_cmp.shape[0] else math.nan,
    }

    np.savez_compressed(
        ep_dir / "decoded_predictions.npz",
        states_model=states_model_np,
        actions_model=actions_model_np,
        recon=recon_np,
        one_step=one_step_np,
        rollout=rollout_np,
        gt_next=gt_next,
    )

    write_metrics_csv(ep_dir / "metrics_summary.csv", [metrics])
    plot_episode(ep_dir, states_model_np, recon_np, one_step_np, rollout_np, feature_plots=16)

    return EpisodeRunResult(
        episode_name=ep_name,
        state_key=state_key,
        action_key=action_key,
        states_model_np=states_model_np,
        actions_model_np=actions_model_np,
        recon_np=recon_np,
        one_step_np=one_step_np,
        rollout_np=rollout_np,
        metrics=metrics,
    )


def write_metrics_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_episode(
    ep_dir: Path,
    states: np.ndarray,
    recon: np.ndarray,
    one_step: np.ndarray,
    rollout: np.ndarray,
    feature_plots: int = 16,
) -> None:
    gt = states
    gt_next = states[1 : 1 + min(one_step.shape[0], rollout.shape[0])]

    def align_width(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = min(pred.shape[1], target.shape[1])
        return pred[:, :w], target[:, :w]

    recon_cmp, gt_cmp = align_width(recon, gt)
    one_cmp, gt_next_one = align_width(one_step, states[1 : 1 + one_step.shape[0]])
    roll_cmp, gt_next_roll = align_width(rollout, states[1 : 1 + rollout.shape[0]])

    # Plot MSE over time.
    plt.figure(figsize=(10, 5))
    plt.plot(mse_by_step(recon_cmp, gt_cmp), label="decoder reconstruction")
    plt.plot(np.arange(1, 1 + one_cmp.shape[0]), mse_by_step(one_cmp, gt_next_one), label="one-step prediction")
    plt.plot(np.arange(1, 1 + roll_cmp.shape[0]), mse_by_step(roll_cmp, gt_next_roll), label="open-loop rollout")
    plt.xlabel("timestep")
    plt.ylabel("MSE")
    plt.title("Prediction error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ep_dir / "mse_over_time.png", dpi=160)
    plt.close()

    # Plot MAE over time.
    plt.figure(figsize=(10, 5))
    plt.plot(mae_by_step(recon_cmp, gt_cmp), label="decoder reconstruction")
    plt.plot(np.arange(1, 1 + one_cmp.shape[0]), mae_by_step(one_cmp, gt_next_one), label="one-step prediction")
    plt.plot(np.arange(1, 1 + roll_cmp.shape[0]), mae_by_step(roll_cmp, gt_next_roll), label="open-loop rollout")
    plt.xlabel("timestep")
    plt.ylabel("MAE")
    plt.title("Prediction absolute error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ep_dir / "mae_over_time.png", dpi=160)
    plt.close()

    # Error heatmap for open-loop rollout.
    if roll_cmp.shape[0] > 0:
        err = np.abs(roll_cmp - gt_next_roll)
        plt.figure(figsize=(12, 6))
        plt.imshow(err.T, aspect="auto")
        plt.xlabel("rollout step")
        plt.ylabel("flattened state feature index")
        plt.title("Open-loop absolute error heatmap")
        plt.colorbar(label="abs error")
        plt.tight_layout()
        plt.savefig(ep_dir / "open_loop_abs_error_heatmap.png", dpi=160)
        plt.close()

    # Top features by open-loop MSE.
    if roll_cmp.shape[0] > 0:
        feat_mse = mse_by_feature(roll_cmp, gt_next_roll)
        top = np.argsort(feat_mse)[::-1][: min(feature_plots, feat_mse.shape[0])]
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(top)), feat_mse[top])
        plt.xticks(np.arange(len(top)), [str(i) for i in top], rotation=45)
        plt.xlabel("flattened feature index")
        plt.ylabel("open-loop feature MSE")
        plt.title("Worst predicted features in open-loop rollout")
        plt.tight_layout()
        plt.savefig(ep_dir / "top_open_loop_feature_mse.png", dpi=160)
        plt.close()

        # Save a few actual-vs-pred time series for the worst features.
        feature_dir = ensure_dir(ep_dir / "feature_timeseries")
        for idx in top[: min(8, len(top))]:
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(gt_next_roll.shape[0]), gt_next_roll[:, idx], label="actual next state")
            plt.plot(np.arange(one_cmp.shape[0]), one_cmp[:, idx], label="one-step prediction")
            plt.plot(np.arange(roll_cmp.shape[0]), roll_cmp[:, idx], label="open-loop rollout")
            plt.xlabel("transition step")
            plt.ylabel(f"feature {idx}")
            plt.title(f"Feature {idx}: actual vs decoded predictions")
            plt.legend()
            plt.tight_layout()
            plt.savefig(feature_dir / f"feature_{idx:04d}.png", dpi=160)
            plt.close()

    # Side-by-side vector images at selected timesteps.
    # This is not semantically pretty, but it immediately reveals nonsense scale/drift.
    n_steps = min(gt_next.shape[0], one_step.shape[0], rollout.shape[0])
    if n_steps > 0:
        selected = sorted(set([0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1]))
        for t in selected:
            actual = gt_next[t]
            one = one_step[t, : actual.shape[0]] if one_step.shape[1] >= actual.shape[0] else one_step[t]
            roll = rollout[t, : actual.shape[0]] if rollout.shape[1] >= actual.shape[0] else rollout[t]
            w = min(actual.shape[0], one.shape[0], roll.shape[0])
            mat = np.stack([actual[:w], one[:w], roll[:w]], axis=0)
            plt.figure(figsize=(12, 3))
            plt.imshow(mat, aspect="auto")
            plt.yticks([0, 1, 2], ["actual", "one-step", "open-loop"])
            plt.xlabel("flattened state feature index")
            plt.title(f"Decoded comparison at transition step {t}")
            plt.colorbar(label="value")
            plt.tight_layout()
            plt.savefig(ep_dir / f"state_vector_compare_t{t:04d}.png", dpi=160)
            plt.close()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run decoded prediction sanity checks for a JEPA/world-model checkpoint.",
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    p.add_argument("--train-episode", type=str, required=True, help="Path to a train .npz episode")
    p.add_argument("--val-episode", type=str, required=True, help="Path to a validation .npz episode")
    p.add_argument("--out-dir", type=str, default="sanity_outputs/decode_rollout", help="Output directory")
    p.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, or cpu")
    p.add_argument("--factory", type=str, default="", help="Optional factory: module:function or /path/to/file.py:function")
    p.add_argument("--state-key", type=str, default="auto", help="NPZ key for state/obs array")
    p.add_argument("--action-key", type=str, default="auto", help="NPZ key for action array")
    p.add_argument("--max-steps", type=int, default=200, help="Max states to evaluate per episode; <=0 means full episode")
    p.add_argument("--one-hot-actions", type=int, default=0, help="If >0, one-hot integer actions with this number of actions")
    p.add_argument("--flatten-state", action=argparse.BooleanOptionalAction, default=True, help="Flatten state trailing dims before model")
    p.add_argument("--flatten-action", action=argparse.BooleanOptionalAction, default=True, help="Flatten action trailing dims before model")
    p.add_argument("--inspect-only", action="store_true", help="Only inspect NPZ/checkpoint format; do not run model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if str(device) != args.device:
        print(f"[warn] requested device={args.device}, but cuda is unavailable; using cpu")

    print(f"[info] output directory: {out_dir}")
    print(f"[info] device: {device}")

    # Always inspect both episodes first.
    collection_lines: List[str] = []
    for split_name, ep_path in [("train", args.train_episode), ("validation", args.val_episode)]:
        data, desc = describe_npz(ep_path)
        ep_out = ensure_dir(out_dir / f"inspect_{split_name}_{safe_name(ep_path)}")
        write_text(ep_out / "episode_format.txt", desc)
        collection_lines.append(f"===== {split_name.upper()} EPISODE: {ep_path} =====")
        collection_lines.append(desc)
        collection_lines.append("")
    write_text(out_dir / "data_collection_summary.txt", "\n".join(collection_lines))
    print(f"[info] wrote NPZ/data collection inspection to {out_dir / 'data_collection_summary.txt'}")

    # Checkpoint summary is useful even in inspect-only mode.
    try:
        ckpt_obj = torch.load(args.checkpoint, map_location=device)
        write_text(out_dir / "checkpoint_summary.txt", summarize_checkpoint_object(ckpt_obj))
        del ckpt_obj
        print(f"[info] wrote checkpoint summary to {out_dir / 'checkpoint_summary.txt'}")
    except Exception as e:
        write_text(out_dir / "checkpoint_summary_error.txt", repr(e) + "\n")
        print(f"[warn] could not load checkpoint for summary: {e}")

    if args.inspect_only:
        print("[done] inspect-only mode complete")
        return

    bundle = load_bundle_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        factory=args.factory,
        out_dir=out_dir,
    )

    results: List[EpisodeRunResult] = []
    for split_name, ep_path in [("train", args.train_episode), ("validation", args.val_episode)]:
        print(f"[info] running decoded sanity check on {split_name}: {ep_path}")
        split_out = ensure_dir(out_dir / split_name)
        res = run_episode_sanity(
            bundle=bundle,
            episode_path=ep_path,
            out_dir=split_out,
            state_key_override=args.state_key,
            action_key_override=args.action_key,
            max_steps=args.max_steps,
            flatten_state=args.flatten_state,
            flatten_action=args.flatten_action,
            one_hot_actions=args.one_hot_actions,
            device=device,
        )
        results.append(res)
        print(
            "[info] "
            f"{split_name} metrics: recon_mse={res.metrics['recon_mse_mean']:.6g}, "
            f"one_step_mse={res.metrics['one_step_mse_mean']:.6g}, "
            f"rollout_mse={res.metrics['rollout_mse_mean']:.6g}"
        )

    write_metrics_csv(out_dir / "all_metrics_summary.csv", [r.metrics for r in results])

    # Small human-readable interpretation guide.
    guide = """
How to read the outputs
=======================

1. Open data_collection_summary.txt first.
   This tells you how each .npz episode is stored: keys, shapes, dtypes, and any metadata.
   If the chosen state/action keys are wrong, rerun with --state-key and --action-key.

2. For each episode folder, inspect:
   - mse_over_time.png
   - mae_over_time.png
   - open_loop_abs_error_heatmap.png
   - top_open_loop_feature_mse.png
   - feature_timeseries/*.png
   - state_vector_compare_tXXXX.png

3. Interpretation:
   - Bad reconstruction means the decoder or encoder-decoder interface is the bottleneck.
   - Good reconstruction but bad one-step means the dynamics/predictor is not learning transitions.
   - Good one-step but bad open-loop means errors compound when the model feeds on itself.
   - Train good but validation bad means generalization issue.

4. The script compares flattened state vectors by default.
   For a nicer SMAC top-down plot, add a schema-specific renderer once you confirm which
   feature indices correspond to unit x/y/health/alive fields.
"""
    write_text(out_dir / "README_how_to_read_outputs.txt", textwrap.dedent(guide).strip() + "\n")
    print(f"[done] wrote all outputs under: {out_dir}")


if __name__ == "__main__":
    main()
