#!/usr/bin/env python3
"""
Exp33 anchored JEPA qualitative rollout sanity check.

Put this file in the smac-jepa-wm repo, e.g.

    tools/sanity_decode_rollout_exp33.py

Run from the smac-jepa-wm root:

    python tools/sanity_decode_rollout_exp33.py \
      --checkpoint runs/rnn_seqmem_exp33_dreamer_7ep_v2_clean/checkpoint.pt \
      --train-episode data/r2_general_2100_full/train/shard_02/r2g_train_0043.npz \
      --val-episode data/r2_general_2100_full/validation/shard_03/r2g_validation_1284.npz \
      --out-dir sanity_outputs/exp33_decode_rollout \
      --device cuda \
      --episode-index 0

Why this script exists:
- The Exp33 checkpoint stores state_dicts, not a runnable model object.
- The selected npz can have smaller per-config dimensions than the checkpoint caps.
  Example: a file may have 7 allies / 6 enemies / 12 actions, while the checkpoint
  was trained with padded caps 9 allies / 10 enemies / 16 actions.
- Therefore we rebuild the exact SMACJEPA + anchored memory architecture from the
  checkpoint metadata/config, and force the dataset tensorization to use checkpoint caps.

Outputs per split:
- decoded_rollout_arrays.npz
- metrics_summary.csv
- mse_over_time.csv
- mse_over_time.png
- open_loop_abs_error_heatmap.png
- entity_xy_tXXXX.png for selected timesteps, assuming token features [1,2] are dx/dy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for plots") from exc


# Make repo-root imports work when this script lives in tools/.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "tools" else Path.cwd().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scalar(x: Any, default: Any = None) -> Any:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.reshape(-1)[0].item()
    return default if x is None else x


def valid_transition_count(npz: Mapping[str, np.ndarray], episode_index: int) -> int:
    if "valid" in npz:
        valid = np.asarray(npz["valid"])
        if valid.ndim == 2:
            row = valid[episode_index].astype(bool)
        else:
            row = valid.astype(bool)
        true_idx = np.flatnonzero(row)
        if true_idx.size:
            # Usually valid is contiguous then padded. Use last true + 1 to preserve indexing.
            return int(true_idx[-1]) + 1
    if "actions" in npz:
        arr = np.asarray(npz["actions"])
        return int(arr.shape[1] if arr.ndim >= 3 else arr.shape[0])
    if "action_onehot" in npz:
        arr = np.asarray(npz["action_onehot"])
        return int(arr.shape[1] if arr.ndim >= 4 else arr.shape[0])
    states = np.asarray(npz["states"])
    return int((states.shape[1] if states.ndim >= 3 else states.shape[0]) - 1)


def slice_one_episode_npz(src_path: Path, dst_path: Path, episode_index: int) -> Dict[str, Any]:
    raw = np.load(src_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    states = data.get("states")
    if states is None:
        raise RuntimeError(f"{src_path} has no 'states' array")
    n_episodes = int(states.shape[0]) if states.ndim >= 3 else 1
    if not (0 <= episode_index < n_episodes):
        raise RuntimeError(f"episode_index={episode_index} out of range for {src_path}; n_episodes={n_episodes}")

    sliced: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        # Slice arrays whose first dimension is the episode dimension.
        if arr.ndim >= 1 and arr.shape[0] == n_episodes and n_episodes > 1:
            sliced[key] = arr[episode_index : episode_index + 1]
        else:
            sliced[key] = arr
    ensure_dir(dst_path.parent)
    np.savez_compressed(dst_path, **sliced)
    return {
        "source": str(src_path),
        "sliced": str(dst_path),
        "n_episodes_in_source": n_episodes,
        "episode_index": episode_index,
        "valid_transitions": valid_transition_count(data, episode_index),
        "source_metadata": {
            "state_dim": scalar(data.get("state_dim")),
            "n_agents": scalar(data.get("n_agents")),
            "n_enemies": scalar(data.get("n_enemies")),
            "n_actions": scalar(data.get("n_actions")),
            "ally_state_feat_size": scalar(data.get("ally_state_feat_size")),
            "enemy_state_feat_size": scalar(data.get("enemy_state_feat_size")),
            "static_dim": scalar(data.get("static_dim")),
            "entity_static_feat_size": scalar(data.get("entity_static_feat_size")),
            "scenario": scalar(data.get("scenario")),
            "terrain_preset": scalar(data.get("terrain_preset")),
            "map_width": scalar(data.get("map_width")),
            "map_height": scalar(data.get("map_height")),
        },
    }


def import_repo_objects():
    try:
        from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
        from smac_jepa.jepa import SMACJEPA
        from smac_jepa.anchored_belief_memory import AnchoredActionConditionedEntityRolloutGRUMemory
        from smac_jepa.train_jepa_exp31_exp33 import (
            ActionConditionedEntityRolloutGRUMemory,
            BeliefEntityRolloutGRUMemory,
            add_feature_valid_masks,
            merge_observed_presence,
            r2_normalize_latent,
        )
    except Exception as exc:
        raise SystemExit(
            "Could not import the local smac-jepa-wm code. Run this from the smac-jepa-wm root, "
            "and make sure the repo has smac_jepa/data/markov_rollout_visibility_dataset.py, "
            "smac_jepa/jepa.py, smac_jepa/anchored_belief_memory.py, and "
            "smac_jepa/train_jepa_exp31_exp33.py.\n"
            f"Original import error: {exc}"
        ) from exc
    return {
        "Dataset": VisibilityMarkovRolloutSMACJEPADataset,
        "SMACJEPA": SMACJEPA,
        "AnchoredMemory": AnchoredActionConditionedEntityRolloutGRUMemory,
        "ActionMemory": ActionConditionedEntityRolloutGRUMemory,
        "BeliefMemory": BeliefEntityRolloutGRUMemory,
        "add_feature_valid_masks": add_feature_valid_masks,
        "merge_observed_presence": merge_observed_presence,
        "r2_normalize_latent": r2_normalize_latent,
    }


def build_dataset(
    dataset_cls: Any,
    npz_path: Path,
    checkpoint: Mapping[str, Any],
    rollout_window: int,
    rollout_horizon: int,
    enemy_visibility_mask: Optional[bool],
    enemy_sight_range: Optional[float],
) -> Any:
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    meta = checkpoint["metadata"]
    return dataset_cls(
        [str(npz_path)],
        rollout_window=int(rollout_window),
        rollout_horizon=int(rollout_horizon),
        mode="entity",
        window_mode="sequential",
        samples_per_epoch=None,
        seed=int(cfg.get("seed", 1)),
        # Crucial: force checkpoint/global caps, not caps inferred from this small config file.
        max_agents=int(meta["max_agents"]),
        max_enemies=int(meta["max_enemies"]),
        max_actions=int(meta["max_actions"]),
        token_dim=int(meta["token_dim"]),
        dynamic_token_dim=int(meta["dynamic_token_dim"]),
        static_dim=int(meta.get("static_dim", 0)),
        entity_static_feat_size=int(meta.get("entity_static_feat_size", 0)),
        enemy_visibility_mask=(
            bool(cfg.get("enemy_visibility_mask", True))
            if enemy_visibility_mask is None
            else bool(enemy_visibility_mask)
        ),
        enemy_sight_range=(
            float(cfg.get("enemy_sight_range", 9.0))
            if enemy_sight_range is None
            else float(enemy_sight_range)
        ),
    )


def build_model(objects: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    meta = checkpoint["metadata"]
    SMACJEPA = objects["SMACJEPA"]
    model = SMACJEPA(
        state_dim=int(meta["state_dim"]),
        n_agents=int(meta["n_agents"]),
        n_actions=int(meta["n_actions"]),
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        action_dim=int(cfg["action_dim"]),
        num_heads=int(cfg["num_heads"]),
        mode=str(meta.get("mode", "entity")),
        max_agents=int(meta["max_agents"]),
        max_enemies=int(meta["max_enemies"]),
        max_actions=int(meta["max_actions"]),
        token_dim=int(meta["token_dim"]),
        static_dim=int(meta.get("static_dim", 0)),
        decoder_weight=float(cfg.get("decoder_weight", 1.0)),
        encoder_layers=int(cfg["encoder_layers"]),
        action_layers=int(cfg["action_layers"]),
        predictor_layers=int(cfg["predictor_layers"]),
        max_context_len=int(cfg.get("max_context_len", 32)),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(f"[warn] model.load_state_dict missing={missing} unexpected={unexpected}", flush=True)
    model.eval()
    return model


def build_memory(objects: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    meta = checkpoint["metadata"]
    latent_dim = int(cfg["latent_dim"])
    memory_dim = int(cfg.get("rollout_memory_dim", 128))
    hidden_dim = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    anchored = bool(meta.get("anchored_belief_memory", False)) or bool(cfg.get("anchored_belief_memory", False))

    if anchored:
        Memory = objects["AnchoredMemory"]
        memory = Memory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            n_actions=int(meta.get("n_actions", meta.get("max_actions"))),
            max_agents=int(meta["max_agents"]),
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)
    elif bool(cfg.get("action_conditioned_memory", False)):
        Memory = objects["ActionMemory"]
        memory = Memory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            n_actions=int(meta.get("n_actions", meta.get("max_actions"))),
            max_agents=int(meta["max_agents"]),
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)
    else:
        Memory = objects["BeliefMemory"]
        memory = Memory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)

    missing, unexpected = memory.load_state_dict(checkpoint["memory_module_state"], strict=False)
    if missing or unexpected:
        print(f"[warn] memory.load_state_dict missing={missing} unexpected={unexpected}", flush=True)
    memory.eval()
    return memory


def masked_mse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.float32)
    while m.ndim < pred.ndim:
        m = m[..., None]
    denom = float(m.sum() * pred.shape[-1]) if m.shape[-1] == 1 else float(m.sum())
    denom = max(denom, 1.0)
    return float((((pred - target) ** 2) * m).sum() / denom)


def masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=np.float32)
    while m.ndim < pred.ndim:
        m = m[..., None]
    denom = float(m.sum() * pred.shape[-1]) if m.shape[-1] == 1 else float(m.sum())
    denom = max(denom, 1.0)
    return float((np.abs(pred - target) * m).sum() / denom)


def flatten_valid_abs_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    # pred/target [T,E,F], mask [T,E]
    err = np.abs(pred - target)
    err[~m] = np.nan
    return err.reshape(err.shape[0], -1)


def save_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_mse(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    t = [int(r["timestep"]) for r in rows]
    plt.figure(figsize=(10, 5))
    for key in ["recon_mse", "one_step_mse", "open_loop_mse"]:
        vals = [float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows]
        plt.plot(t, vals, label=key)
    plt.xlabel("target timestep")
    plt.ylabel("masked decoded MSE")
    plt.title("Decoded error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_heatmap(path: Path, matrix: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="absolute error")
    plt.xlabel("flattened entity-token feature index")
    plt.ylabel("target timestep")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()



def entity_slot_name(slot_idx: int, n_agents: int) -> str:
    if slot_idx < n_agents:
        return f"A{slot_idx}"
    return f"E{slot_idx - n_agents}"


def entity_faction(slot_idx: int, n_agents: int) -> str:
    return "ally" if slot_idx < n_agents else "enemy"


def feature_layout(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return semantic feature names for the dynamic portion of ally/enemy tokens.

    This follows smac_jepa.decoder.format_entity_predictions:
    allies: hp, cooldown_or_energy, dx, dy, optional shield, unit_type_*
    enemies: hp, dx, dy, optional shield, unit_type_*
    The remaining token dimensions, if requested, are static_*/extra_* features.
    """
    max_agents = int(metadata.get("max_agents", metadata.get("n_agents", 0)))
    max_enemies = int(metadata.get("max_enemies", metadata.get("n_enemies", 0)))
    token_dim = int(metadata.get("token_dim", 0))
    dynamic_token_dim = int(metadata.get("dynamic_token_dim", token_dim))
    ally_size = int(metadata.get("ally_state_feat_size", dynamic_token_dim))
    enemy_size = int(metadata.get("enemy_state_feat_size", dynamic_token_dim))
    num_unit_types = int(metadata.get("num_unit_types", 0))
    ally_has_shields = bool(metadata.get("ally_has_shields", False))
    enemy_has_shields = bool(metadata.get("enemy_has_shields", False))

    ally = ["hp", "cooldown_or_energy", "dx", "dy"]
    if ally_has_shields:
        ally.append("shield")
    ally.extend([f"unit_type_{i}" for i in range(num_unit_types)])
    ally = ally[:ally_size]
    while len(ally) < ally_size:
        ally.append(f"ally_extra_{len(ally)}")

    enemy = ["hp", "dx", "dy"]
    if enemy_has_shields:
        enemy.append("shield")
    enemy.extend([f"unit_type_{i}" for i in range(num_unit_types)])
    enemy = enemy[:enemy_size]
    while len(enemy) < enemy_size:
        enemy.append(f"enemy_extra_{len(enemy)}")

    # Append static/extra token dimensions for all-token tables.
    if token_dim > ally_size:
        ally_all = ally + [f"static_or_extra_{i}" for i in range(ally_size, token_dim)]
    else:
        ally_all = ally[:token_dim]
    if token_dim > enemy_size:
        enemy_all = enemy + [f"static_or_extra_{i}" for i in range(enemy_size, token_dim)]
    else:
        enemy_all = enemy[:token_dim]

    return {
        "max_agents": max_agents,
        "max_enemies": max_enemies,
        "token_dim": token_dim,
        "dynamic_token_dim": dynamic_token_dim,
        "ally_size": ally_size,
        "enemy_size": enemy_size,
        "ally_dynamic": ally,
        "enemy_dynamic": enemy,
        "ally_all": ally_all,
        "enemy_all": enemy_all,
        "important_names": {"hp", "cooldown_or_energy", "dx", "dy", "shield"},
        "position_names": {"dx", "dy"},
        "health_names": {"hp", "shield"},
    }


def slot_feature_names(layout: Mapping[str, Any], slot_idx: int, mode: str = "important") -> List[Tuple[int, str, str]]:
    n_agents = int(layout["max_agents"])
    is_ally = slot_idx < n_agents
    names = layout["ally_all"] if is_ally else layout["enemy_all"]
    dyn_len = int(layout["ally_size"] if is_ally else layout["enemy_size"])
    out: List[Tuple[int, str, str]] = []
    for feat_idx, name in enumerate(names):
        if mode == "important" and name not in layout["important_names"]:
            continue
        if mode == "dynamic" and feat_idx >= dyn_len:
            continue
        group = feature_group(name, feat_idx, dyn_len)
        out.append((feat_idx, name, group))
    return out


def feature_group(name: str, feat_idx: int, dynamic_len: int) -> str:
    if name in {"dx", "dy"}:
        return "position"
    if name in {"hp", "shield"}:
        return "health_shield"
    if name == "cooldown_or_energy":
        return "cooldown"
    if name.startswith("unit_type_"):
        return "unit_type"
    if feat_idx >= dynamic_len:
        return "static_or_extra"
    return "other_dynamic"


def _fmt_float(x: Any, ndigits: int = 6) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return v
        return round(v, ndigits)
    except Exception:
        return float("nan")


def _action_id_for_slot(action_seq: Optional[np.ndarray], transition_idx: int, slot_idx: int, n_agents: int) -> Optional[int]:
    if action_seq is None or slot_idx >= n_agents or transition_idx >= action_seq.shape[0]:
        return None
    a = action_seq[transition_idx]
    try:
        if a.ndim == 2 and slot_idx < a.shape[0]:
            return int(np.argmax(a[slot_idx]))
        if a.ndim == 1 and slot_idx < a.shape[0]:
            # Already discrete ids in some loaders.
            return int(a[slot_idx])
    except Exception:
        return None
    return None


def build_entity_value_rows(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    timestep_1based: int,
    feature_mode: str,
) -> List[Dict[str, Any]]:
    """Rows for one target timestep.

    timestep_1based=1 corresponds to target state x_1 predicted from x_0/action_0.
    """
    layout = feature_layout(metadata)
    n_agents = int(layout["max_agents"])
    t = int(timestep_1based) - 1
    actual = arrays["actual"][1:]  # target x_{t+1}
    recon = arrays["reconstruction"][1:]
    one = arrays["one_step"]
    roll = arrays["open_loop"]
    target_mask = arrays["actual_mask"][1:]
    slot_mask = arrays.get("slot_mask", arrays["actual_mask"])[1:]
    input_obs = arrays.get("input_entity_mask_seq")
    one_presence = arrays.get("one_step_presence")
    roll_presence = arrays.get("open_loop_presence")
    action_seq = arrays.get("action_seq")
    if t < 0 or t >= min(actual.shape[0], one.shape[0], roll.shape[0]):
        return []

    rows: List[Dict[str, Any]] = []
    for slot_idx in range(actual.shape[1]):
        present = bool(slot_mask[t, slot_idx] > 0) if slot_mask is not None else bool(target_mask[t, slot_idx] > 0)
        if not present:
            continue
        observed_prev = None
        observed_target = None
        if input_obs is not None:
            # input_obs has states x_0...; t target is x_{t+1}, previous input is x_t.
            observed_prev = bool(input_obs[t, slot_idx] > 0) if t < input_obs.shape[0] else None
            observed_target = bool(input_obs[t + 1, slot_idx] > 0) if (t + 1) < input_obs.shape[0] else None
        slot_name = entity_slot_name(slot_idx, n_agents)
        faction = entity_faction(slot_idx, n_agents)
        action_id = _action_id_for_slot(action_seq, t, slot_idx, n_agents)
        one_pres = None if one_presence is None else _fmt_float(one_presence[t, slot_idx])
        roll_pres = None if roll_presence is None else _fmt_float(roll_presence[t, slot_idx])
        for feat_idx, feat_name, group in slot_feature_names(layout, slot_idx, mode=feature_mode):
            a = float(actual[t, slot_idx, feat_idx])
            rc = float(recon[t, slot_idx, feat_idx])
            o = float(one[t, slot_idx, feat_idx])
            r = float(roll[t, slot_idx, feat_idx])
            rows.append(
                {
                    "target_timestep": int(timestep_1based),
                    "transition_action_timestep": int(t),
                    "slot_idx": int(slot_idx),
                    "entity": slot_name,
                    "faction": faction,
                    "action_id_if_ally": action_id,
                    "slot_present": present,
                    "observed_at_input_t": observed_prev,
                    "observed_at_target_t": observed_target,
                    "one_step_presence_score": one_pres,
                    "open_loop_presence_score": roll_pres,
                    "feature_idx": int(feat_idx),
                    "feature": feat_name,
                    "feature_group": group,
                    "actual": _fmt_float(a),
                    "reconstruction": _fmt_float(rc),
                    "one_step": _fmt_float(o),
                    "open_loop": _fmt_float(r),
                    "recon_abs_error": _fmt_float(abs(rc - a)),
                    "one_step_abs_error": _fmt_float(abs(o - a)),
                    "open_loop_abs_error": _fmt_float(abs(r - a)),
                    "one_minus_actual": _fmt_float(o - a),
                    "open_loop_minus_actual": _fmt_float(r - a),
                }
            )
    return rows


def build_compact_entity_rows(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    timestep_1based: int,
) -> List[Dict[str, Any]]:
    layout = feature_layout(metadata)
    n_agents = int(layout["max_agents"])
    t = int(timestep_1based) - 1
    actual = arrays["actual"][1:]
    one = arrays["one_step"]
    roll = arrays["open_loop"]
    target_mask = arrays["actual_mask"][1:]
    slot_mask = arrays.get("slot_mask", arrays["actual_mask"])[1:]
    input_obs = arrays.get("input_entity_mask_seq")
    action_seq = arrays.get("action_seq")
    if t < 0 or t >= min(actual.shape[0], one.shape[0], roll.shape[0]):
        return []
    rows: List[Dict[str, Any]] = []
    wanted = ["hp", "shield", "cooldown_or_energy", "dx", "dy"]
    for slot_idx in range(actual.shape[1]):
        present = bool(slot_mask[t, slot_idx] > 0) if slot_mask is not None else bool(target_mask[t, slot_idx] > 0)
        if not present:
            continue
        name_to_idx = {name: idx for idx, name, _grp in slot_feature_names(layout, slot_idx, mode="dynamic")}
        row: Dict[str, Any] = {
            "target_timestep": int(timestep_1based),
            "entity": entity_slot_name(slot_idx, n_agents),
            "faction": entity_faction(slot_idx, n_agents),
            "action_id_if_ally": _action_id_for_slot(action_seq, t, slot_idx, n_agents),
            "observed_at_input_t": bool(input_obs[t, slot_idx] > 0) if input_obs is not None and t < input_obs.shape[0] else None,
            "observed_at_target_t": bool(input_obs[t + 1, slot_idx] > 0) if input_obs is not None and t + 1 < input_obs.shape[0] else None,
        }
        for feat in wanted:
            if feat not in name_to_idx:
                continue
            j = name_to_idx[feat]
            a = float(actual[t, slot_idx, j])
            o = float(one[t, slot_idx, j])
            r = float(roll[t, slot_idx, j])
            row[f"actual_{feat}"] = _fmt_float(a)
            row[f"one_step_{feat}"] = _fmt_float(o)
            row[f"open_loop_{feat}"] = _fmt_float(r)
            row[f"one_abs_err_{feat}"] = _fmt_float(abs(o - a))
            row[f"open_abs_err_{feat}"] = _fmt_float(abs(r - a))
        rows.append(row)
    return rows


def build_summary_tables(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return slot_summary, feature_summary, top_one, top_open."""
    layout = feature_layout(metadata)
    n_agents = int(layout["max_agents"])
    actual = arrays["actual"][1:]
    recon = arrays["reconstruction"][1:]
    one = arrays["one_step"]
    roll = arrays["open_loop"]
    target_mask = arrays["actual_mask"][1:]
    slot_mask = arrays.get("slot_mask", arrays["actual_mask"])[1:]
    T = min(actual.shape[0], one.shape[0], roll.shape[0])
    actual, recon, one, roll = actual[:T], recon[:T], one[:T], roll[:T]
    mask = slot_mask[:T].astype(bool)

    slot_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    all_feature_rows: List[Dict[str, Any]] = []

    for slot_idx in range(actual.shape[1]):
        slot_valid = mask[:, slot_idx]
        if not slot_valid.any():
            continue
        slot_name = entity_slot_name(slot_idx, n_agents)
        faction = entity_faction(slot_idx, n_agents)
        feature_specs = slot_feature_names(layout, slot_idx, mode="dynamic")
        # Slot/group summaries.
        groups: Dict[str, List[int]] = {}
        for feat_idx, _feat_name, group in feature_specs:
            groups.setdefault(group, []).append(feat_idx)
        groups["all_dynamic"] = [idx for idx, _name, _grp in feature_specs]
        for group, feat_indices in groups.items():
            if not feat_indices:
                continue
            a = actual[:, slot_idx, feat_indices]
            rc = recon[:, slot_idx, feat_indices]
            o = one[:, slot_idx, feat_indices]
            r = roll[:, slot_idx, feat_indices]
            m = slot_valid[:, None]
            denom = max(float(m.sum() * len(feat_indices)), 1.0)
            slot_rows.append(
                {
                    "entity": slot_name,
                    "slot_idx": slot_idx,
                    "faction": faction,
                    "feature_group": group,
                    "valid_timesteps": int(slot_valid.sum()),
                    "actual_mean": _fmt_float(np.where(m, a, np.nan).mean()),
                    "recon_mae": _fmt_float((np.abs(rc - a) * m).sum() / denom),
                    "one_step_mae": _fmt_float((np.abs(o - a) * m).sum() / denom),
                    "open_loop_mae": _fmt_float((np.abs(r - a) * m).sum() / denom),
                    "recon_mse": _fmt_float((((rc - a) ** 2) * m).sum() / denom),
                    "one_step_mse": _fmt_float((((o - a) ** 2) * m).sum() / denom),
                    "open_loop_mse": _fmt_float((((r - a) ** 2) * m).sum() / denom),
                }
            )
        # Per feature summaries and top-error source rows.
        for feat_idx, feat_name, group in feature_specs:
            valid_values = slot_valid
            a = actual[:, slot_idx, feat_idx]
            rc = recon[:, slot_idx, feat_idx]
            o = one[:, slot_idx, feat_idx]
            r = roll[:, slot_idx, feat_idx]
            if not valid_values.any():
                continue
            feature_rows.append(
                {
                    "entity": slot_name,
                    "slot_idx": slot_idx,
                    "faction": faction,
                    "feature_idx": feat_idx,
                    "feature": feat_name,
                    "feature_group": group,
                    "valid_timesteps": int(valid_values.sum()),
                    "actual_mean": _fmt_float(np.nanmean(np.where(valid_values, a, np.nan))),
                    "actual_min": _fmt_float(np.nanmin(np.where(valid_values, a, np.nan))),
                    "actual_max": _fmt_float(np.nanmax(np.where(valid_values, a, np.nan))),
                    "recon_mae": _fmt_float(np.abs(rc[valid_values] - a[valid_values]).mean()),
                    "one_step_mae": _fmt_float(np.abs(o[valid_values] - a[valid_values]).mean()),
                    "open_loop_mae": _fmt_float(np.abs(r[valid_values] - a[valid_values]).mean()),
                    "one_step_max_abs_error": _fmt_float(np.abs(o[valid_values] - a[valid_values]).max()),
                    "open_loop_max_abs_error": _fmt_float(np.abs(r[valid_values] - a[valid_values]).max()),
                }
            )
            for t in np.flatnonzero(valid_values):
                all_feature_rows.append(
                    {
                        "target_timestep": int(t + 1),
                        "entity": slot_name,
                        "slot_idx": slot_idx,
                        "faction": faction,
                        "feature_idx": feat_idx,
                        "feature": feat_name,
                        "feature_group": group,
                        "actual": _fmt_float(a[t]),
                        "reconstruction": _fmt_float(rc[t]),
                        "one_step": _fmt_float(o[t]),
                        "open_loop": _fmt_float(r[t]),
                        "one_step_abs_error": _fmt_float(abs(o[t] - a[t])),
                        "open_loop_abs_error": _fmt_float(abs(r[t] - a[t])),
                    }
                )

    top_one = sorted(all_feature_rows, key=lambda row: float(row["one_step_abs_error"]), reverse=True)
    top_open = sorted(all_feature_rows, key=lambda row: float(row["open_loop_abs_error"]), reverse=True)
    return slot_rows, feature_rows, top_one, top_open


def write_entity_tables(
    *,
    out_dir: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    table_timesteps: Iterable[int],
    feature_mode: str,
    top_error_k: int,
) -> None:
    table_dir = ensure_dir(out_dir / "entity_value_tables")
    compact_dir = ensure_dir(out_dir / "entity_compact_tables")
    for t in table_timesteps:
        rows = build_entity_value_rows(
            arrays=arrays,
            metadata=metadata,
            timestep_1based=int(t),
            feature_mode=feature_mode,
        )
        if rows:
            save_csv(table_dir / f"entity_values_t{int(t):04d}.csv", rows)
        compact_rows = build_compact_entity_rows(
            arrays=arrays,
            metadata=metadata,
            timestep_1based=int(t),
        )
        if compact_rows:
            save_csv(compact_dir / f"entity_compact_t{int(t):04d}.csv", compact_rows)
            write_compact_markdown(compact_dir / f"entity_compact_t{int(t):04d}.md", compact_rows, int(t))

    slot_rows, feature_rows, top_one, top_open = build_summary_tables(arrays=arrays, metadata=metadata)
    save_csv(out_dir / "slot_error_summary.csv", slot_rows)
    save_csv(out_dir / "feature_error_summary.csv", feature_rows)
    save_csv(out_dir / "top_one_step_feature_errors.csv", top_one[: int(top_error_k)])
    save_csv(out_dir / "top_open_loop_feature_errors.csv", top_open[: int(top_error_k)])
    write_table_readme(out_dir / "README_entity_tables.md", metadata, feature_mode)


def write_compact_markdown(path: Path, rows: List[Mapping[str, Any]], timestep: int) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Keep columns limited to the fields that exist.
    priority_cols = [
        "entity",
        "faction",
        "action_id_if_ally",
        "observed_at_input_t",
        "observed_at_target_t",
        "actual_hp",
        "one_step_hp",
        "open_loop_hp",
        "actual_shield",
        "one_step_shield",
        "open_loop_shield",
        "actual_cooldown_or_energy",
        "one_step_cooldown_or_energy",
        "open_loop_cooldown_or_energy",
        "actual_dx",
        "one_step_dx",
        "open_loop_dx",
        "actual_dy",
        "one_step_dy",
        "open_loop_dy",
        "one_abs_err_hp",
        "open_abs_err_hp",
        "one_abs_err_dx",
        "open_abs_err_dx",
        "one_abs_err_dy",
        "open_abs_err_dy",
    ]
    cols = [c for c in priority_cols if any(c in r for r in rows)]
    lines = [f"# Compact entity values, target timestep {timestep}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_readme(path: Path, metadata: Mapping[str, Any], feature_mode: str) -> None:
    layout = feature_layout(metadata)
    text = f"""# Entity value tables

These tables compare the decoded target token values against reconstruction, one-step prediction, and open-loop prediction.

Important convention:
- `target_timestep = 1` means actual `x_1`, predicted from `x_0` and `action_0`.
- Values are normalized entity-token values, not raw map pixels.
- Allies use dynamic feature order: `{layout['ally_dynamic']}`.
- Enemies use dynamic feature order: `{layout['enemy_dynamic']}`.
- `dx` and `dy` are normalized offsets from the map center in the repo's human-readable decoder.
- `observed_at_input_t` indicates whether the entity was visible/observed at the input state used for that transition.
- `observed_at_target_t` indicates whether the entity was visible/observed at the resulting target state.

Files:
- `entity_value_tables/entity_values_tXXXX.csv`: long table, one row per entity-feature.
- `entity_compact_tables/entity_compact_tXXXX.csv/md`: compact per-entity table for hp/shield/cooldown/dx/dy.
- `slot_error_summary.csv`: aggregate errors per entity slot and feature group.
- `feature_error_summary.csv`: aggregate errors per entity slot and individual feature.
- `top_one_step_feature_errors.csv`: largest one-step feature errors.
- `top_open_loop_feature_errors.csv`: largest open-loop feature errors.

Current detailed table feature mode: `{feature_mode}`.
Use `--value-table-features dynamic` to include all dynamic features, or `--value-table-features all` to include static/extra token dimensions too.
"""
    path.write_text(text, encoding="utf-8")


def plot_entity_xy(
    path: Path,
    actual: np.ndarray,
    one_step: np.ndarray,
    open_loop: np.ndarray,
    mask: np.ndarray,
    timestep: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    # Slot-aligned diagnostic: actual[j], one_step[j], open_loop[j] are same entity slot j.
    # Feature axes use semantic decoder order: allies hp,cooldown,dx,dy; enemies hp,dx,dy.
    # Since ally/enemy dx/dy indices differ, this plot uses feature 1/2 for continuity with earlier plots.
    if actual.shape[-1] < 3:
        return
    if metadata is not None:
        n_agents = int(metadata.get("max_agents", metadata.get("n_agents", 0)))
    else:
        n_agents = 9
    valid = mask[timestep].astype(bool)
    if not valid.any():
        return
    slot_indices = np.where(valid)[0]
    a = actual[timestep, valid]
    o = one_step[timestep, valid] if timestep < one_step.shape[0] else None
    r = open_loop[timestep, valid]
    plt.figure(figsize=(8, 8))
    plt.scatter(a[:, 1], a[:, 2], marker="o", label="actual")
    if o is not None:
        plt.scatter(o[:, 1], o[:, 2], marker="x", label="one-step")
    plt.scatter(r[:, 1], r[:, 2], marker="+", label="open-loop")
    for j, slot_idx in enumerate(slot_indices):
        name = entity_slot_name(int(slot_idx), n_agents)
        plt.text(a[j, 1], a[j, 2], name, fontsize=8)
        if o is not None:
            plt.plot([a[j, 1], o[j, 1]], [a[j, 2], o[j, 2]], linewidth=0.7, alpha=0.55)
        plt.plot([a[j, 1], r[j, 1]], [a[j, 2], r[j, 2]], linewidth=0.7, alpha=0.55, linestyle="--")
    plt.xlabel("token feature 1")
    plt.ylabel("token feature 2")
    plt.title(f"Slot-aligned decoded comparison, timestep {timestep + 1}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def safe_detach(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


@torch.no_grad()
def run_inference_on_batch(
    *,
    model: torch.nn.Module,
    memory: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
    merge_observed_presence: Any,
    r2_normalize_latent: Any,
) -> Dict[str, np.ndarray]:
    entity_seq = batch["entity_seq"]
    observation_mask_seq = batch["entity_mask_seq"]
    target_entity_seq = batch.get("target_entity_seq", entity_seq)
    target_entity_mask_seq = batch.get("target_entity_mask_seq", observation_mask_seq)
    slot_mask_seq = batch.get("entity_slot_mask_seq", target_entity_mask_seq)
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")
    static_flat = static_condition if static_condition is not None else None
    r2_enabled = bool(cfg.get("r2_latent_normalize", False))

    bsz, seq_len, entities, token_dim = entity_seq.shape
    if bsz != 1:
        raise RuntimeError(f"Expected batch size 1 for qualitative script; got {bsz}")
    transition_count = min(action_seq.shape[1], seq_len - 1)

    input_raw = model.encoder(entity_seq, observation_mask_seq)
    target_raw = model.encoder(target_entity_seq, target_entity_mask_seq)
    input_latents = r2_normalize_latent(input_raw, observation_mask_seq, enabled=r2_enabled)
    target_latents = r2_normalize_latent(target_raw, target_entity_mask_seq, enabled=r2_enabled)

    # Decoder sanity check on full-state target latents.
    recon_full = model.decode_entities(target_latents)

    # Teacher-forced one-step prediction over the whole selected segment.
    main_memory = memory.initial_memory(
        bsz, entities, device=entity_seq.device, dtype=input_latents.dtype
    )
    main_presence = slot_mask_seq[:, 0].to(input_latents.dtype)
    action_context_seq = None
    if getattr(memory, "uses_action", False):
        if hasattr(memory, "precompute_action_context_sequence"):
            action_context_seq = memory.precompute_action_context_sequence(
                action_seq, action_mask_seq, entities=entities, dtype=input_latents.dtype
            )

    one_step_pred: List[torch.Tensor] = []
    one_step_presence: List[torch.Tensor] = []
    posterior_latents: List[torch.Tensor] = []
    for t in range(transition_count):
        z_start = input_latents[:, t]
        observed_start = observation_mask_seq[:, t]
        slot_start = slot_mask_seq[:, t]
        main_presence = merge_observed_presence(main_presence, observed_start, slot_start)
        posterior = memory.condition(z_start, main_memory, main_presence)
        posterior_latents.append(posterior)
        action_h = action_seq[:, t : t + 1]
        action_mask_h = action_mask_seq[:, t : t + 1]
        timestep_mask = torch.ones((bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype)
        pred_raw = model.predictor(
            posterior.unsqueeze(1),
            action_h,
            action_mask_h,
            timestep_mask,
            slot_start.unsqueeze(1),
            static_flat,
        )[:, 0]
        future_slot = slot_mask_seq[:, t + 1]
        pred_raw = pred_raw * future_slot.unsqueeze(-1)
        pred = r2_normalize_latent(pred_raw, future_slot, enabled=r2_enabled)
        one_step_pred.append(pred)
        presence_logits = model.predict_presence(pred)
        future_presence = torch.sigmoid(presence_logits.float()).to(pred.dtype) * future_slot
        one_step_presence.append(future_presence)
        if action_context_seq is not None:
            main_memory = memory.update(
                z_start,
                main_memory,
                observed_start,
                action_context=action_context_seq[:, t],
            )
        elif getattr(memory, "uses_action", False):
            main_memory = memory.update(
                z_start,
                main_memory,
                observed_start,
                action=action_seq[:, t],
                action_mask=action_mask_seq[:, t],
            )
        else:
            main_memory = memory.update(z_start, main_memory, observed_start)
        main_presence = future_presence

    one_step_latent = torch.stack(one_step_pred, dim=1)
    one_step_decoded = model.decode_entities(one_step_latent.reshape(bsz * transition_count, entities, -1)).reshape(
        bsz, transition_count, entities, token_dim
    )

    # Pure chain rollout from t=0: after the first state, feed predictions back into the model.
    rollout_memory = memory.initial_memory(
        bsz, entities, device=entity_seq.device, dtype=input_latents.dtype
    )
    z = input_latents[:, 0]
    current_presence = slot_mask_seq[:, 0].to(input_latents.dtype)
    current_update_gate = observation_mask_seq[:, 0]
    current_slot = slot_mask_seq[:, 0]
    open_loop_pred: List[torch.Tensor] = []
    open_loop_presence: List[torch.Tensor] = []
    for t in range(transition_count):
        current_presence = merge_observed_presence(current_presence, current_update_gate, current_slot)
        belief = memory.condition(z, rollout_memory, current_presence)
        action_h = action_seq[:, t : t + 1]
        action_mask_h = action_mask_seq[:, t : t + 1]
        timestep_mask = torch.ones((bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype)
        pred_raw = model.predictor(
            belief.unsqueeze(1),
            action_h,
            action_mask_h,
            timestep_mask,
            current_slot.unsqueeze(1),
            static_flat,
        )[:, 0]
        future_slot = slot_mask_seq[:, t + 1]
        pred_raw = pred_raw * future_slot.unsqueeze(-1)
        pred = r2_normalize_latent(pred_raw, future_slot, enabled=r2_enabled)
        open_loop_pred.append(pred)
        presence_logits = model.predict_presence(pred)
        future_presence = torch.sigmoid(presence_logits.float()).to(pred.dtype) * future_slot
        open_loop_presence.append(future_presence)

        if action_context_seq is not None:
            rollout_memory = memory.update(
                z,
                rollout_memory,
                current_update_gate,
                action_context=action_context_seq[:, t],
            )
        elif getattr(memory, "uses_action", False):
            rollout_memory = memory.update(
                z,
                rollout_memory,
                current_update_gate,
                action=action_seq[:, t],
                action_mask=action_mask_seq[:, t],
            )
        else:
            rollout_memory = memory.update(z, rollout_memory, current_update_gate)
        z = pred
        current_presence = future_presence
        current_update_gate = future_presence
        current_slot = future_slot

    open_loop_latent = torch.stack(open_loop_pred, dim=1)
    open_loop_decoded = model.decode_entities(open_loop_latent.reshape(bsz * transition_count, entities, -1)).reshape(
        bsz, transition_count, entities, token_dim
    )

    return {
        "actual": safe_detach(target_entity_seq[0, : transition_count + 1]),
        "actual_mask": safe_detach(target_entity_mask_seq[0, : transition_count + 1]),
        "slot_mask": safe_detach(slot_mask_seq[0, : transition_count + 1]),
        "state_mask": safe_detach(state_mask[0, : transition_count + 1]),
        "reconstruction": safe_detach(recon_full[0, : transition_count + 1]),
        "one_step": safe_detach(one_step_decoded[0]),
        "open_loop": safe_detach(open_loop_decoded[0]),
        "one_step_presence": safe_detach(torch.stack(one_step_presence, dim=1)[0]),
        "open_loop_presence": safe_detach(torch.stack(open_loop_presence, dim=1)[0]),
        "action_seq": safe_detach(action_seq[0, :transition_count]),
        "action_mask_seq": safe_detach(action_mask_seq[0, :transition_count]),
        "input_entity_seq": safe_detach(entity_seq[0, : transition_count + 1]),
        "input_entity_mask_seq": safe_detach(observation_mask_seq[0, : transition_count + 1]),
    }


def write_metrics_and_plots(
    out_dir: Path,
    arrays: Mapping[str, np.ndarray],
    xy_timesteps: Iterable[int],
    metadata: Mapping[str, Any],
    table_timesteps: Iterable[int],
    value_table_features: str,
    top_error_k: int,
) -> Dict[str, float]:
    actual = arrays["actual"]  # [T+1,E,F]
    target = actual[1:]
    mask = arrays["actual_mask"][1:]
    recon = arrays["reconstruction"][1:]
    one = arrays["one_step"]
    roll = arrays["open_loop"]
    T = min(target.shape[0], one.shape[0], roll.shape[0])
    target = target[:T]
    mask = mask[:T]
    recon = recon[:T]
    one = one[:T]
    roll = roll[:T]

    rows: List[Dict[str, Any]] = []
    for i in range(T):
        rows.append(
            {
                "timestep": i + 1,
                "valid_entities": int(mask[i].sum()),
                "recon_mse": masked_mse_np(recon[i : i + 1], target[i : i + 1], mask[i : i + 1]),
                "recon_mae": masked_mae_np(recon[i : i + 1], target[i : i + 1], mask[i : i + 1]),
                "one_step_mse": masked_mse_np(one[i : i + 1], target[i : i + 1], mask[i : i + 1]),
                "one_step_mae": masked_mae_np(one[i : i + 1], target[i : i + 1], mask[i : i + 1]),
                "open_loop_mse": masked_mse_np(roll[i : i + 1], target[i : i + 1], mask[i : i + 1]),
                "open_loop_mae": masked_mae_np(roll[i : i + 1], target[i : i + 1], mask[i : i + 1]),
            }
        )
    save_csv(out_dir / "mse_over_time.csv", rows)
    plot_mse(out_dir / "mse_over_time.png", rows)
    plot_heatmap(
        out_dir / "open_loop_abs_error_heatmap.png",
        flatten_valid_abs_error(roll, target, mask),
        "Open-loop decoded absolute error",
    )
    plot_heatmap(
        out_dir / "one_step_abs_error_heatmap.png",
        flatten_valid_abs_error(one, target, mask),
        "One-step decoded absolute error",
    )

    for t in xy_timesteps:
        idx = int(t) - 1
        if 0 <= idx < T:
            plot_entity_xy(out_dir / f"entity_xy_t{t:04d}.png", target, one, roll, mask, idx, metadata=metadata)

    write_entity_tables(
        out_dir=out_dir,
        arrays=arrays,
        metadata=metadata,
        table_timesteps=table_timesteps,
        feature_mode=value_table_features,
        top_error_k=top_error_k,
    )

    summary = {
        "timesteps_evaluated": int(T),
        "reconstruction_mse": masked_mse_np(recon, target, mask),
        "reconstruction_mae": masked_mae_np(recon, target, mask),
        "one_step_mse": masked_mse_np(one, target, mask),
        "one_step_mae": masked_mae_np(one, target, mask),
        "open_loop_mse": masked_mse_np(roll, target, mask),
        "open_loop_mae": masked_mae_np(roll, target, mask),
        "open_loop_last_step_mse": float(rows[-1]["open_loop_mse"]) if rows else float("nan"),
        "one_step_last_step_mse": float(rows[-1]["one_step_mse"]) if rows else float("nan"),
    }
    save_csv(out_dir / "metrics_summary.csv", [summary])
    return summary


def run_one_file(
    *,
    label: str,
    source_npz: Path,
    checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    memory: torch.nn.Module,
    objects: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    split_dir = ensure_dir(Path(args.out_dir) / label / source_npz.stem)
    tmp_dir = ensure_dir(split_dir / "_tmp")
    sliced_path = tmp_dir / f"{source_npz.stem}_episode_{args.episode_index}.npz"
    slice_info = slice_one_episode_npz(source_npz, sliced_path, args.episode_index)
    valid_transitions = int(slice_info["valid_transitions"])
    horizon = int(args.rollout_horizon)
    if valid_transitions < horizon + 1:
        raise RuntimeError(
            f"{source_npz} episode {args.episode_index} has only {valid_transitions} valid transitions; "
            f"need at least rollout_horizon+1={horizon + 1}. Try --rollout-horizon 1."
        )
    max_transitions = valid_transitions if args.max_transitions is None else min(valid_transitions, int(args.max_transitions))

    # IMPORTANT:
    # Do NOT try to make one dataset window span the whole valid prefix.
    # The repo dataset class is a *training-window* dataset and it filters with a strict
    # "context window + rollout horizon must fit inside valid transitions" rule. If we set
    # rollout_window = valid_transitions - horizon, many files produce exactly zero valid
    # windows because the endpoint lands on/past the boundary.
    #
    # Use the checkpoint training context length by default, then shrink only if the selected
    # episode is too short. This matches how the model was trained and avoids
    # ValueError("No valid training windows found").
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    requested_window = args.rollout_window
    if requested_window is None:
        requested_window = int(cfg.get("rollout_window", 20))
    # Leave at least one spare transition because some dataset versions use strict < rather
    # than <= when testing candidate windows.
    max_allowed_window = max(1, max_transitions - horizon - 1)
    rollout_window = max(1, min(int(requested_window), int(max_allowed_window)))

    print(
        f"[info] {label}/{source_npz.name}: valid_transitions={valid_transitions} "
        f"max_transitions={max_transitions} rollout_window={rollout_window} "
        f"rollout_horizon={horizon}",
        flush=True,
    )
    dataset = build_dataset(
        objects["Dataset"],
        sliced_path,
        checkpoint,
        rollout_window=rollout_window,
        rollout_horizon=horizon,
        enemy_visibility_mask=args.enemy_visibility_mask,
        enemy_sight_range=args.enemy_sight_range,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError(f"Dataset produced no windows for {source_npz}") from exc

    # Add feature masks if the trainer helper is available; not needed for core inference,
    # but useful to check that the dataset item is compatible with Exp31/33 helpers.
    try:
        batch = objects["add_feature_valid_masks"](batch, dataset)
    except Exception as exc:
        print(f"[warn] add_feature_valid_masks failed; continuing without it: {exc}", flush=True)

    batch = to_device(batch, device)
    arrays = run_inference_on_batch(
        model=model,
        memory=memory,
        batch=batch,
        cfg=checkpoint.get("resolved_config", checkpoint.get("config", {})),
        merge_observed_presence=objects["merge_observed_presence"],
        r2_normalize_latent=objects["r2_normalize_latent"],
    )
    np.savez_compressed(split_dir / "decoded_rollout_arrays.npz", **arrays)
    summary = write_metrics_and_plots(split_dir, arrays, xy_timesteps=args.xy_timesteps, metadata=checkpoint["metadata"], table_timesteps=args.table_timesteps, value_table_features=args.value_table_features, top_error_k=args.top_error_k)

    report = {
        "label": label,
        "source_npz": str(source_npz),
        "output_dir": str(split_dir),
        "slice_info": slice_info,
        "checkpoint_caps_used_for_tensorization": {
            k: checkpoint["metadata"].get(k)
            for k in [
                "state_dim",
                "n_agents",
                "n_enemies",
                "n_actions",
                "max_agents",
                "max_enemies",
                "max_actions",
                "token_dim",
                "dynamic_token_dim",
                "static_dim",
                "entity_static_feat_size",
            ]
        },
        "rollout_window": rollout_window,
        "rollout_horizon": horizon,
        "max_transitions": max_transitions,
        "metrics": summary,
    }
    (split_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[done] {label}: T={summary['timesteps_evaluated']} "
        f"recon_mse={summary['reconstruction_mse']:.6g} "
        f"one_step_mse={summary['one_step_mse']:.6g} "
        f"open_loop_mse={summary['open_loop_mse']:.6g} -> {split_dir}",
        flush=True,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp33 decoded rollout sanity check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-episode", required=True)
    parser.add_argument("--val-episode", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument(
        "--rollout-window",
        type=int,
        default=None,
        help="Context window for the repo dataset. Defaults to checkpoint resolved_config['rollout_window'], usually 20.",
    )
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--enemy-visibility-mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enemy-sight-range", type=float, default=None)
    parser.add_argument(
        "--xy-timesteps",
        type=int,
        nargs="*",
        default=[1, 2, 5, 10, 20, 50, 100, 150, 200],
        help="Target timesteps to plot as entity XY scatter, using token features 1/2.",
    )
    parser.add_argument(
        "--table-timesteps",
        type=int,
        nargs="*",
        default=[1, 2, 5, 10, 20],
        help="Target timesteps to export per-entity real-value tables.",
    )
    parser.add_argument(
        "--value-table-features",
        choices=["important", "dynamic", "all"],
        default="important",
        help="Which token features to include in entity_values_tXXXX.csv. important=hp/shield/cooldown/dx/dy; dynamic=all semantic dynamic features incl unit_type; all=all token dims incl static/extra.",
    )
    parser.add_argument(
        "--top-error-k",
        type=int,
        default=80,
        help="Number of largest feature errors to save in top_*_feature_errors.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = resolve_device(args.device)
    checkpoint = _torch_load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected checkpoint dict; got {type(checkpoint)}")
    for required in ["model_state", "memory_module_state", "metadata"]:
        if required not in checkpoint:
            raise RuntimeError(f"Checkpoint missing required key {required!r}")

    objects = import_repo_objects()
    model = build_model(objects, checkpoint, device)
    memory = build_memory(objects, checkpoint, device)

    meta = checkpoint["metadata"]
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    global_info = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "checkpoint_metadata": meta,
        "important_config": {
            k: cfg.get(k)
            for k in [
                "latent_dim",
                "hidden_dim",
                "action_dim",
                "num_heads",
                "encoder_layers",
                "action_layers",
                "predictor_layers",
                "rollout_window",
                "rollout_horizon",
                "rollout_memory_dim",
                "rollout_memory_hidden_dim",
                "rollout_memory_no_residual",
                "r2_latent_normalize",
                "enemy_visibility_mask",
                "enemy_sight_range",
                "anchored_belief_memory",
                "anchored_belief_version",
                "memory_architecture",
            ]
        },
    }
    (out_dir / "global_run_info.json").write_text(json.dumps(global_info, indent=2) + "\n", encoding="utf-8")
    print(
        "[info] loaded checkpoint "
        f"latent_dim={cfg.get('latent_dim')} memory_dim={cfg.get('rollout_memory_dim')} "
        f"caps=agents{meta.get('max_agents')}/enemies{meta.get('max_enemies')}/actions{meta.get('max_actions')} "
        f"anchored={meta.get('anchored_belief_memory')}",
        flush=True,
    )

    reports = []
    reports.append(
        run_one_file(
            label="train",
            source_npz=Path(args.train_episode),
            checkpoint=checkpoint,
            model=model,
            memory=memory,
            objects=objects,
            args=args,
            device=device,
        )
    )
    reports.append(
        run_one_file(
            label="validation",
            source_npz=Path(args.val_episode),
            checkpoint=checkpoint,
            model=model,
            memory=memory,
            objects=objects,
            args=args,
            device=device,
        )
    )
    (out_dir / "combined_report.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    save_csv(
        out_dir / "combined_metrics_summary.csv",
        [
            {
                "label": r["label"],
                "source_npz": r["source_npz"],
                "episode_index": r["slice_info"]["episode_index"],
                "valid_transitions": r["slice_info"]["valid_transitions"],
                "rollout_window": r["rollout_window"],
                "rollout_horizon": r["rollout_horizon"],
                **r["metrics"],
            }
            for r in reports
        ],
    )
    print(f"[done] wrote combined report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
