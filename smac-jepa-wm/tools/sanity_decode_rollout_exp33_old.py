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


def plot_entity_xy(
    path: Path,
    actual: np.ndarray,
    one_step: np.ndarray,
    open_loop: np.ndarray,
    mask: np.ndarray,
    timestep: int,
) -> None:
    # Token convention from repo docs: unit record exposes hp, dx, dy, ...
    # So feature 1/2 are the most useful visual guess for position.
    if actual.shape[-1] < 3:
        return
    valid = mask[timestep].astype(bool)
    if not valid.any():
        return
    a = actual[timestep, valid]
    o = one_step[timestep, valid] if timestep < one_step.shape[0] else None
    r = open_loop[timestep, valid]
    plt.figure(figsize=(6, 6))
    plt.scatter(a[:, 1], a[:, 2], marker="o", label="actual")
    if o is not None:
        plt.scatter(o[:, 1], o[:, 2], marker="x", label="one-step")
    plt.scatter(r[:, 1], r[:, 2], marker="+", label="open-loop")
    plt.xlabel("token feature 1, usually dx")
    plt.ylabel("token feature 2, usually dy")
    plt.title(f"Entity position-like decoded comparison, timestep {timestep}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
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


def write_metrics_and_plots(out_dir: Path, arrays: Mapping[str, np.ndarray], xy_timesteps: Iterable[int]) -> Dict[str, float]:
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
            plot_entity_xy(out_dir / f"entity_xy_t{t:04d}.png", target, one, roll, mask, idx)

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
    # Need p+h-1 <= max_transitions. This covers the selected segment from state 0 to max_transitions.
    rollout_window = max(1, max_transitions - horizon + 1)
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
    summary = write_metrics_and_plots(split_dir, arrays, xy_timesteps=args.xy_timesteps)

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
