from __future__ import annotations

"""
Weekend Exp42-51 trainer for SMAC-JEPA RNN seqmem experiments.

This file intentionally does not replace the existing trainer. It imports the stable
components from train_markov_rollout_rnn_visibility_seqmem_experiments.py and adds
first-pass implementations of the Exp42-51 ideas:

- event-balanced sampling
- event/delta/inverse-dynamics auxiliary losses
- hidden copy-vs-update residual/gate losses with enemy-only scope
- local action counterfactual effect regularizer
- typed hidden update losses
- R2-adapter-aware auxiliary probes
- longer horizon support via normal --rollout-horizon
- uncertainty/NLL head
- prioritized hard-window replay / PER
- reward-relevant event sampling via sample weights

The implementation is deliberately defensive: missing reward/availability targets
become zero-loss diagnostics instead of crashing. Smoke tests should still verify
that the wiring works before a full weekend run.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.modules import sigreg_loss
from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

# Reuse stable helpers/classes from the existing trainer.
from smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments import (  # noqa: E402
    ActionConditionedEntityRolloutGRUMemory,
    pooled_action_context,
    resolve_device,
    scheduled_value,
    temporal_time_weights,
    weighted_bce,
    weighted_mse,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekend Exp42-51 SMAC-JEPA RNN seqmem trainer")

    # Base trainer flags, kept compatible with existing runner/evals.
    parser.add_argument("--manifest", default=None, help="Entity dataset split manifest")
    parser.add_argument("--data-dir", default=None, help="Directory containing .npz files to auto-split")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-size", default="default", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--action-layers", type=int)
    parser.add_argument("--predictor-layers", type=int)
    parser.add_argument("--max-context-len", type=int, default=32)
    parser.add_argument("--rollout-window", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--window-mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--enemy-visibility-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enemy-sight-range", type=float, default=9.0)
    parser.add_argument("--temporal-loss", choices=["uniform", "lambda", "flat-decay"], default="lambda")
    parser.add_argument("--td-lambda", "--temporal-lambda", dest="td_lambda", type=float, default=0.9)
    parser.add_argument("--flat-decay-start", type=int, default=None)
    parser.add_argument("--flat-decay-final-weight", type=float, default=0.5)
    parser.add_argument("--detach-rollout-targets", action="store_true")
    parser.add_argument("--unweighted-aux-losses", action="store_true")
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-weight-start", type=float, default=None)
    parser.add_argument("--sigreg-weight-end", type=float, default=None)
    parser.add_argument("--sigreg-warmup-epochs", type=int, default=0)
    parser.add_argument("--decoder-weight", type=float, default=1.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--action-conditioned-memory", action="store_true")
    parser.add_argument("--one-step-weight", type=float, default=0.0)
    parser.add_argument("--target-mode", choices=["full", "observed"], default="full")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rollout-memory-dim", type=int, default=128)
    parser.add_argument("--rollout-memory-hidden-dim", type=int, default=None)
    parser.add_argument("--rollout-memory-no-residual", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="SMAC-JEPA-losses")
    parser.add_argument("--wandb-entity", default="kialok-nus")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    # Exp39/40-style sampling and action/change pressure.
    parser.add_argument("--event-balanced-sampling", action="store_true")
    parser.add_argument("--event-fraction", type=float, default=0.0)
    parser.add_argument("--event-pool-fraction", type=float, default=0.25)
    parser.add_argument("--event-change-threshold", type=float, default=0.01)
    parser.add_argument("--event-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--inverse-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--inverse-dynamics-hidden-dim", type=int, default=256)

    # Exp42/43/51 hidden copy-vs-update.
    parser.add_argument("--hidden-change-residual-weight", type=float, default=0.0)
    parser.add_argument("--hidden-change-gate-weight", type=float, default=0.0)
    parser.add_argument("--hidden-change-gate-temperature", type=float, default=0.10)
    parser.add_argument(
        "--hidden-change-scope",
        choices=["all_hidden", "enemy_hidden_only", "all"],
        default="all_hidden",
    )

    # Exp44 local action counterfactual.
    parser.add_argument("--local-action-counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--local-action-neighbor-radius", type=float, default=9.0)
    parser.add_argument("--local-action-drift-weight", type=float, default=0.0)

    # Exp45 typed hidden update losses.
    parser.add_argument("--typed-hidden-update-loss", action="store_true")
    parser.add_argument("--position-delta-weight", type=float, default=0.0)
    parser.add_argument("--health-delta-weight", type=float, default=0.0)
    parser.add_argument("--presence-transition-weight", type=float, default=0.0)

    # Exp46 R2-adapter-aware probes.
    parser.add_argument("--r2-adapter-probe-weight", type=float, default=0.0)
    parser.add_argument("--reward-probe-weight", type=float, default=0.0)
    parser.add_argument("--avail-probe-weight", type=float, default=0.0)
    parser.add_argument("--alive-probe-weight", type=float, default=0.0)

    # Exp48 uncertainty-aware hidden belief.
    parser.add_argument("--hidden-uncertainty-head", action="store_true")
    parser.add_argument("--hidden-nll-weight", type=float, default=0.0)
    parser.add_argument("--hidden-mixture-components", type=int, default=1)

    # Exp49/51 PER.
    parser.add_argument("--priority-replay", action="store_true")
    parser.add_argument("--priority-replay-warmup-epochs", type=int, default=1)
    parser.add_argument("--priority-replay-alpha", type=float, default=0.6)
    parser.add_argument("--priority-replay-uniform-mix", type=float, default=0.5)
    parser.add_argument("--priority-replay-ema", type=float, default=0.9)
    parser.add_argument("--priority-replay-cap", type=float, default=5.0)
    parser.add_argument("--priority-replay-score", default="hidden_changed,event_changed,presence")
    parser.add_argument("--priority-replay-max-scan", type=int, default=0, help="0 means scan all dataset samples")

    # Exp50 reward-relevant event sampling.
    parser.add_argument("--reward-event-balanced-sampling", action="store_true")
    parser.add_argument("--reward-event-fraction", type=float, default=0.0)
    parser.add_argument("--reward-event-types", default="damage,enemy_death,ally_death,terminal")

    return parser.parse_args()


def to_device_keep_index(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def resolved_arch_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    preset = get_model_preset(args.model_size)
    return {
        "latent_dim": args.latent_dim or preset.latent_dim,
        "hidden_dim": args.hidden_dim or preset.hidden_dim,
        "action_dim": args.action_dim or preset.action_dim,
        "num_heads": args.num_heads or preset.num_heads,
        "encoder_layers": args.encoder_layers or preset.encoder_layers,
        "action_layers": args.action_layers or preset.action_layers,
        "predictor_layers": args.predictor_layers or preset.predictor_layers,
        "batch_size": args.batch_size or preset.batch_size,
        "lr": args.lr or preset.lr,
    }


def load_data_paths_from_args(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        return [str(path) for path in load_manifest(args.manifest, args.split)]
    if args.data_dir is None:
        raise SystemExit("Either --manifest or --data-dir must be provided.")
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 .npz files in {data_dir}, found {len(files)}.")
    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    eval_count = max(1, round(len(files) * args.eval_fraction))
    eval_files = sorted(shuffled[:eval_count])
    train_files = sorted(shuffled[eval_count:])
    if args.split == "train":
        selected = train_files
    elif args.split in {"eval", "test"}:
        selected = eval_files
    else:
        raise SystemExit(f"Unknown split: {args.split}. Use train or eval.")
    print(
        f"Auto-split from {data_dir}: total={len(files)} train={len(train_files)} eval={len(eval_files)} using split={args.split}",
        flush=True,
    )
    return [str(path) for path in selected]


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        if not isinstance(sample, dict):
            raise TypeError("Expected dataset sample to be a dict")
        sample = dict(sample)
        sample["__index"] = int(index)
        return sample


class WeekendAuxHeads(nn.Module):
    def __init__(self, *, latent_dim: int, token_dim: int, n_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.token_dim = int(token_dim)
        self.n_actions = int(n_actions)
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_actions)
        )
        self.adapter_probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, token_dim)
        )
        self.alive_head = nn.Linear(latent_dim, 1)
        self.uncertainty_head = nn.Linear(latent_dim, latent_dim)
        self.reward_probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.avail_probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_actions)
        )


def _feature_dims(token_dim: int) -> tuple[slice, slice, slice]:
    pos = slice(0, min(2, token_dim))
    health_start = 2 if token_dim > 2 else 0
    health_end = min(4, token_dim)
    health = slice(health_start, health_end)
    rest = slice(health_end, token_dim)
    return pos, health, rest


def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def _safe_weighted_mse_features(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * max(1, pred.shape[-1])
    return ((pred - target).pow(2) * mask).sum() / denom


def _dynamic_dim_from_batch(batch: dict[str, torch.Tensor], fallback: int) -> int:
    val = batch.get("dynamic_token_dim")
    if val is None:
        return int(fallback)
    if torch.is_tensor(val):
        return int(val.flatten()[0].item())
    return int(val)


def event_score_from_sample(sample: dict[str, Any], *, dynamic_dim: int, threshold: float) -> float:
    x = sample.get("target_entity_seq", sample.get("entity_seq"))
    m = sample.get("target_entity_mask_seq", sample.get("entity_mask_seq"))
    if x is None or m is None:
        return 0.0
    x_t = torch.as_tensor(x).float()
    m_t = torch.as_tensor(m).float()
    if x_t.dim() < 3 or x_t.shape[0] < 2:
        return 0.0
    d = max(1, min(int(dynamic_dim), int(x_t.shape[-1])))
    diff = (x_t[1:, :, :d] - x_t[:-1, :, :d]).abs()
    valid = (m_t[1:] * m_t[:-1]).unsqueeze(-1)
    if valid.sum() <= 0:
        return 0.0
    max_change = (diff * valid).amax().item()
    mean_change = (diff * valid).sum().item() / (valid.sum().item() * d + 1e-6)
    return float(max_change > threshold) + float(mean_change)


def reward_event_score_from_sample(sample: dict[str, Any], *, dynamic_dim: int, threshold: float) -> float:
    # Reward-relevant proxy: true reward if present, otherwise large dynamic/mask transitions.
    for key in ("reward_seq", "reward", "rewards"):
        if key in sample:
            r = torch.as_tensor(sample[key]).float()
            if r.abs().max().item() > 1e-6:
                return 2.0 + float(r.abs().mean().item())
    x = sample.get("target_entity_seq", sample.get("entity_seq"))
    m = sample.get("target_entity_mask_seq", sample.get("entity_mask_seq"))
    if x is None or m is None:
        return 0.0
    x_t = torch.as_tensor(x).float()
    m_t = torch.as_tensor(m).float()
    if x_t.dim() < 3 or x_t.shape[0] < 2:
        return 0.0
    d = max(1, min(int(dynamic_dim), int(x_t.shape[-1])))
    dyn = x_t[:, :, :d]
    diff = (dyn[1:] - dyn[:-1]).abs()
    mask_transition = (m_t[1:] - m_t[:-1]).abs().max().item()
    large_change = (diff.max().item() > max(threshold * 5.0, threshold + 1e-6))
    return float(mask_transition > 0.0) + float(large_change)


def compute_scan_scores(dataset: Dataset, *, dynamic_dim: int, threshold: float, reward_relevant: bool, max_scan: int) -> torch.Tensor:
    n = len(dataset)
    scores = torch.zeros(n, dtype=torch.float32)
    limit = n if max_scan <= 0 else min(n, max_scan)
    print(f"[scan] scoring {limit}/{n} samples reward_relevant={reward_relevant}", flush=True)
    for i in range(limit):
        sample = dataset[i]
        if reward_relevant:
            s = reward_event_score_from_sample(sample, dynamic_dim=dynamic_dim, threshold=threshold)
        else:
            s = event_score_from_sample(sample, dynamic_dim=dynamic_dim, threshold=threshold)
        scores[i] = float(s)
        if (i + 1) % 5000 == 0:
            print(f"[scan] {i + 1}/{limit}", flush=True)
    if limit < n and limit > 0:
        mean = scores[:limit].mean().item()
        scores[limit:] = mean
    return scores


def weights_from_scores(
    *,
    n: int,
    base_event_scores: torch.Tensor | None,
    reward_event_scores: torch.Tensor | None,
    priorities: torch.Tensor | None,
    args: argparse.Namespace,
    epoch: int,
) -> torch.Tensor:
    weights = torch.ones(n, dtype=torch.float32)
    if args.event_balanced_sampling and base_event_scores is not None:
        event = (base_event_scores > 0).float()
        event_rate = event.mean().clamp_min(1e-6)
        event_boost = max(1.0, float(args.event_fraction) / float(event_rate)) if args.event_fraction > 0 else 1.0
        weights = weights * (1.0 + event * min(event_boost, 20.0))
    if args.reward_event_balanced_sampling and reward_event_scores is not None:
        rev = (reward_event_scores > 0).float()
        rate = rev.mean().clamp_min(1e-6)
        boost = max(1.0, float(args.reward_event_fraction) / float(rate)) if args.reward_event_fraction > 0 else 1.0
        weights = weights * (1.0 + rev * min(boost, 20.0))
    if args.priority_replay and priorities is not None and epoch > int(args.priority_replay_warmup_epochs):
        p = priorities.float().clamp_min(1e-6)
        p = p / p.mean().clamp_min(1e-6)
        p = p.clamp(max=float(args.priority_replay_cap)).pow(float(args.priority_replay_alpha))
        mix = float(args.priority_replay_uniform_mix)
        weights = (mix * weights) + ((1.0 - mix) * p)
    return weights.clamp_min(1e-6)


def make_loader(indexed_dataset: IndexedDataset, weights: torch.Tensor, *, batch_size: int, num_workers: int, num_samples: int | None) -> DataLoader:
    sampler = WeightedRandomSampler(
        weights=weights.double(),
        num_samples=int(num_samples or len(indexed_dataset)),
        replacement=True,
    )
    return DataLoader(indexed_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def weekend_rollout_losses(
    model: SMACJEPA,
    memory_module: EntityRolloutGRUMemory,
    aux_heads: WeekendAuxHeads,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    rollout_window: int,
    rollout_horizon: int,
    sigreg_weight: float,
    decoder_weight: float,
    presence_weight: float,
    one_step_weight: float,
    target_mode: str,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
) -> dict[str, torch.Tensor]:
    entity_seq = batch["entity_seq"]
    entity_mask_seq = batch["entity_mask_seq"]
    if target_mode == "observed":
        target_entity_seq_full = entity_seq
        target_entity_mask_seq_full = entity_mask_seq
    elif target_mode == "full":
        target_entity_seq_full = batch.get("target_entity_seq", entity_seq)
        target_entity_mask_seq_full = batch.get("target_entity_mask_seq", entity_mask_seq)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)
    input_latents = model.encoder(entity_seq, entity_mask_seq)
    target_latents = model.encoder(target_entity_seq_full, target_entity_mask_seq_full)
    _, _, entities, latent_dim = input_latents.shape
    token_dim = int(target_entity_seq_full.shape[-1])
    dyn_dim = max(1, min(_dynamic_dim_from_batch(batch, token_dim), token_dim))
    n_agents = int(getattr(args, "n_agents_runtime", 0) or 0)
    # Fallback: most SMACLite configs put allies first; use half if metadata is not present.
    if n_agents <= 0:
        n_agents = max(1, entities // 2)

    main_memory = memory_module.initial_memory(bsz, entities, device=entity_seq.device, dtype=input_latents.dtype)
    static_flat = static_condition if static_condition is not None else None

    pred_by_start: list[torch.Tensor] = []
    target_by_start: list[torch.Tensor] = []
    target_entity_by_start: list[torch.Tensor] = []
    target_entity_mask_by_start: list[torch.Tensor] = []
    input_entity_mask_future_by_start: list[torch.Tensor] = []
    start_entity_by_start: list[torch.Tensor] = []
    slot_mask_by_start: list[torch.Tensor] = []
    valid_by_start: list[torch.Tensor] = []
    action_by_start: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []

    cf_effect_terms: list[torch.Tensor] = []

    for start_idx in range(p):
        z_start = input_latents[:, start_idx]
        start_entity_mask = entity_mask_seq[:, start_idx]
        rollout_memory = main_memory
        z = z_start
        current_entity_mask = start_entity_mask
        pred_steps: list[torch.Tensor] = []
        target_steps: list[torch.Tensor] = []
        target_entity_steps: list[torch.Tensor] = []
        target_entity_mask_steps: list[torch.Tensor] = []
        input_entity_mask_future_steps: list[torch.Tensor] = []
        start_entity_steps: list[torch.Tensor] = []
        slot_mask_steps: list[torch.Tensor] = []
        valid_steps: list[torch.Tensor] = []
        action_steps: list[torch.Tensor] = []

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1
            action_h = action_seq[:, action_idx : action_idx + 1]
            action_mask_h = action_mask_seq[:, action_idx : action_idx + 1]
            valid_h = state_mask[:, target_idx]
            timestep_mask_h = torch.ones((bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype)
            entity_mask_h = current_entity_mask.unsqueeze(1)
            z_conditioned = memory_module.condition(z, rollout_memory, current_entity_mask)
            pred_h = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask_h,
                entity_mask_h,
                static_flat,
            )[:, 0]
            pred_h = pred_h * current_entity_mask.unsqueeze(-1)

            if args.local_action_counterfactual_weight > 0.0 and step == 0:
                cf_action_h = torch.zeros_like(action_h)
                cf_pred_h = model.predictor(
                    z_conditioned.unsqueeze(1),
                    cf_action_h,
                    action_mask_h,
                    timestep_mask_h,
                    entity_mask_h,
                    static_flat,
                )[:, 0]
                cf_pred_h = cf_pred_h * current_entity_mask.unsqueeze(-1)
                effect = (pred_h - cf_pred_h).abs().mean(dim=-1)
                cf_effect_terms.append(effect.mean())

            target_mask_h = target_entity_mask_seq_full[:, target_idx]
            pred_steps.append(pred_h)
            target_steps.append(target_latents[:, target_idx])
            target_entity_steps.append(target_entity_seq_full[:, target_idx])
            target_entity_mask_steps.append(target_mask_h)
            input_entity_mask_future_steps.append(entity_mask_seq[:, target_idx])
            start_entity_steps.append(entity_seq[:, start_idx])
            slot_mask_steps.append(batch["entity_slot_mask_seq"][:, target_idx])
            valid_steps.append(valid_h)
            action_steps.append(action_seq[:, action_idx])

            if getattr(memory_module, "uses_action", False):
                rollout_memory = memory_module.update(
                    pred_h,
                    rollout_memory,
                    target_mask_h,
                    action=action_h[:, 0],
                    action_mask=action_mask_h[:, 0],
                )
            else:
                rollout_memory = memory_module.update(pred_h, rollout_memory, target_mask_h)
            memory_norms.append(rollout_memory.detach().float().norm(dim=-1).mean())
            z = pred_h
            current_entity_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))
        target_by_start.append(torch.stack(target_steps, dim=1))
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))
        target_entity_mask_by_start.append(torch.stack(target_entity_mask_steps, dim=1))
        input_entity_mask_future_by_start.append(torch.stack(input_entity_mask_future_steps, dim=1))
        start_entity_by_start.append(torch.stack(start_entity_steps, dim=1))
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))
        valid_by_start.append(torch.stack(valid_steps, dim=1))
        action_by_start.append(torch.stack(action_steps, dim=1))

        real_action_h = action_seq[:, start_idx]
        real_action_mask_h = action_mask_seq[:, start_idx]
        if getattr(memory_module, "uses_action", False):
            main_memory = memory_module.update(
                z_start,
                main_memory,
                start_entity_mask,
                action=real_action_h,
                action_mask=real_action_mask_h,
            )
        else:
            main_memory = memory_module.update(z_start, main_memory, start_entity_mask)
        memory_norms.append(main_memory.detach().float().norm(dim=-1).mean())

    pred_latent = torch.stack(pred_by_start, dim=1)
    target_latent = torch.stack(target_by_start, dim=1)
    target_entity = torch.stack(target_entity_by_start, dim=1)
    target_entity_mask = torch.stack(target_entity_mask_by_start, dim=1)
    input_future_mask = torch.stack(input_entity_mask_future_by_start, dim=1)
    start_entity = torch.stack(start_entity_by_start, dim=1)
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)
    valid_mask = torch.stack(valid_by_start, dim=1)
    action_targets = torch.stack(action_by_start, dim=1)

    target_for_pred = target_latent.detach() if detach_rollout_targets else target_latent
    weights = temporal_time_weights(
        h,
        mode=args.temporal_loss,
        td_lambda=args.td_lambda,
        flat_decay_start=args.flat_decay_start,
        flat_decay_final_weight=args.flat_decay_final_weight,
        device=pred_latent.device,
        dtype=pred_latent.dtype,
    )
    uniform_weights = torch.ones_like(weights)
    mask = target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)

    pred_loss = weighted_mse(pred_latent, target_for_pred, mask, weights)
    pred_loss_uniform = weighted_mse(pred_latent, target_for_pred, mask, uniform_weights)
    one_step_loss = weighted_mse(
        pred_latent[:, :, 0:1],
        target_for_pred[:, :, 0:1],
        mask[:, :, 0:1],
        torch.ones(1, device=pred_latent.device, dtype=pred_latent.dtype),
    )

    decoded = model.decode_entities(pred_latent.reshape(bsz * p * h, entities, latent_dim))
    decoded = decoded.reshape(bsz, p, h, entities, -1)
    aux_weights = uniform_weights if unweighted_aux_losses else weights
    decoded_loss = weighted_mse(decoded, target_entity, mask, aux_weights)
    presence_logits = model.predict_presence(pred_latent.reshape(bsz * p * h, entities, latent_dim)).reshape(bsz, p, h, entities)
    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(presence_logits, target_entity_mask, presence_mask, aux_weights)
    reg_latents = torch.cat([input_latents, target_latents], dim=1)
    reg_masks = torch.cat([entity_mask_seq, target_entity_mask_seq_full], dim=1)
    reg_loss = sigreg_loss(reg_latents, reg_masks)

    # Shared masks/deltas for auxiliary losses.
    dyn_slice = slice(0, dyn_dim)
    decoded_dyn = decoded[..., dyn_slice]
    target_dyn = target_entity[..., dyn_slice]
    start_dyn = start_entity[..., dyn_slice]
    target_delta = target_dyn - start_dyn
    pred_delta = decoded_dyn - start_dyn
    delta_mag = target_delta.abs().amax(dim=-1)
    changed_mask = (delta_mag > float(args.event_change_threshold)).float() * target_entity_mask * valid_mask.unsqueeze(-1)
    hidden_mask = (target_entity_mask > 0).float() * (input_future_mask <= 0).float() * valid_mask.unsqueeze(-1)
    if args.hidden_change_scope == "all":
        hidden_scope_mask = target_entity_mask.float() * valid_mask.unsqueeze(-1)
    else:
        hidden_scope_mask = hidden_mask
    if args.hidden_change_scope == "enemy_hidden_only":
        entity_ids = torch.arange(entities, device=pred_latent.device).view(1, 1, 1, entities)
        enemy_mask = (entity_ids >= int(n_agents)).float()
        hidden_scope_mask = hidden_scope_mask * enemy_mask
    hidden_changed_mask = hidden_scope_mask * (delta_mag > float(args.event_change_threshold)).float()
    hidden_unchanged_mask = hidden_scope_mask * (delta_mag <= float(args.event_change_threshold)).float()

    delta_loss = _safe_weighted_mse_features(pred_delta, target_delta, target_entity_mask * valid_mask.unsqueeze(-1))
    event_dynamics_loss = _safe_weighted_mse_features(pred_delta, target_delta, changed_mask)
    hidden_change_residual_loss = _safe_weighted_mse_features(pred_delta, target_delta, hidden_changed_mask)
    pred_change_score = pred_delta.abs().mean(dim=-1) / max(float(args.hidden_change_gate_temperature), 1e-6)
    pred_change_prob = torch.sigmoid(pred_change_score)
    target_change = (delta_mag > float(args.event_change_threshold)).float()
    hidden_change_gate_loss = _safe_masked_mean((pred_change_prob - target_change).pow(2), hidden_scope_mask)

    typed_position_loss = torch.zeros((), device=pred_latent.device)
    typed_health_loss = torch.zeros((), device=pred_latent.device)
    typed_presence_transition_loss = torch.zeros((), device=pred_latent.device)
    if args.typed_hidden_update_loss:
        pos_slice, health_slice, _ = _feature_dims(token_dim)
        if pos_slice.stop and pos_slice.stop > pos_slice.start:
            typed_position_loss = _safe_weighted_mse_features(
                decoded[..., pos_slice] - start_entity[..., pos_slice],
                target_entity[..., pos_slice] - start_entity[..., pos_slice],
                hidden_changed_mask,
            )
        if health_slice.stop and health_slice.stop > health_slice.start:
            typed_health_loss = _safe_weighted_mse_features(
                decoded[..., health_slice] - start_entity[..., health_slice],
                target_entity[..., health_slice] - start_entity[..., health_slice],
                hidden_changed_mask,
            )
        prev_presence = (start_entity.abs().sum(dim=-1) > 0).float()
        transition_target = (target_entity_mask - prev_presence).abs().clamp(0, 1)
        typed_presence_transition_loss = _safe_masked_mean(
            F.binary_cross_entropy_with_logits(presence_logits, transition_target, reduction="none"),
            hidden_scope_mask,
        )

    inverse_dynamics_loss = torch.zeros((), device=pred_latent.device)
    if args.inverse_dynamics_weight > 0.0:
        pooled_cur = pred_latent.detach().mean(dim=3)
        pooled_tgt = target_latent.detach().mean(dim=3)
        inv_in = torch.cat([pooled_cur, pooled_tgt], dim=-1)
        inv_logits = aux_heads.inverse_head(inv_in)
        flat_action = action_targets.reshape(bsz * p * h, *action_targets.shape[3:])
        if flat_action.dim() >= 3:
            flat_action_mask = torch.ones(flat_action.shape[0], flat_action.shape[1], device=flat_action.device, dtype=flat_action.dtype)
        else:
            flat_action_mask = None
        action_ctx = pooled_action_context(flat_action, flat_action_mask, n_actions=aux_heads.n_actions).reshape(bsz, p, h, -1)
        inverse_dynamics_loss = F.binary_cross_entropy_with_logits(inv_logits, action_ctx, reduction="none").mean()

    cf_loss = torch.zeros((), device=pred_latent.device)
    if args.local_action_counterfactual_weight > 0.0 and cf_effect_terms:
        effect = torch.stack(cf_effect_terms).mean()
        # Encourage the world model to have a nonzero bounded action response.
        cf_loss = -torch.clamp(effect, max=0.05)

    r2_adapter_probe_loss = torch.zeros((), device=pred_latent.device)
    if args.r2_adapter_probe_weight > 0.0:
        probe_decoded = aux_heads.adapter_probe(pred_latent)
        r2_adapter_probe_loss = _safe_weighted_mse_features(probe_decoded, target_entity, target_entity_mask * valid_mask.unsqueeze(-1))

    alive_probe_loss = torch.zeros((), device=pred_latent.device)
    if args.alive_probe_weight > 0.0:
        alive_logits = aux_heads.alive_head(pred_latent).squeeze(-1)
        alive_probe_loss = _safe_masked_mean(
            F.binary_cross_entropy_with_logits(alive_logits, target_entity_mask, reduction="none"),
            presence_mask,
        )

    reward_probe_loss = torch.zeros((), device=pred_latent.device)
    if args.reward_probe_weight > 0.0:
        # Use true reward targets if the dataset has them; otherwise this remains a no-op diagnostic.
        reward_key = next((k for k in ("reward_seq", "reward", "rewards") if k in batch), None)
        if reward_key is not None:
            pooled = pred_latent.mean(dim=3)
            reward_pred = aux_heads.reward_probe(pooled).squeeze(-1)
            reward_t = batch[reward_key]
            while reward_t.dim() < reward_pred.dim():
                reward_t = reward_t.unsqueeze(1)
            reward_t = reward_t[..., : reward_pred.shape[-1]].to(dtype=reward_pred.dtype)
            reward_probe_loss = F.mse_loss(reward_pred, reward_t.expand_as(reward_pred))

    avail_probe_loss = torch.zeros((), device=pred_latent.device)
    if args.avail_probe_weight > 0.0 and "avail_actions_seq" in batch:
        pooled = pred_latent.mean(dim=3)
        avail_logits = aux_heads.avail_probe(pooled)
        avail_t = batch["avail_actions_seq"][:, 1 : 1 + p * h]
        # Defensive fallback for unknown shape.
        avail_t = avail_t.reshape(bsz, p, h, -1)[..., : aux_heads.n_actions].to(dtype=avail_logits.dtype)
        avail_probe_loss = F.binary_cross_entropy_with_logits(avail_logits, avail_t, reduction="mean")

    hidden_nll_loss = torch.zeros((), device=pred_latent.device)
    if args.hidden_uncertainty_head and args.hidden_nll_weight > 0.0:
        logvar = aux_heads.uncertainty_head(pred_latent).clamp(min=-5.0, max=5.0)
        nll = 0.5 * ((pred_latent - target_for_pred).pow(2) * torch.exp(-logvar) + logvar)
        hidden_nll_loss = _safe_masked_mean(nll.mean(dim=-1), hidden_scope_mask)

    total_loss = (
        pred_loss
        + one_step_weight * one_step_loss
        + sigreg_weight * reg_loss
        + decoder_weight * decoded_loss
        + presence_weight * presence_loss
        + float(args.delta_loss_weight) * delta_loss
        + float(args.event_dynamics_weight) * event_dynamics_loss
        + float(args.inverse_dynamics_weight) * inverse_dynamics_loss
        + float(args.hidden_change_residual_weight) * hidden_change_residual_loss
        + float(args.hidden_change_gate_weight) * hidden_change_gate_loss
        + float(args.local_action_counterfactual_weight) * cf_loss
        + float(args.position_delta_weight) * typed_position_loss
        + float(args.health_delta_weight) * typed_health_loss
        + float(args.presence_transition_weight) * typed_presence_transition_loss
        + float(args.r2_adapter_probe_weight) * r2_adapter_probe_loss
        + float(args.reward_probe_weight) * reward_probe_loss
        + float(args.avail_probe_weight) * avail_probe_loss
        + float(args.alive_probe_weight) * alive_probe_loss
        + float(args.hidden_nll_weight) * hidden_nll_loss
    )

    # Per-sample priority: mean dynamic rollout error + hidden changed error + presence error proxy.
    per_sample_dyn = ((decoded_dyn - target_dyn).abs() * mask[..., :1]).mean(dim=(1, 2, 3, 4))
    hc_mask = hidden_changed_mask.unsqueeze(-1)
    per_sample_hc = ((decoded_dyn - target_dyn).abs() * hc_mask).sum(dim=(1, 2, 3, 4)) / (hc_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0) * dyn_dim)
    pres_err = (torch.sigmoid(presence_logits) - target_entity_mask).abs()
    per_sample_pres = (pres_err * presence_mask).sum(dim=(1, 2, 3)) / presence_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    sample_priority = (per_sample_dyn + per_sample_hc + 0.25 * per_sample_pres).detach()

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "pred_loss_uniform": pred_loss_uniform.detach(),
        "one_step_loss": one_step_loss.detach(),
        "weighted_one_step_loss": (one_step_weight * one_step_loss).detach(),
        "sigreg_loss": reg_loss,
        "decoded_loss": decoded_loss,
        "presence_loss": presence_loss,
        "weighted_presence_loss": (presence_weight * presence_loss).detach(),
        "delta_loss": delta_loss.detach(),
        "weighted_delta_loss": (float(args.delta_loss_weight) * delta_loss).detach(),
        "event_dynamics_loss": event_dynamics_loss.detach(),
        "weighted_event_dynamics_loss": (float(args.event_dynamics_weight) * event_dynamics_loss).detach(),
        "inverse_dynamics_loss": inverse_dynamics_loss.detach(),
        "hidden_change_residual_loss": hidden_change_residual_loss.detach(),
        "hidden_change_gate_loss": hidden_change_gate_loss.detach(),
        "typed_position_loss": typed_position_loss.detach(),
        "typed_health_loss": typed_health_loss.detach(),
        "typed_presence_transition_loss": typed_presence_transition_loss.detach(),
        "r2_adapter_probe_loss": r2_adapter_probe_loss.detach(),
        "reward_probe_loss": reward_probe_loss.detach(),
        "avail_probe_loss": avail_probe_loss.detach(),
        "alive_probe_loss": alive_probe_loss.detach(),
        "hidden_nll_loss": hidden_nll_loss.detach(),
        "local_action_counterfactual_loss": cf_loss.detach(),
        "hidden_changed_count": hidden_changed_mask.sum().detach(),
        "hidden_unchanged_count": hidden_unchanged_mask.sum().detach(),
        "changed_count": changed_mask.sum().detach(),
        "sample_priority_mean": sample_priority.mean().detach(),
        "temporal_weight_sum": weights.sum().detach(),
        "rollout_window": torch.tensor(float(p), device=pred_latent.device),
        "rollout_horizon": torch.tensor(float(h), device=pred_latent.device),
        "memory_norm_mean": torch.stack(memory_norms).mean() if memory_norms else torch.tensor(0.0, device=pred_latent.device),
        "sample_priority": sample_priority,
    }
    with torch.no_grad():
        for step in range(h):
            step_loss = weighted_mse(
                pred_latent[:, :, step : step + 1],
                target_for_pred[:, :, step : step + 1],
                mask[:, :, step : step + 1],
                torch.ones(1, device=pred_latent.device, dtype=pred_latent.dtype),
            )
            losses[f"pred_loss_h{step + 1}"] = step_loss.detach()
    return losses


def main() -> None:
    args = parse_args()
    arch = resolved_arch_from_args(args)
    if args.rollout_window < 1 or args.rollout_horizon < 1:
        raise SystemExit("--rollout-window and --rollout-horizon must be >= 1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    data_paths = load_data_paths_from_args(args)
    cap_paths = data_paths  # hotfix: infer metadata caps from selected split only
    cap_dataset = SMACJEPADataset(cap_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata
    dataset = VisibilityMarkovRolloutSMACJEPADataset(
        data_paths,
        rollout_window=args.rollout_window,
        rollout_horizon=args.rollout_horizon,
        mode="entity",
        window_mode=args.window_mode,
        samples_per_epoch=None,  # sampling is handled by DataLoader sampler here
        seed=args.seed,
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        dynamic_token_dim=cap_metadata.dynamic_token_dim,
        static_dim=cap_metadata.static_dim,
        entity_static_feat_size=cap_metadata.entity_static_feat_size,
        enemy_visibility_mask=args.enemy_visibility_mask,
        enemy_sight_range=args.enemy_sight_range,
    )
    indexed_dataset = IndexedDataset(dataset)
    n = len(indexed_dataset)
    dynamic_dim = int(getattr(cap_metadata, "dynamic_token_dim", cap_metadata.token_dim))
    args.n_agents_runtime = int(dataset.metadata.n_agents)

    base_event_scores: torch.Tensor | None = None
    reward_event_scores: torch.Tensor | None = None
    if args.event_balanced_sampling or args.priority_replay:
        base_event_scores = compute_scan_scores(
            dataset,
            dynamic_dim=dynamic_dim,
            threshold=float(args.event_change_threshold),
            reward_relevant=False,
            max_scan=int(args.priority_replay_max_scan),
        )
    if args.reward_event_balanced_sampling:
        reward_event_scores = compute_scan_scores(
            dataset,
            dynamic_dim=dynamic_dim,
            threshold=float(args.event_change_threshold),
            reward_relevant=True,
            max_scan=int(args.priority_replay_max_scan),
        )
    priorities = torch.ones(n, dtype=torch.float32)

    model = SMACJEPA(
        state_dim=dataset.metadata.state_dim,
        n_agents=dataset.metadata.n_agents,
        n_actions=dataset.metadata.n_actions,
        latent_dim=int(arch["latent_dim"]),
        hidden_dim=int(arch["hidden_dim"]),
        action_dim=int(arch["action_dim"]),
        num_heads=int(arch["num_heads"]),
        mode=dataset.metadata.mode,
        max_agents=dataset.metadata.max_agents,
        max_enemies=dataset.metadata.max_enemies,
        max_actions=dataset.metadata.max_actions,
        token_dim=dataset.metadata.token_dim,
        static_dim=dataset.metadata.static_dim,
        decoder_weight=args.decoder_weight,
        encoder_layers=int(arch["encoder_layers"]),
        action_layers=int(arch["action_layers"]),
        predictor_layers=int(arch["predictor_layers"]),
        max_context_len=args.max_context_len,
    ).to(device)

    if args.action_conditioned_memory:
        memory_module = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=args.rollout_memory_dim,
            n_actions=dataset.metadata.n_actions,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)
    else:
        memory_module = EntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=args.rollout_memory_dim,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)

    aux_heads = WeekendAuxHeads(
        latent_dim=int(arch["latent_dim"]),
        token_dim=dataset.metadata.token_dim,
        n_actions=dataset.metadata.n_actions,
        hidden_dim=int(args.inverse_dynamics_hidden_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(memory_module.parameters()) + list(aux_heads.parameters()),
        lr=float(arch["lr"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 1
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "memory_module_state" in checkpoint:
            memory_module.load_state_dict(checkpoint["memory_module_state"])
        if "aux_heads_state" in checkpoint:
            aux_heads.load_state_dict(checkpoint["aux_heads_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if "priority_replay_state" in checkpoint:
            pr = checkpoint["priority_replay_state"].get("priorities")
            if torch.is_tensor(pr) and pr.numel() == n:
                priorities = pr.detach().cpu().float()

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "weekend_exp42_51_seqmem",
        "enemy_visibility_mask": args.enemy_visibility_mask,
        "enemy_sight_range": args.enemy_sight_range,
        "action_conditioned_memory": args.action_conditioned_memory,
        "one_step_weight": args.one_step_weight,
        "target_mode": args.target_mode,
        "segment_action_len": args.rollout_window + args.rollout_horizon,
        "segment_state_len": args.rollout_window + args.rollout_horizon + 1,
        "implemented_exp42_51_flags": True,
        "n_agents": dataset.metadata.n_agents,
        "n_actions": dataset.metadata.n_actions,
        "token_dim": dataset.metadata.token_dim,
        "dynamic_token_dim": dynamic_dim,
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("W&B logging requested with --wandb, but wandb is not installed.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or out_dir.name,
            config=saved_config,
            mode=args.wandb_mode,
            dir=str(out_dir),
        )
        wandb_run.watch(model, log=None)
        wandb_run.watch(memory_module, log=None)
        wandb_run.watch(aux_heads, log=None)

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "memory_module_state": memory_module.state_dict(),
                "aux_heads_state": aux_heads.state_dict(),
                "metadata": {
                    "state_dim": dataset.metadata.state_dim,
                    "n_agents": dataset.metadata.n_agents,
                    "n_actions": dataset.metadata.n_actions,
                    "n_enemies": dataset.metadata.n_enemies,
                    "ally_state_feat_size": dataset.metadata.ally_state_feat_size,
                    "enemy_state_feat_size": dataset.metadata.enemy_state_feat_size,
                    "ally_has_shields": dataset.metadata.ally_has_shields,
                    "enemy_has_shields": dataset.metadata.enemy_has_shields,
                    "num_unit_types": dataset.metadata.num_unit_types,
                    "max_agents": dataset.metadata.max_agents,
                    "max_enemies": dataset.metadata.max_enemies,
                    "max_actions": dataset.metadata.max_actions,
                    "token_dim": dataset.metadata.token_dim,
                    "dynamic_token_dim": dataset.metadata.dynamic_token_dim,
                    "static_dim": dataset.metadata.static_dim,
                    "entity_static_feat_size": dataset.metadata.entity_static_feat_size,
                    "mode": dataset.metadata.mode,
                },
                "config": vars(args),
                "resolved_config": saved_config,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "priority_replay_state": {"priorities": priorities.cpu()},
                "epoch": epoch_to_save,
                "global_step": global_step,
            },
            checkpoint_path,
        )

    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")
    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []

    model.train()
    memory_module.train()
    aux_heads.train()
    print(
        "weekend_exp42_51_seqmem "
        f"p={args.rollout_window} h={args.rollout_horizon} "
        f"event_balanced={args.event_balanced_sampling} priority_replay={args.priority_replay} "
        f"hidden_change_residual={args.hidden_change_residual_weight} scope={args.hidden_change_scope} "
        f"r2_adapter_probe={args.r2_adapter_probe_weight} uncertainty={args.hidden_uncertainty_head}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        active_sigreg_weight = scheduled_value(
            epoch=epoch,
            fixed_value=args.sigreg_weight,
            start_value=args.sigreg_weight_start,
            end_value=args.sigreg_weight_end,
            warmup_epochs=args.sigreg_warmup_epochs,
        )
        sample_weights = weights_from_scores(
            n=n,
            base_event_scores=base_event_scores,
            reward_event_scores=reward_event_scores,
            priorities=priorities,
            args=args,
            epoch=epoch,
        )
        loader = make_loader(
            indexed_dataset,
            sample_weights,
            batch_size=int(arch["batch_size"]),
            num_workers=args.num_workers,
            num_samples=args.samples_per_epoch,
        )
        print(
            f"epoch_schedule epoch={epoch} active_sigreg_weight={active_sigreg_weight:.6f} "
            f"sampler_weight_min={sample_weights.min().item():.4f} max={sample_weights.max().item():.4f} mean={sample_weights.mean().item():.4f}",
            flush=True,
        )
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        for batch in loader:
            global_step += 1
            epoch_batches += 1
            raw_indices = batch.get("__index")
            batch = to_device_keep_index(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with autocast_context:
                losses = weekend_rollout_losses(
                    model,
                    memory_module,
                    aux_heads,
                    batch,
                    args,
                    rollout_window=args.rollout_window,
                    rollout_horizon=args.rollout_horizon,
                    sigreg_weight=active_sigreg_weight,
                    decoder_weight=args.decoder_weight,
                    presence_weight=args.presence_weight,
                    one_step_weight=args.one_step_weight,
                    target_mode=args.target_mode,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                )
            scaler.scale(losses["total_loss"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(memory_module.parameters()) + list(aux_heads.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            if args.priority_replay and raw_indices is not None and "sample_priority" in losses:
                idxs = raw_indices.detach().cpu().long().flatten()
                scores = losses["sample_priority"].detach().cpu().float().flatten()
                ema = float(args.priority_replay_ema)
                for idx, score in zip(idxs.tolist(), scores.tolist()):
                    if 0 <= idx < priorities.numel():
                        priorities[idx] = ema * priorities[idx] + (1.0 - ema) * float(score)

            row: dict[str, float | int] = {"epoch": epoch, "step": global_step, "active_sigreg_weight": active_sigreg_weight}
            for key, value in losses.items():
                if key == "sample_priority":
                    continue
                row[key] = float(value.detach().cpu())
            logger.log(row)
            step_rows.append(row)

            if wandb_run is not None:
                log_dict = {f"train/{key}": value for key, value in row.items() if key not in {"epoch", "step"}}
                log_dict["train/epoch"] = epoch
                log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
                log_dict["train/priority_mean"] = float(priorities.mean().item())
                log_dict["train/priority_max"] = float(priorities.max().item())
                wandb_run.log(log_dict, step=global_step)

            for key, value in row.items():
                if key in {"epoch", "step"}:
                    continue
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value)

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    "epoch={epoch} step={step} total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} "
                    "event_loss={event_dynamics_loss:.6f} hidden_resid={hidden_change_residual_loss:.6f} "
                    "priority={sample_priority_mean:.6f}".format(**row),
                    flush=True,
                )

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save checkpoint.")
        epoch_row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
        for key, value in epoch_sums.items():
            epoch_row[key] = value / max(epoch_batches, 1)
        epoch_row["priority_mean"] = float(priorities.mean().item())
        epoch_row["priority_max"] = float(priorities.max().item())
        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row)

        if wandb_run is not None:
            wandb_run.log({f"epoch/{k}": v for k, v in epoch_row.items()}, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} priority_max={priority_max:.6f}".format(**epoch_row),
            flush=True,
        )
        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}", flush=True)

    write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
    write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
    write_svg_line_plot(step_rows, "step", "pred_loss", "Prediction Loss Per Training Step", out_dir / "pred_loss_by_step.svg")
    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
