from __future__ import annotations

"""
Repaired Exp42-44 trainer.

Design constraints:
- Start from the known-good Markov rollout + RNN seqmem trainer contract.
- Preserve Exp40-style base pressure: event-balanced sampling, event dynamics, delta loss,
  and inverse/action pressure with gradients into the predictor.
- Add only one new mechanism per repaired experiment:
    Exp42: weak copy/update on all slots (global weak residual/copy pressure)
    Exp43: enemy-hidden-only copy/update
    Exp44: local action counterfactual pressure
- Fail loudly when a requested mechanism is inactive.

This script intentionally saves standard checkpoint fields:
    model_state, memory_module_state, metadata, config, resolved_config, optimizer_state, scaler_state
so the existing eval scripts can load the model/memory exactly like previous seqmem checkpoints.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from smac_jepa.data import SMACJEPADataset, load_manifest
from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.modules import sigreg_loss
from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments import (
    ActionConditionedEntityRolloutGRUMemory,
    pooled_action_context,
    temporal_time_weights,
    weighted_mse,
    weighted_bce,
    scheduled_value,
)
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repaired Exp42-44 trainer based on Exp40 seqmem")
    p.add_argument("--manifest", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--split", default="train")
    p.add_argument("--model-size", default="default", choices=sorted(MODEL_PRESETS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--action-dim", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--encoder-layers", type=int)
    p.add_argument("--action-layers", type=int)
    p.add_argument("--predictor-layers", type=int)
    p.add_argument("--max-context-len", type=int, default=32)
    p.add_argument("--rollout-window", type=int, default=20)
    p.add_argument("--rollout-horizon", type=int, default=5)
    p.add_argument("--window-mode", choices=["sequential", "random"], default="random")
    p.add_argument("--samples-per-epoch", type=int, default=None)
    p.add_argument("--enemy-visibility-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enemy-sight-range", type=float, default=9.0)
    p.add_argument("--temporal-loss", choices=["uniform", "lambda", "flat-decay"], default="lambda")
    p.add_argument("--td-lambda", "--temporal-lambda", dest="td_lambda", type=float, default=0.9)
    p.add_argument("--flat-decay-start", type=int, default=None)
    p.add_argument("--flat-decay-final-weight", type=float, default=0.5)
    p.add_argument("--detach-rollout-targets", action="store_true")
    p.add_argument("--unweighted-aux-losses", action="store_true")
    p.add_argument("--sigreg-weight", type=float, default=0.005)
    p.add_argument("--sigreg-weight-start", type=float, default=None)
    p.add_argument("--sigreg-weight-end", type=float, default=None)
    p.add_argument("--sigreg-warmup-epochs", type=int, default=0)
    p.add_argument("--decoder-weight", type=float, default=0.005)
    p.add_argument("--presence-weight", type=float, default=0.01)
    p.add_argument("--action-conditioned-memory", action="store_true")
    p.add_argument("--one-step-weight", type=float, default=0.5)
    p.add_argument("--target-mode", choices=["full", "observed"], default="full")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--rollout-memory-dim", type=int, default=128)
    p.add_argument("--rollout-memory-hidden-dim", type=int, default=None)
    p.add_argument("--rollout-memory-no-residual", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="SMAC-JEPA-losses")
    p.add_argument("--wandb-entity", default="kialok-nus")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    # Repaired Exp40-base flags.
    p.add_argument("--event-balanced-sampling", action="store_true")
    p.add_argument("--event-fraction", type=float, default=0.5)
    p.add_argument("--event-change-threshold", type=float, default=0.01)
    p.add_argument("--event-sampler-scan-limit", type=int, default=50000)
    p.add_argument("--event-dynamics-weight", type=float, default=0.0)
    p.add_argument("--delta-loss-weight", type=float, default=0.0)
    p.add_argument("--inverse-dynamics-weight", type=float, default=0.0)
    p.add_argument("--inverse-dynamics-hidden-dim", type=int, default=256)

    # Repaired Exp42/43 copy-vs-update.
    p.add_argument("--hidden-change-residual-weight", type=float, default=0.0)
    p.add_argument("--hidden-change-copy-weight", type=float, default=0.0)
    p.add_argument("--hidden-change-threshold", type=float, default=0.01)
    p.add_argument(
        "--hidden-change-scope",
        choices=["none", "all_slots", "hidden_only", "enemy_only", "enemy_hidden_only"],
        default="none",
    )

    # Repaired Exp44 local action counterfactual.
    p.add_argument("--local-action-counterfactual-weight", type=float, default=0.0)
    p.add_argument("--local-action-neighbor-radius", type=float, default=6.0)
    p.add_argument("--local-action-drift-weight", type=float, default=0.05)
    p.add_argument("--local-action-effect-margin", type=float, default=5e-4)

    # Guardrails.
    p.add_argument("--audit-strict", action="store_true")
    p.add_argument("--audit-min-hidden-count", type=float, default=1.0)
    p.add_argument("--audit-min-hidden-changed-count", type=float, default=1.0)
    p.add_argument("--audit-min-hidden-unchanged-count", type=float, default=1.0)
    p.add_argument("--audit-min-counterfactual-count", type=float, default=1.0)
    p.add_argument("--experiment-name", default="repaired")
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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


def load_data_paths(args: argparse.Namespace) -> list[str]:
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
    return [str(x) for x in (train_files if args.split == "train" else eval_files)]


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def dynamic_dims_from_metadata(dynamic_token_dim: int, feat_dim: int) -> int:
    if dynamic_token_dim and dynamic_token_dim > 0:
        return min(int(dynamic_token_dim), feat_dim)
    return min(4, feat_dim)


def make_entity_scope_mask(
    *,
    scope: str,
    hidden_mask: torch.Tensor,
    entity_slot_mask: torch.Tensor,
    max_agents: int,
) -> torch.Tensor:
    # hidden_mask/entity_slot_mask: [B, P, H, E]
    if scope == "none":
        return torch.zeros_like(entity_slot_mask).float()
    if scope == "all_slots":
        return entity_slot_mask.float()
    if scope == "hidden_only":
        return hidden_mask.float()

    if scope in ("enemy_only", "enemy_hidden_only"):
        e = int(entity_slot_mask.shape[-1])

        # SMAC-JEPA R2 layout is allies first, enemies second.
        # Common layout: 9 ally slots + 10 enemy slots = 19 total.
        enemy_start = int(max_agents)
        if e == 19 and not (0 < enemy_start < e):
            enemy_start = 9
        elif not (0 < enemy_start < e):
            enemy_start = e // 2

        enemy = torch.zeros(e, device=entity_slot_mask.device, dtype=hidden_mask.dtype)
        enemy[enemy_start:] = 1.0
        enemy = enemy.view(1, 1, 1, e)

        if scope == "enemy_only":
            return entity_slot_mask.float() * enemy

        # Strict natural-hidden enemy mode. This can be zero if the sampled
        # batch has no naturally hidden enemy targets.
        return hidden_mask.float() * enemy

    raise ValueError(scope)


class RepairedAuxHeads(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.inverse = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )


def build_last_seen_cache(entity_seq: torch.Tensor, entity_mask_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # entity_seq [B,T,E,F], entity_mask_seq [B,T,E]
    # IMPORTANT: return the last observation strictly BEFORE each timestep.
    # For a target at t+1, copy/update must compare against what the belief knew
    # before seeing t+1, not against the target itself. The previous broken bundle
    # updated before appending, which made visible all-slot copy/update degenerate.
    b, t, e, f = entity_seq.shape
    last = torch.zeros(b, e, f, device=entity_seq.device, dtype=entity_seq.dtype)
    valid = torch.zeros(b, e, device=entity_seq.device, dtype=entity_seq.dtype)
    last_list = []
    valid_list = []
    for idx in range(t):
        last_list.append(last)
        valid_list.append(valid)
        obs = entity_mask_seq[:, idx].float()
        last = torch.where(obs.unsqueeze(-1).bool(), entity_seq[:, idx], last)
        valid = torch.maximum(valid, obs)
    return torch.stack(last_list, dim=1), torch.stack(valid_list, dim=1)


def repaired_rollout_losses(
    model: SMACJEPA,
    memory_module: nn.Module,
    aux_heads: RepairedAuxHeads,
    batch: dict[str, torch.Tensor],
    *,
    rollout_window: int,
    rollout_horizon: int,
    temporal_loss_mode: str,
    td_lambda: float,
    flat_decay_start: int | None,
    flat_decay_final_weight: float,
    sigreg_weight: float,
    decoder_weight: float,
    presence_weight: float,
    one_step_weight: float,
    target_mode: str,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
    dynamic_token_dim: int,
    max_agents: int,
    event_change_threshold: float,
    event_dynamics_weight: float,
    delta_loss_weight: float,
    inverse_dynamics_weight: float,
    hidden_change_residual_weight: float,
    hidden_change_copy_weight: float,
    hidden_change_threshold: float,
    hidden_change_scope: str,
    local_action_counterfactual_weight: float,
    local_action_neighbor_radius: float,
    local_action_drift_weight: float,
    local_action_effect_margin: float,
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
        raise ValueError(target_mode)

    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")
    entity_slot_seq = batch.get("entity_slot_mask_seq", target_entity_mask_seq_full)

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)
    input_latents = model.encoder(entity_seq, entity_mask_seq)
    target_latents = model.encoder(target_entity_seq_full, target_entity_mask_seq_full)
    _, _, entities, latent_dim = input_latents.shape
    feat_dim = target_entity_seq_full.shape[-1]
    dyn_dim = dynamic_dims_from_metadata(dynamic_token_dim, feat_dim)

    last_seen_seq, last_seen_valid_seq = build_last_seen_cache(entity_seq, entity_mask_seq)

    main_memory = memory_module.initial_memory(bsz, entities, device=entity_seq.device, dtype=input_latents.dtype)
    static_flat = static_condition if static_condition is not None else None

    pred_by_start, target_by_start = [], []
    target_entity_by_start, prev_entity_by_start, last_seen_by_start = [], [], []
    target_entity_mask_by_start, observed_mask_by_start, last_seen_valid_by_start = [], [], []
    slot_mask_by_start, valid_by_start = [], []
    cf_losses, cf_counts, cf_near_effects, cf_far_effects = [], [], [], []
    memory_norms = []

    for start_idx in range(p):
        z = input_latents[:, start_idx]
        # Two-mask repair:
        # - current_condition_mask says which current slots are actually observed and should overwrite memory.
        # - current_predict_mask says which slots physically exist and may be predicted.
        # Using the observed mask for both was the old partial-observability bug: hidden slots could
        # be supplied by memory.condition(), then immediately zeroed before the loss, blocking gradients.
        current_condition_mask = entity_mask_seq[:, start_idx]
        current_predict_mask = target_entity_mask_seq_full[:, start_idx] if target_mode == "full" else current_condition_mask
        rollout_memory = main_memory
        pred_steps, target_steps, target_entity_steps, prev_entity_steps, last_seen_steps = [], [], [], [], []
        target_mask_steps, observed_mask_steps, last_valid_steps, slot_mask_steps, valid_steps = [], [], [], [], []

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1
            action_h = action_seq[:, action_idx : action_idx + 1]
            action_mask_h = action_mask_seq[:, action_idx : action_idx + 1]
            valid_h = state_mask[:, target_idx]
            timestep_mask_h = torch.ones((bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype)
            entity_mask_h = current_predict_mask.unsqueeze(1)
            z_conditioned = memory_module.condition(z, rollout_memory, current_condition_mask)
            pred_h = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask_h,
                entity_mask_h,
                static_flat,
            )[:, 0]
            pred_h = pred_h * current_predict_mask.unsqueeze(-1)

            if local_action_counterfactual_weight > 0 and step == 0:
                cf_action = action_h.clone()
                # Pick the first currently observed ally with a non-NOOP action per sample,
                # instead of always mutating ally 0. This makes Exp44 an actual local
                # action counterfactual and avoids silently doing nothing when ally 0 noops.
                # Only mutate currently observed/alive allies; do not pick an unobserved slot as actor.
                ally_alive = current_condition_mask[:, : int(max_agents)].float()
                if cf_action.dim() == 4:
                    action_ids = action_h[:, 0, : int(max_agents)].argmax(dim=-1)
                elif cf_action.dim() == 3:
                    action_ids = action_h[:, 0, : int(max_agents)].long()
                else:
                    action_ids = torch.ones((bsz, int(max_agents)), device=entity_seq.device, dtype=torch.long)
                nonnoop = ((action_ids != 0).float() * ally_alive)
                has_actor = (nonnoop.sum(dim=1) > 0)
                selected_actor = nonnoop.argmax(dim=1)
                batch_idx = torch.arange(bsz, device=entity_seq.device)
                if cf_action.dim() == 4:
                    cf_action[batch_idx, 0, selected_actor, :] = 0
                    cf_action[batch_idx, 0, selected_actor, 0] = 1
                elif cf_action.dim() == 3:
                    cf_action[batch_idx, 0, selected_actor] = 0
                selected_nonnoop = has_actor.float()
                pred_cf = model.predictor(
                    z_conditioned.unsqueeze(1),
                    cf_action,
                    action_mask_h,
                    timestep_mask_h,
                    entity_mask_h,
                    static_flat,
                )[:, 0]
                pred_cf = pred_cf * current_predict_mask.unsqueeze(-1)
                effect = (pred_h - pred_cf).pow(2).mean(dim=-1)  # [B,E]
                # Locality in physical coords. Assumes first two dynamic dims are x/y-like coordinates.
                pos = target_entity_seq_full[:, start_idx, :, :2]
                actor_pos = pos[batch_idx, selected_actor, :].unsqueeze(1)
                dist = (pos - actor_pos).pow(2).sum(dim=-1).sqrt()
                enemy = torch.zeros(entities, device=entity_seq.device, dtype=entity_seq.dtype)
                enemy[int(max_agents):] = 1.0
                actor_entity = torch.zeros((bsz, entities), device=entity_seq.device, dtype=entity_seq.dtype)
                actor_entity[batch_idx, selected_actor] = 1.0
                alive = target_entity_mask_seq_full[:, start_idx].float()
                near_enemy = (dist <= float(local_action_neighbor_radius)).float() * enemy.view(1, entities) * alive
                # Penalize unrelated drift, but do not punish the changed actor itself.
                far_entity = (1.0 - near_enemy.clamp(max=1.0)) * alive * (1.0 - actor_entity)
                active = selected_nonnoop.view(bsz, 1)
                near_count = (near_enemy * active).sum().clamp_min(1.0)
                far_count = (far_entity * active).sum().clamp_min(1.0)
                near_effect = (effect * near_enemy * active).sum() / near_count
                far_effect = (effect * far_entity * active).sum() / far_count
                cf_loss = F.relu(torch.as_tensor(local_action_effect_margin, device=effect.device, dtype=effect.dtype) - near_effect)
                cf_loss = cf_loss + float(local_action_drift_weight) * far_effect
                cf_losses.append(cf_loss)
                cf_counts.append((near_enemy * active).sum().detach())
                cf_near_effects.append(near_effect.detach())
                cf_far_effects.append(far_effect.detach())

            target_mask_h = target_entity_mask_seq_full[:, target_idx]
            pred_steps.append(pred_h)
            target_steps.append(target_latents[:, target_idx])
            target_entity_steps.append(target_entity_seq_full[:, target_idx])
            prev_entity_steps.append(target_entity_seq_full[:, target_idx - 1])
            last_seen_steps.append(last_seen_seq[:, target_idx])
            target_mask_steps.append(target_mask_h)
            observed_mask_steps.append(entity_mask_seq[:, target_idx])
            last_valid_steps.append(last_seen_valid_seq[:, target_idx])
            slot_mask_steps.append(entity_slot_seq[:, target_idx])
            valid_steps.append(valid_h)

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
            current_condition_mask = target_mask_h
            current_predict_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))
        target_by_start.append(torch.stack(target_steps, dim=1))
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))
        prev_entity_by_start.append(torch.stack(prev_entity_steps, dim=1))
        last_seen_by_start.append(torch.stack(last_seen_steps, dim=1))
        target_entity_mask_by_start.append(torch.stack(target_mask_steps, dim=1))
        observed_mask_by_start.append(torch.stack(observed_mask_steps, dim=1))
        last_seen_valid_by_start.append(torch.stack(last_valid_steps, dim=1))
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))
        valid_by_start.append(torch.stack(valid_steps, dim=1))

        real_action_h = action_seq[:, start_idx]
        real_action_mask_h = action_mask_seq[:, start_idx]
        if getattr(memory_module, "uses_action", False):
            main_memory = memory_module.update(
                input_latents[:, start_idx],
                main_memory,
                entity_mask_seq[:, start_idx],
                action=real_action_h,
                action_mask=real_action_mask_h,
            )
        else:
            main_memory = memory_module.update(input_latents[:, start_idx], main_memory, entity_mask_seq[:, start_idx])
        memory_norms.append(main_memory.detach().float().norm(dim=-1).mean())

    pred_latent = torch.stack(pred_by_start, dim=1)
    target_latent = torch.stack(target_by_start, dim=1)
    target_entity = torch.stack(target_entity_by_start, dim=1)
    prev_entity = torch.stack(prev_entity_by_start, dim=1)
    last_seen_entity = torch.stack(last_seen_by_start, dim=1)
    target_entity_mask = torch.stack(target_entity_mask_by_start, dim=1)
    observed_mask = torch.stack(observed_mask_by_start, dim=1)
    last_seen_valid = torch.stack(last_seen_valid_by_start, dim=1)
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)
    valid_mask = torch.stack(valid_by_start, dim=1)

    target_for_pred = target_latent.detach() if detach_rollout_targets else target_latent
    weights = temporal_time_weights(
        h,
        mode=temporal_loss_mode,
        td_lambda=td_lambda,
        flat_decay_start=flat_decay_start,
        flat_decay_final_weight=flat_decay_final_weight,
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
    decoded = model.decode_entities(pred_latent.reshape(bsz * p * h, entities, latent_dim)).reshape(bsz, p, h, entities, -1)
    aux_weights = uniform_weights if unweighted_aux_losses else weights
    decoded_loss = weighted_mse(decoded, target_entity, mask, aux_weights)
    presence_logits = model.predict_presence(pred_latent.reshape(bsz * p * h, entities, latent_dim)).reshape(bsz, p, h, entities)
    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(presence_logits, target_entity_mask, presence_mask, aux_weights)
    reg_latents = torch.cat([input_latents, target_latents], dim=1)
    reg_masks = torch.cat([entity_mask_seq, target_entity_mask_seq_full], dim=1)
    reg_loss = sigreg_loss(reg_latents, reg_masks)

    # Exp40-style decoded event/delta auxiliary pressure.
    dyn_pred = decoded[..., :dyn_dim]
    dyn_target = target_entity[..., :dyn_dim]
    dyn_prev = prev_entity[..., :dyn_dim]
    dyn_last = last_seen_entity[..., :dyn_dim]
    dyn_mask = mask[..., :1]
    change_mag = (dyn_target - dyn_prev).abs().mean(dim=-1)
    event_mask = ((change_mag > float(event_change_threshold)).float() * target_entity_mask * valid_mask.unsqueeze(-1)).unsqueeze(-1)
    event_dynamics_loss = weighted_mse(dyn_pred, dyn_target, event_mask, aux_weights)
    delta_loss = weighted_mse(dyn_pred - dyn_prev, dyn_target - dyn_prev, dyn_mask, aux_weights)

    # Action pressure with gradients into pred_latent.
    action_ctx_steps = []
    for start_idx in range(p):
        per_h = []
        for step in range(h):
            a = action_seq[:, start_idx + step]
            am = action_mask_seq[:, start_idx + step]
            per_h.append(pooled_action_context(a, am, n_actions=aux_heads.inverse[-1].out_features))
        action_ctx_steps.append(torch.stack(per_h, dim=1))
    action_ctx = torch.stack(action_ctx_steps, dim=1)  # [B,P,H,C]
    pooled_pred = pred_latent.mean(dim=3)
    pooled_tgt = target_latent.detach().mean(dim=3)
    inv_logits = aux_heads.inverse(torch.cat([pooled_pred, pooled_tgt], dim=-1))
    inverse_dynamics_loss = F.binary_cross_entropy_with_logits(inv_logits, action_ctx.float())

    # Copy-vs-update. Exp42 can use all_slots; Exp43 enemy_hidden_only.
    hidden_mask = (1.0 - observed_mask.float()) * target_entity_mask.float() * last_seen_valid.float() * valid_mask.unsqueeze(-1).float()
    scope_mask = make_entity_scope_mask(
        scope=hidden_change_scope,
        hidden_mask=hidden_mask,
        entity_slot_mask=target_entity_mask.float() * valid_mask.unsqueeze(-1).float(),
        max_agents=max_agents,
    )
    # Any copy/update target needs a real previous observation anchor.
    scope_mask = scope_mask * last_seen_valid.float()
    residual_mag = (dyn_target - dyn_last).abs().mean(dim=-1)
    changed = (residual_mag > float(hidden_change_threshold)).float() * scope_mask
    unchanged = (1.0 - (residual_mag > float(hidden_change_threshold)).float()) * scope_mask
    hidden_residual_loss = weighted_mse(dyn_pred - dyn_last, dyn_target - dyn_last, changed.unsqueeze(-1), aux_weights)
    hidden_copy_loss = weighted_mse(dyn_pred, dyn_last, unchanged.unsqueeze(-1), aux_weights)
    hidden_active_count = changed.sum().detach() + unchanged.sum().detach()
    hidden_changed_count = changed.sum().detach()
    hidden_unchanged_count = unchanged.sum().detach()

    if cf_losses:
        local_cf_loss = torch.stack(cf_losses).mean()
        local_cf_count = torch.stack(cf_counts).sum()
        local_near_effect = torch.stack(cf_near_effects).mean()
        local_far_effect = torch.stack(cf_far_effects).mean()
    else:
        local_cf_loss = torch.tensor(0.0, device=pred_latent.device, dtype=pred_latent.dtype)
        local_cf_count = torch.tensor(0.0, device=pred_latent.device, dtype=pred_latent.dtype)
        local_near_effect = torch.tensor(0.0, device=pred_latent.device, dtype=pred_latent.dtype)
        local_far_effect = torch.tensor(0.0, device=pred_latent.device, dtype=pred_latent.dtype)

    total_loss = (
        pred_loss
        + one_step_weight * one_step_loss
        + sigreg_weight * reg_loss
        + decoder_weight * decoded_loss
        + presence_weight * presence_loss
        + event_dynamics_weight * event_dynamics_loss
        + delta_loss_weight * delta_loss
        + inverse_dynamics_weight * inverse_dynamics_loss
        + hidden_change_residual_weight * hidden_residual_loss
        + hidden_change_copy_weight * hidden_copy_loss
        + local_action_counterfactual_weight * local_cf_loss
    )

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
        "event_dynamics_loss": event_dynamics_loss,
        "weighted_event_dynamics_loss": (event_dynamics_weight * event_dynamics_loss).detach(),
        "delta_loss": delta_loss,
        "weighted_delta_loss": (delta_loss_weight * delta_loss).detach(),
        "inverse_dynamics_loss": inverse_dynamics_loss,
        "weighted_inverse_dynamics_loss": (inverse_dynamics_weight * inverse_dynamics_loss).detach(),
        "hidden_residual_loss": hidden_residual_loss,
        "weighted_hidden_residual_loss": (hidden_change_residual_weight * hidden_residual_loss).detach(),
        "hidden_copy_loss": hidden_copy_loss,
        "weighted_hidden_copy_loss": (hidden_change_copy_weight * hidden_copy_loss).detach(),
        "hidden_active_count": hidden_active_count,
        "hidden_changed_count": hidden_changed_count,
        "hidden_unchanged_count": hidden_unchanged_count,
        "local_cf_loss": local_cf_loss,
        "weighted_local_cf_loss": (local_action_counterfactual_weight * local_cf_loss).detach(),
        "local_cf_count": local_cf_count.detach(),
        "local_cf_near_effect": local_near_effect.detach(),
        "local_cf_far_effect": local_far_effect.detach(),
        "event_active_count": event_mask.sum().detach(),
        "temporal_weight_sum": weights.sum().detach(),
        "rollout_window": torch.tensor(float(p), device=pred_latent.device),
        "rollout_horizon": torch.tensor(float(h), device=pred_latent.device),
        "memory_norm_mean": torch.stack(memory_norms).mean() if memory_norms else torch.tensor(0.0, device=pred_latent.device),
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


def compute_event_sampler_weights(dataset: Any, *, threshold: float, desired_fraction: float, scan_limit: int) -> list[float]:
    n = len(dataset)
    scan_n = min(n, int(scan_limit))
    flags: list[int] = []
    for i in range(scan_n):
        try:
            sample = dataset[i]
            target = sample.get("target_entity_seq", sample["entity_seq"])
            mask = sample.get("target_entity_mask_seq", sample["entity_mask_seq"])
            dyn_dim = min(4, target.shape[-1])
            diff = (target[1:, :, :dyn_dim] - target[:-1, :, :dyn_dim]).abs().mean(dim=-1)
            m = mask[1:].float()
            flags.append(int(((diff > threshold).float() * m).sum().item() > 0))
        except Exception:
            flags.append(0)
    if scan_n < n:
        # Repeat scan pattern if samples_per_epoch is longer than scan limit.
        reps = (n + scan_n - 1) // max(scan_n, 1)
        flags = (flags * reps)[:n]
    n_event = sum(flags)
    n_non = len(flags) - n_event
    if n_event == 0 or n_non == 0:
        return [1.0] * n
    desired_fraction = min(max(float(desired_fraction), 0.01), 0.99)
    event_weight = (desired_fraction / (1.0 - desired_fraction)) * (n_non / max(n_event, 1))
    return [float(event_weight if flag else 1.0) for flag in flags]


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

    data_paths = load_data_paths(args)
    # Important repair: infer caps from selected split only; avoids stale/missing paths from other manifests.
    cap_dataset = SMACJEPADataset(data_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata
    dataset = VisibilityMarkovRolloutSMACJEPADataset(
        data_paths,
        rollout_window=args.rollout_window,
        rollout_horizon=args.rollout_horizon,
        mode="entity",
        window_mode=args.window_mode,
        samples_per_epoch=args.samples_per_epoch,
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

    sampler = None
    sampler_summary = {"event_balanced": bool(args.event_balanced_sampling)}
    if args.event_balanced_sampling:
        weights = compute_event_sampler_weights(
            dataset,
            threshold=args.event_change_threshold,
            desired_fraction=args.event_fraction,
            scan_limit=args.event_sampler_scan_limit,
        )
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        sampler_summary |= {
            "sampler_weight_min": float(min(weights)),
            "sampler_weight_max": float(max(weights)),
            "sampler_weight_mean": float(sum(weights) / max(len(weights), 1)),
            "sampler_event_like_count": int(sum(1 for w in weights if abs(float(w) - 1.0) > 1e-6)),
            "sampler_len": len(weights),
        }

    loader = DataLoader(
        dataset,
        batch_size=int(arch["batch_size"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
    )

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
    aux_heads = RepairedAuxHeads(int(arch["latent_dim"]), dataset.metadata.n_actions, args.inverse_dynamics_hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(memory_module.parameters()) + list(aux_heads.parameters()),
        lr=float(arch["lr"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, global_step = 1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "memory_module_state" in ckpt:
            memory_module.load_state_dict(ckpt["memory_module_state"])
        if "aux_heads_state" in ckpt:
            aux_heads.load_state_dict(ckpt["aux_heads_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and amp_enabled:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "repaired_exp42_44_seqmem_exp40_base",
        "objective_family": "r2offline_repaired",
        "r2_latent_normalize": True,
        "two_mask_rollout_prediction": True,
        "main_rollout_uses_full_slot_mask_for_prediction": True,
        "sampler_summary": sampler_summary,
        "dynamic_token_dim": int(dataset.metadata.dynamic_token_dim),
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("wandb requested but unavailable")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or out_dir.name,
            config=saved_config,
            mode=args.wandb_mode,
            dir=str(out_dir),
        )

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
                "epoch": epoch_to_save,
                "global_step": global_step,
            },
            checkpoint_path,
        )

    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")
    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []
    model.train(); memory_module.train(); aux_heads.train()
    print(
        "repaired_exp42_44_seqmem "
        f"experiment={args.experiment_name} p={args.rollout_window} h={args.rollout_horizon} "
        f"event_balanced={args.event_balanced_sampling} event_dyn={args.event_dynamics_weight} "
        f"delta={args.delta_loss_weight} inv={args.inverse_dynamics_weight} "
        f"hidden_resid={args.hidden_change_residual_weight} hidden_copy={args.hidden_change_copy_weight} "
        f"scope={args.hidden_change_scope} local_cf={args.local_action_counterfactual_weight} "
        f"sampler={sampler_summary}",
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
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        for batch in loader:
            global_step += 1
            epoch_batches += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with autocast_context:
                losses = repaired_rollout_losses(
                    model,
                    memory_module,
                    aux_heads,
                    batch,
                    rollout_window=args.rollout_window,
                    rollout_horizon=args.rollout_horizon,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=args.td_lambda,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                    sigreg_weight=active_sigreg_weight,
                    decoder_weight=args.decoder_weight,
                    presence_weight=args.presence_weight,
                    one_step_weight=args.one_step_weight,
                    target_mode=args.target_mode,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                    dynamic_token_dim=dataset.metadata.dynamic_token_dim,
                    max_agents=dataset.metadata.max_agents,
                    event_change_threshold=args.event_change_threshold,
                    event_dynamics_weight=args.event_dynamics_weight,
                    delta_loss_weight=args.delta_loss_weight,
                    inverse_dynamics_weight=args.inverse_dynamics_weight,
                    hidden_change_residual_weight=args.hidden_change_residual_weight,
                    hidden_change_copy_weight=args.hidden_change_copy_weight,
                    hidden_change_threshold=args.hidden_change_threshold,
                    hidden_change_scope=args.hidden_change_scope,
                    local_action_counterfactual_weight=args.local_action_counterfactual_weight,
                    local_action_neighbor_radius=args.local_action_neighbor_radius,
                    local_action_drift_weight=args.local_action_drift_weight,
                    local_action_effect_margin=args.local_action_effect_margin,
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
            row: dict[str, float | int] = {"epoch": epoch, "step": global_step, "active_sigreg_weight": active_sigreg_weight}
            for k, v in losses.items():
                row[k] = float(v.detach().cpu())
            logger.log(row)
            step_rows.append(row)
            for k, v in row.items():
                if k in {"epoch", "step"}:
                    continue
                epoch_sums[k] = epoch_sums.get(k, 0.0) + float(v)
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in row.items()}, step=global_step)
            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    "epoch={epoch} step={step} total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "decoded={decoded_loss:.6f} event={event_dynamics_loss:.6f} delta={delta_loss:.6f} "
                    "inv={inverse_dynamics_loss:.6f} hidden_resid={hidden_residual_loss:.6f} "
                    "hidden_copy={hidden_copy_loss:.6f} hidden_count={hidden_active_count:.1f} "
                    "hidden_changed={hidden_changed_count:.1f} hidden_unchanged={hidden_unchanged_count:.1f} "
                    "local_cf={local_cf_loss:.6f} local_count={local_cf_count:.1f} "
                    "local_near={local_cf_near_effect:.6f} local_far={local_cf_far_effect:.6f}".format(**row),
                    flush=True,
                )

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save checkpoint.")
        epoch_row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
        for k, v in epoch_sums.items():
            epoch_row[k] = v / max(epoch_batches, 1)
        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row)
        if wandb_run is not None:
            wandb_run.log({f"epoch/{k}": v for k, v in epoch_row.items()}, step=global_step)
        print(
            "epoch_summary epoch={epoch} step={step} total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "event={event_dynamics_loss:.6f} delta={delta_loss:.6f} inv={inverse_dynamics_loss:.6f} "
            "hidden_resid={hidden_residual_loss:.6f} hidden_copy={hidden_copy_loss:.6f} "
            "hidden_count={hidden_active_count:.1f} hidden_changed={hidden_changed_count:.1f} "
            "hidden_unchanged={hidden_unchanged_count:.1f} local_cf={local_cf_loss:.6f} "
            "local_count={local_cf_count:.1f} local_near={local_cf_near_effect:.6f} "
            "local_far={local_cf_far_effect:.6f}".format(**epoch_row),
            flush=True,
        )
        if args.audit_strict:
            if (args.hidden_change_residual_weight > 0 or args.hidden_change_copy_weight > 0):
                if float(epoch_row.get("hidden_active_count", 0.0)) < args.audit_min_hidden_count:
                    raise RuntimeError(
                        "AUDIT FAIL: hidden copy/update requested but hidden_active_count is too low. "
                        f"count={epoch_row.get('hidden_active_count')}"
                    )
                if args.hidden_change_residual_weight > 0 and float(epoch_row.get("hidden_changed_count", 0.0)) < args.audit_min_hidden_changed_count:
                    raise RuntimeError(
                        "AUDIT FAIL: hidden residual/update requested but hidden_changed_count is too low. "
                        f"count={epoch_row.get('hidden_changed_count')}"
                    )
                if args.hidden_change_copy_weight > 0 and float(epoch_row.get("hidden_unchanged_count", 0.0)) < args.audit_min_hidden_unchanged_count:
                    raise RuntimeError(
                        "AUDIT FAIL: hidden copy requested but hidden_unchanged_count is too low. "
                        f"count={epoch_row.get('hidden_unchanged_count')}"
                    )
            if args.local_action_counterfactual_weight > 0:
                if float(epoch_row.get("local_cf_count", 0.0)) < args.audit_min_counterfactual_count:
                    raise RuntimeError(
                        "AUDIT FAIL: local counterfactual requested but local_cf_count is too low. "
                        f"count={epoch_row.get('local_cf_count')}"
                    )
            if args.event_balanced_sampling and sampler_summary.get("sampler_event_like_count", 0) == 0:
                print("AUDIT WARN: event-balanced sampler found zero event-like samples", flush=True)

        epoch_ckpt = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_ckpt)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_ckpt} and {out_dir / 'checkpoint.pt'}", flush=True)

    write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
    write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
    write_svg_line_plot(step_rows, "step", "pred_loss", "Prediction Loss Per Step", out_dir / "pred_loss_by_step.svg")
    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
