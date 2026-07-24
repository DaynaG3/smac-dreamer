from __future__ import annotations

import torch
from torch import nn

from .action_adapter import JEPAActionAdapter
from .feature_adapter import JEPAFeatureAdapter
from .online_tokens import JEPAVisibilityConfig
from .state import JEPAStateSpec, pack_state, unpack_state


class FrozenJEPAWorldModel(nn.Module):
    """RSSM-compatible operational adapter around a frozen pretrained JEPA.

    Integration rules:
      - Raw observation visibility is still used for encoding/update.
      - Hidden-but-previously-seen entities must remain exposed to actor/value/
        reward through a belief/presence mask.
      - The frozen JEPA core and memory remain no-grad.
      - The feature adapter is trainable and must NOT be under no_grad.
    """

    def __init__(
        self,
        *,
        core: nn.Module,
        memory_module: nn.Module,
        info,
        feature_dim: int,
        presence_threshold: float = 0.5,
    ):
        super().__init__()
        self.core = core
        self.memory_module = memory_module
        self.info = info
        meta = info.metadata
        self.entities = int(meta["max_agents"]) + int(meta["max_enemies"])
        self.latent_dim = int(info.latent_dim)
        self.memory_dim = int(info.memory_dim)
        self.static_dim = int(meta.get("static_dim", 0))
        self.max_agents = int(meta["max_agents"])
        self.max_actions = int(meta["max_actions"])
        self.presence_threshold = float(presence_threshold)

        anchored = bool(info.resolved_config.get("anchored_belief_memory", False))
        default_presence_mode = "soft" if anchored else "hard"
        self.presence_rollout_mode = str(
            info.resolved_config.get("presence_rollout_mode", default_presence_mode)
        ).strip().lower()
        if self.presence_rollout_mode not in {"soft", "hard"}:
            raise ValueError(
                "Unsupported JEPA presence_rollout_mode: "
                f"{self.presence_rollout_mode!r}"
            )

        self.visibility_config = JEPAVisibilityConfig(
            enemy_visibility_mask=bool(meta.get("enemy_visibility_mask", False)),
            enemy_sight_range=float(meta.get("enemy_sight_range", 9.0)),
            xy_indices=tuple(meta.get("visibility_xy_indices", (2, 3))),
        )
        self.state_spec = JEPAStateSpec(
            self.entities, self.latent_dim, self.memory_dim, self.static_dim
        )
        self.action_adapter = JEPAActionAdapter(
            max_agents=self.max_agents,
            max_actions=self.max_actions,
            checkpoint_n_actions=int(meta["n_actions"]),
        )
        self.feature_adapter = JEPAFeatureAdapter(
            latent_dim=self.latent_dim,
            memory_dim=self.memory_dim,
            static_dim=self.static_dim,
            out_dim=int(feature_dim),
            num_entities=self.entities,
        )
        self.feat_size = int(feature_dim)
        self.flat_stoch = self.entities * self.latent_dim
        self._freeze_core()

    def _freeze_core(self) -> None:
        for module in (self.core, self.memory_module):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.train(False)
        self.memory_module.train(False)
        return self

    def parameters_frozen(self):
        yield from self.core.parameters()
        yield from self.memory_module.parameters()

    def _seen_mask_from_memory(self, memory: torch.Tensor) -> torch.Tensor | None:
        """Return anchored-memory seen mask [B,E], if available.

        Exp33 anchored memory stores a `seen` scalar per entity. We avoid hard
        coding memory offsets by using the memory module's private splitter when
        present. If this is not an anchored memory module, return None and the
        caller falls back to the current exposure mask.
        """
        if not getattr(self.memory_module, "anchored_belief_memory", False):
            return None
        split = getattr(self.memory_module, "_split", None)
        if split is None:
            return None
        try:
            _, _, seen, _ = split(memory)
        except Exception:
            return None
        return (seen.squeeze(-1) > 0.5).to(dtype=memory.dtype)

    def _belief_mask(
        self,
        visibility_or_exposure_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Build the mask used to expose entity slots to policy/value/reward.

        This deliberately differs from raw current visibility:
          belief_mask = visible/exposed_now OR (previously_seen AND structural_slot)

        For non-anchored memories, this reduces to the incoming mask times the
        structural slot mask. For anchored Exp33 memory, hidden-but-seen enemies
        stay active in the feature path.
        """
        base = visibility_or_exposure_mask.to(dtype=memory.dtype) * slot_mask.to(dtype=memory.dtype)
        seen = self._seen_mask_from_memory(memory)
        if seen is None:
            return base
        return torch.maximum(base, seen * slot_mask.to(dtype=memory.dtype))

    @torch.no_grad()
    def initial(self, batch_size: int, *, device=None, dtype=torch.float32):
        device = device or next(self.feature_adapter.parameters()).device
        z = torch.zeros(batch_size, self.entities, self.latent_dim, device=device, dtype=dtype)
        memory = self.memory_module.initial_memory(
            batch_size, self.entities, device=device, dtype=dtype
        )
        entity_mask = torch.zeros(batch_size, self.entities, device=device, dtype=dtype)
        slot_mask = torch.zeros(batch_size, self.entities, device=device, dtype=dtype)
        static = torch.zeros(batch_size, self.static_dim, device=device, dtype=dtype)
        return z, pack_state(memory, entity_mask, slot_mask, static)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        entity = obs["jepa_entity"]
        mask = obs["jepa_entity_mask"].to(dtype=entity.dtype)
        squeeze_time = False
        if entity.ndim == 3:
            entity = entity.unsqueeze(1)
            mask = mask.unsqueeze(1)
            squeeze_time = True
        with torch.no_grad():
            z = self.core.encoder(entity, mask)
        if squeeze_time:
            z = z[:, 0]
            mask = mask[:, 0]
            slot_mask = obs["jepa_entity_slot_mask"].to(dtype=entity.dtype)
            static_condition = obs["jepa_static_condition"].to(dtype=entity.dtype)
        else:
            slot_mask = obs["jepa_entity_slot_mask"].to(dtype=entity.dtype)
            static_condition = obs["jepa_static_condition"].to(dtype=entity.dtype)
        if not torch.isfinite(z).all():
            raise FloatingPointError("non-finite JEPA encoder latent")
        return {
            "z": z.detach(),
            # This is raw current observation visibility/presence from tokenizer.
            "entity_mask": mask.detach(),
            "slot_mask": slot_mask.detach(),
            "static_condition": static_condition.detach(),
        }

    def obs_step(self, prev_z, prev_deter, prev_action, encoded, reset):
        memory, prev_exposure_mask, slot_mask, static = unpack_state(prev_deter, self.state_spec)
        reset = reset.to(dtype=prev_z.dtype).reshape(prev_z.shape[0], 1, 1)
        memory = torch.where(reset > 0, torch.zeros_like(memory), memory)
        prev_z = torch.where(reset > 0, torch.zeros_like(prev_z), prev_z)

        z = encoded["z"].detach()
        cur_visibility_mask = encoded["entity_mask"].detach()
        cur_slot = encoded["slot_mask"].detach()
        cur_static = encoded["static_condition"].detach()

        action, action_mask = self.action_adapter.flat_to_jepa(
            prev_action, cur_slot[:, : self.max_agents]
        )
        with torch.no_grad():
            # For the memory update, use the previous state's exposure mask. In
            # anchored memory this lets previous hidden-belief state participate
            # in the update contract instead of treating all hidden entities as
            # absent merely because they are not currently visible.
            if getattr(self.memory_module, "uses_action", False):
                next_memory = self.memory_module.update(
                    prev_z.detach(),
                    memory,
                    prev_exposure_mask,
                    action=action,
                    action_mask=action_mask,
                )
            else:
                next_memory = self.memory_module.update(
                    prev_z.detach(), memory, prev_exposure_mask
                )
            next_memory = torch.where(reset > 0, torch.zeros_like(next_memory), next_memory)

            # Pack belief exposure for downstream get_feat(), not raw visibility.
            cur_exposure_mask = self._belief_mask(cur_visibility_mask, cur_slot, next_memory)

        return z, pack_state(next_memory.detach(), cur_exposure_mask.detach(), cur_slot, cur_static)

    def observe(self, encoded_sequence, action_sequence, initial_state, reset_sequence):
        obs_len = int(encoded_sequence["z"].shape[1])
        if int(action_sequence.shape[1]) != obs_len:
            raise ValueError(
                "JEPA observe requires one previous action per observation: "
                f"got actions={int(action_sequence.shape[1])}, observations={obs_len}. "
                "Use [zero_initial_action, a0, a1, ...] for states [s0, s1, s2, ...]."
            )
        if int(reset_sequence.shape[1]) != obs_len:
            raise ValueError(
                f"JEPA observe reset length {int(reset_sequence.shape[1])} != observation length {obs_len}"
            )
        z_prev, deter_prev = initial_state
        zs, deters = [], []
        for t in range(obs_len):
            encoded_t = {k: v[:, t] for k, v in encoded_sequence.items()}
            z_prev, deter_prev = self.obs_step(
                z_prev,
                deter_prev,
                action_sequence[:, t],
                encoded_t,
                reset_sequence[:, t],
            )
            zs.append(z_prev)
            deters.append(deter_prev)
        return torch.stack(zs, 1), torch.stack(deters, 1)

    def img_step(self, z, deter, action):
        memory, exposure_mask, slot_mask, static = unpack_state(deter, self.state_spec)
        action_jepa, action_mask = self.action_adapter.flat_to_jepa(
            action, slot_mask[:, : self.max_agents]
        )
        with torch.no_grad():
            belief_mask = self._belief_mask(exposure_mask, slot_mask, memory)
            conditioned = self.memory_module.condition(z, memory, belief_mask)
            pred = self.core.predictor(
                conditioned.unsqueeze(1),
                action_jepa.unsqueeze(1),
                action_mask.unsqueeze(1),
                torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype),
                belief_mask.unsqueeze(1),
                static,
            )[:, 0]
            logits = self.core.predict_presence(pred)
            presence_probability = torch.sigmoid(logits).to(dtype=z.dtype)
            if self.presence_rollout_mode == "soft":
                next_visible_or_present = presence_probability * slot_mask
            else:
                next_visible_or_present = (
                    presence_probability >= self.presence_threshold
                ).to(dtype=z.dtype) * slot_mask
            pred = pred * next_visible_or_present.unsqueeze(-1)

            if getattr(self.memory_module, "uses_action", False):
                next_memory = self.memory_module.update(
                    pred,
                    memory,
                    next_visible_or_present,
                    action=action_jepa,
                    action_mask=action_mask,
                )
            else:
                next_memory = self.memory_module.update(
                    pred, memory, next_visible_or_present
                )
            next_exposure_mask = self._belief_mask(
                next_visible_or_present, slot_mask, next_memory
            )

        if not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite JEPA predicted latent")
        return pred.detach(), pack_state(
            next_memory.detach(), next_exposure_mask.detach(), slot_mask, static
        )

    def imagine_with_action(self, z, deter, action_sequence):
        zs, deters = [], []
        for t in range(action_sequence.shape[1]):
            z, deter = self.img_step(z, deter, action_sequence[:, t])
            zs.append(z)
            deters.append(deter)
        return torch.stack(zs, 1), torch.stack(deters, 1)

    def get_feat(self, z, deter):
        memory, exposure_mask, slot_mask, static = unpack_state(
            deter.reshape(-1, deter.shape[-1]), self.state_spec
        )
        z_flat = z.reshape(memory.shape[0], self.entities, self.latent_dim)

        # Frozen JEPA memory/core conditioning: no gradients needed.
        with torch.no_grad():
            belief_mask = self._belief_mask(exposure_mask, slot_mask, memory)
            conditioned = self.memory_module.condition(z_flat, memory, belief_mask)

        # Trainable R2 interface: this must remain outside no_grad.
        feat = self.feature_adapter(
            conditioned.detach(),
            memory.detach(),
            belief_mask.detach(),
            static.detach(),
        )
        return feat.reshape(*deter.shape[:-1], -1)
