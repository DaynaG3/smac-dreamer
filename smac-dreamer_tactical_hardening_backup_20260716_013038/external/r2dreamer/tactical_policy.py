"""Shared tactical latent policy for the centralized R2-Dreamer actor.

This module is intentionally independent of JEPA, replay, SMAClite, and action
mask construction. It selects one team-level discrete tactic and produces a
zero-initialized residual over the existing primitive-action logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.distributions import Categorical


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass(frozen=True)
class TacticalSettings:
    enabled: bool = False
    num_tactics: int = 4
    embedding_dim: int = 16
    hidden_dim: int = 128
    tactic_pg_scale: float = 1.0
    tactic_entropy_scale: float = 3.0e-4
    balance_loss_scale: float = 1.0e-3
    effect_loss_scale: float = 1.0e-3
    effect_target: float = 0.02
    max_effect_states: int = 256
    duration: int = 1

    @classmethod
    def from_config(cls, cfg: Any) -> "TacticalSettings":
        return cls(
            enabled=bool(_cfg_get(cfg, "enabled", False)),
            num_tactics=int(_cfg_get(cfg, "num_tactics", 4)),
            embedding_dim=int(_cfg_get(cfg, "embedding_dim", 16)),
            hidden_dim=int(_cfg_get(cfg, "hidden_dim", 128)),
            tactic_pg_scale=float(_cfg_get(cfg, "tactic_pg_scale", 1.0)),
            tactic_entropy_scale=float(_cfg_get(cfg, "tactic_entropy_scale", 3.0e-4)),
            balance_loss_scale=float(_cfg_get(cfg, "balance_loss_scale", 1.0e-3)),
            effect_loss_scale=float(_cfg_get(cfg, "effect_loss_scale", 1.0e-3)),
            effect_target=float(_cfg_get(cfg, "effect_target", 0.02)),
            max_effect_states=int(_cfg_get(cfg, "max_effect_states", 256)),
            duration=int(_cfg_get(cfg, "duration", 1)),
        )

    def validate(self) -> None:
        if self.num_tactics < 2:
            raise ValueError("tactical_mixture.num_tactics must be >= 2")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("tactical embedding/hidden dimensions must be positive")
        if self.duration != 1:
            raise ValueError(
                "Tactical Mixture v1 supports duration=1 only. Persistent "
                "tactics belong to the later hierarchical extension."
            )
        for name in (
            "tactic_pg_scale",
            "tactic_entropy_scale",
            "balance_loss_scale",
            "effect_loss_scale",
            "effect_target",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"tactical_mixture.{name} must be non-negative")
        if self.max_effect_states <= 0:
            raise ValueError("tactical_mixture.max_effect_states must be positive")


class TacticalMixturePolicy(nn.Module):
    """Team-level categorical selector and tactic-conditioned logit residual."""

    SCHEMA_VERSION = 1

    def __init__(self, feature_dim: int, action_logit_dim: int, config: Any) -> None:
        super().__init__()
        self.settings = TacticalSettings.from_config(config)
        self.settings.validate()
        self.feature_dim = int(feature_dim)
        self.action_logit_dim = int(action_logit_dim)
        if self.feature_dim <= 0 or self.action_logit_dim <= 0:
            raise ValueError("feature_dim and action_logit_dim must be positive")

        k = self.settings.num_tactics
        hidden = self.settings.hidden_dim
        emb = self.settings.embedding_dim
        self.selector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, k),
        )
        self.embedding = nn.Embedding(k, emb)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim + emb, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.action_logit_dim),
        )
        self.reset_parameters()

    @property
    def num_tactics(self) -> int:
        return self.settings.num_tactics

    def reset_parameters(self) -> None:
        # Uniform initial tactic selector.
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        # Exact legacy-policy equivalence for every tactic.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def selector_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"tactical feature dim {feat.shape[-1]} != expected {self.feature_dim}"
            )
        return self.selector(feat.float())

    def selector_dist(self, feat: torch.Tensor) -> Categorical:
        return Categorical(logits=self.selector_logits(feat))

    def select_tactic(self, feat: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        logits = self.selector_logits(feat)
        return logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()

    def residual_logits(self, feat: torch.Tensor, tactic: torch.Tensor) -> torch.Tensor:
        tactic = tactic.to(device=feat.device, dtype=torch.long)
        if tactic.shape != feat.shape[:-1]:
            raise ValueError(
                f"tactic shape {tuple(tactic.shape)} must equal feature leading "
                f"shape {tuple(feat.shape[:-1])}"
            )
        emb = self.embedding(tactic)
        return self.residual(torch.cat([feat.float(), emb], dim=-1))

    def combine_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
        tactic: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape[:-1] != feat.shape[:-1]:
            raise ValueError("base-logit and feature leading shapes differ")
        if base_logits.shape[-1] != self.action_logit_dim:
            raise ValueError("base-logit action dimension is incompatible")
        residual = self.residual_logits(feat, tactic).to(base_logits.dtype)
        return base_logits + residual

    def all_combined_logits(self, base_logits: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Return tactic-conditioned logits with shape (..., K, action_dim)."""
        leading = feat.shape[:-1]
        k = self.num_tactics
        feat_all = feat.unsqueeze(-2).expand(*leading, k, self.feature_dim)
        base_all = base_logits.unsqueeze(-2).expand(*leading, k, self.action_logit_dim)
        tactic = torch.arange(k, device=feat.device, dtype=torch.long)
        tactic = tactic.view(*([1] * len(leading)), k).expand(*leading, k)
        return self.combine_logits(base_all, feat_all, tactic)

    @staticmethod
    def _weights(
        weights: torch.Tensor | None,
        shape: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        if weights is None:
            out = torch.ones(tuple(shape), device=device, dtype=torch.float32)
        else:
            out = weights.detach().to(device=device, dtype=torch.float32)
            out = torch.broadcast_to(out, tuple(shape))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)

    def balance_loss(
        self,
        tactic_logits: torch.Tensor,
        state_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL of weighted marginal tactic usage to the uniform distribution."""
        probs = tactic_logits.float().softmax(dim=-1)
        weights = self._weights(state_weights, probs.shape[:-1], probs.device)
        denominator = weights.sum().clamp_min(1.0)
        marginal = (
            probs * weights.unsqueeze(-1)
        ).reshape(-1, self.num_tactics).sum(0) / denominator
        marginal = marginal / marginal.sum().clamp_min(1e-8)
        uniform_log = -torch.log(
            torch.tensor(float(self.num_tactics), device=probs.device)
        )
        loss = (
            marginal.clamp_min(1e-8)
            * (marginal.clamp_min(1e-8).log() - uniform_log)
        ).sum()
        return loss, marginal

    def effect_js(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
        action_mask: torch.Tensor,
        agent_active: torch.Tensor,
        actor_shape: Sequence[int],
        state_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked mean pairwise JS divergence between tactic-conditioned policies."""
        actor_shape = tuple(int(value) for value in actor_shape)
        num_agents = len(actor_shape)
        if num_agents == 0 or len(set(actor_shape)) != 1:
            raise ValueError("tactical effect metric requires equal action sizes per agent")
        num_actions = actor_shape[0]
        if num_agents * num_actions != self.action_logit_dim:
            raise ValueError("actor shape does not match action-logit dimension")

        flat_feat = feat.reshape(-1, self.feature_dim)
        flat_base = base_logits.reshape(-1, self.action_logit_dim)
        flat_mask = action_mask.reshape(-1, num_agents, num_actions).bool()
        flat_active = agent_active.reshape(-1, num_agents).bool()
        flat_weights = self._weights(
            state_weights, feat.shape[:-1], feat.device
        ).reshape(-1)

        valid = flat_active.any(-1) & (flat_weights > 0)
        indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return flat_feat.sum() * 0.0
        max_states = min(self.settings.max_effect_states, int(indices.numel()))
        if indices.numel() > max_states:
            pick = torch.linspace(
                0, indices.numel() - 1, max_states, device=indices.device
            ).round().long()
            indices = indices[pick]

        feat_sel = flat_feat[indices]
        base_sel = flat_base[indices]
        mask_sel = flat_mask[indices]
        active_sel = flat_active[indices]
        weight_sel = flat_weights[indices]

        logits = self.all_combined_logits(base_sel, feat_sel)
        logits = logits.reshape(-1, self.num_tactics, num_agents, num_actions)
        mask = mask_sel.unsqueeze(1).expand_as(logits)
        probs = logits.masked_fill(~mask, -1.0e9).float().softmax(dim=-1)

        active_weight = active_sel.float() * weight_sel.unsqueeze(-1)
        denominator = active_weight.sum().clamp_min(1.0)
        total = probs.sum() * 0.0
        pairs = 0
        eps = 1e-8
        for left in range(self.num_tactics):
            for right in range(left + 1, self.num_tactics):
                p = probs[:, left].clamp_min(eps)
                q = probs[:, right].clamp_min(eps)
                middle = 0.5 * (p + q)
                js = 0.5 * (
                    (p * (p.log() - middle.log())).sum(-1)
                    + (q * (q.log() - middle.log())).sum(-1)
                )
                total = total + (js * active_weight).sum() / denominator
                pairs += 1
        return total / max(pairs, 1)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "architecture": "tactical_mixture_v1",
            "feature_dim": self.feature_dim,
            "action_logit_dim": self.action_logit_dim,
            "num_tactics": self.settings.num_tactics,
            "embedding_dim": self.settings.embedding_dim,
            "hidden_dim": self.settings.hidden_dim,
            "duration": self.settings.duration,
        }

    def assert_legacy_equivalence_ready(self) -> None:
        for name, layer in (("selector", self.selector[-1]), ("residual", self.residual[-1])):
            if torch.count_nonzero(layer.weight).item() != 0:
                raise RuntimeError(f"{name} final weight is not zero-initialized")
            if torch.count_nonzero(layer.bias).item() != 0:
                raise RuntimeError(f"{name} final bias is not zero-initialized")
