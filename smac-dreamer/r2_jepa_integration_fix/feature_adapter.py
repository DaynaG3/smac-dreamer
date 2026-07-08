from __future__ import annotations

import torch
from torch import nn


class JEPAFeatureAdapter(nn.Module):
    """Slot-preserving adapter from frozen JEPA entity state to R2-Dreamer feature.

    The previous adapter mean-pooled all entity slots before projecting to the R2
    feature vector. That destroyed ally/enemy identity and made it very hard for
    the centralised R2 actor/value/reward heads to learn per-agent control from
    JEPA slots.

    This adapter instead:
      1. embeds each entity slot independently,
      2. multiplies only by the belief/presence mask supplied by world_model.py,
      3. flattens the ordered slots, preserving slot identity,
      4. projects the whole ordered slot table + static condition to out_dim.

    Output is still a single global feature vector, so it remains compatible with
    the existing R2-Dreamer heads. The important change is that the global vector
    now contains ordered slot information instead of only a masked mean.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        static_dim: int,
        out_dim: int,
        num_entities: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.static_dim = int(static_dim)
        self.out_dim = int(out_dim)
        self.num_entities = int(num_entities)
        self.hidden_dim = int(hidden_dim or max(64, min(512, out_dim), latent_dim + memory_dim))

        self.entity_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + self.memory_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.num_entities * self.hidden_dim + self.static_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(
        self,
        conditioned_z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor,
        static_condition: torch.Tensor,
    ) -> torch.Tensor:
        if conditioned_z.ndim != 3:
            raise ValueError(f"conditioned_z must be [B,E,Z], got {tuple(conditioned_z.shape)}")
        if memory.ndim != 3:
            raise ValueError(f"memory must be [B,E,M], got {tuple(memory.shape)}")
        if entity_mask.ndim != 2:
            raise ValueError(f"entity_mask must be [B,E], got {tuple(entity_mask.shape)}")
        if conditioned_z.shape[:2] != memory.shape[:2]:
            raise ValueError(
                "conditioned_z and memory must share [B,E]: "
                f"{tuple(conditioned_z.shape)} vs {tuple(memory.shape)}"
            )
        if conditioned_z.shape[1] != self.num_entities:
            raise ValueError(
                f"expected {self.num_entities} entity slots, got {conditioned_z.shape[1]}"
            )
        if entity_mask.shape != conditioned_z.shape[:2]:
            raise ValueError(
                f"entity_mask shape {tuple(entity_mask.shape)} incompatible with slots "
                f"{tuple(conditioned_z.shape[:2])}"
            )

        dtype = conditioned_z.dtype
        x = torch.cat([conditioned_z, memory.to(dtype=dtype)], dim=-1)
        x = self.entity_mlp(x)

        # This mask should be the belief/presence exposure mask, not raw current
        # visibility. world_model.py is responsible for constructing it.
        mask = entity_mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * mask

        # Preserve ordered entity slots instead of mean pooling them away.
        x = x.reshape(x.shape[0], self.num_entities * self.hidden_dim)
        static = static_condition.to(dtype=x.dtype)
        feat = self.proj(torch.cat([x, static], dim=-1))

        if not torch.isfinite(feat).all():
            raise FloatingPointError("non-finite JEPA feature adapter output")
        return feat
