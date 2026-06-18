import argparse
import pathlib
import sys

import torch
from gymnasium import spaces
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from dreamer import Dreamer
from smacdreamer.jepa.checkpoint import JEPACheckpointInfo
from smacdreamer.jepa.memory import EntityRolloutGRUMemory
from train_r2dreamer_smaclite_debug import make_config


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(5, 6)

            def forward(self, entity, mask):
                return self.proj(entity) * mask.unsqueeze(-1)

        self.encoder = Encoder()
        self.predictor = nn.Linear(9, 6)
        self.presence = nn.Linear(6, 1)

    def predict_presence(self, latents):
        return self.presence(latents).squeeze(-1)


def _metadata():
    return {
        "state_dim": 8,
        "n_agents": 2,
        "n_enemies": 1,
        "n_actions": 3,
        "ally_state_feat_size": 3,
        "enemy_state_feat_size": 2,
        "ally_has_shields": False,
        "enemy_has_shields": False,
        "num_unit_types": 0,
        "max_agents": 2,
        "max_enemies": 1,
        "max_actions": 3,
        "token_dim": 5,
        "dynamic_token_dim": 3,
        "static_dim": 4,
        "entity_static_feat_size": 2,
        "mode": "entity",
        "latent_dim": 6,
        "memory_dim": 7,
        "action_conditioned_memory": False,
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
        "latent_normalization": "none",
    }


def test_jepa_dreamer_constructs_and_keeps_core_eval(monkeypatch, tmp_path):
    import smacdreamer.jepa.checkpoint as checkpoint_mod

    meta = _metadata()

    def fake_loader(*args, **kwargs):
        return (
            TinyCore(),
            EntityRolloutGRUMemory(latent_dim=6, memory_dim=7),
            JEPACheckpointInfo("synthetic", "0" * 64, meta, {}, "synthetic", False, 6, 7, 3),
        )

    monkeypatch.setattr(checkpoint_mod, "load_frozen_jepa_checkpoint", fake_loader)
    cfg = make_config(argparse.Namespace(steps=10, batch_size=1, batch_length=2, units=16, deter=32, imag_horizon=2))
    cfg.model.action_masking = True
    cfg.model.world_model = {
        "backend": "jepa",
        "jepa": {
            "checkpoint": str(tmp_path / "unused.pt"),
            "strict_checkpoint": True,
            "freeze_core": True,
            "presence_threshold": 0.5,
            "feature_dim": 64,
            "live_metadata": meta,
        },
    }
    obs_space = spaces.Dict({
        "jepa_entity": spaces.Box(-10, 10, shape=(3, 5), dtype=float),
        "jepa_entity_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_entity_slot_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_static_condition": spaces.Box(-10, 10, shape=(4,), dtype=float),
        "avail_actions": spaces.Box(0, 1, shape=(6,), dtype=float),
        "agent_slot_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "agent_alive_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
    })
    act_space = spaces.MultiDiscrete([3, 3])
    act_space.multi_discrete = True
    agent = Dreamer(cfg.model, obs_space, act_space)
    assert agent.world_model_backend == "jepa"
    agent.train()
    assert not agent.jepa_world_model.core.training
    assert sum(p.numel() for p in agent.jepa_world_model.parameters_frozen() if p.requires_grad) == 0
