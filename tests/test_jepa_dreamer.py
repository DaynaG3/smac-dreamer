import argparse
import pathlib
import sys

import pytest
import torch
from gymnasium import spaces

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "scripts"):
    sys.path.insert(0, str(p))
JEPA = pathlib.Path("/Users/kialok/Desktop/NUS Mods/smac-jepa-wm-main")
if JEPA.exists():
    sys.path.insert(0, str(JEPA))

from dreamer import Dreamer
from smacdreamer.jepa.memory import EntityRolloutGRUMemory
from train_r2dreamer_smaclite_debug import make_config


def _checkpoint(path):
    if not JEPA.exists():
        pytest.skip("local smac-jepa-wm checkout is not available")
    from smac_jepa.jepa import SMACJEPA

    meta = {
        "state_dim": 8, "n_agents": 2, "n_actions": 3, "n_enemies": 1,
        "ally_state_feat_size": 3, "enemy_state_feat_size": 2,
        "ally_has_shields": False, "enemy_has_shields": False, "num_unit_types": 0,
        "max_agents": 2, "max_enemies": 1, "max_actions": 3,
        "token_dim": 5, "dynamic_token_dim": 3, "static_dim": 4,
        "entity_static_feat_size": 2, "mode": "entity",
    }
    cfg = {"latent_dim": 6, "hidden_dim": 8, "action_dim": 4, "num_heads": 2, "rollout_memory_dim": 7}
    model = SMACJEPA(
        state_dim=8, n_agents=2, n_actions=3, latent_dim=6, hidden_dim=8, action_dim=4,
        num_heads=2, mode="entity", max_agents=2, max_enemies=1, max_actions=3,
        token_dim=5, static_dim=4,
    )
    memory = EntityRolloutGRUMemory(latent_dim=6, memory_dim=7)
    torch.save({"model_state": model.state_dict(), "memory_module_state": memory.state_dict(),
                "metadata": meta, "resolved_config": cfg}, path)


def test_jepa_dreamer_constructs_and_keeps_core_eval(tmp_path):
    ckpt = tmp_path / "jepa.pt"
    _checkpoint(ckpt)
    cfg = make_config(argparse.Namespace(steps=10, batch_size=1, batch_length=2, units=16, deter=32, imag_horizon=2))
    cfg.model.action_masking = True
    cfg.model.world_model = {
        "backend": "jepa",
        "jepa": {
            "checkpoint": str(ckpt),
            "strict_checkpoint": True,
            "freeze_core": True,
            "presence_threshold": 0.5,
            "feature_dim": 64,
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
