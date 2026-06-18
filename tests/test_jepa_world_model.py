import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
JEPA = pathlib.Path("/Users/kialok/Desktop/NUS Mods/smac-jepa-wm-main")
if JEPA.exists():
    sys.path.insert(0, str(JEPA))

from smacdreamer.jepa.checkpoint import JEPACheckpointInfo
from smacdreamer.jepa.memory import EntityRolloutGRUMemory
from smacdreamer.jepa.world_model import FrozenJEPAWorldModel


def _model():
    if not JEPA.exists():
        pytest.skip("local smac-jepa-wm checkout is not available")
    from smac_jepa.jepa import SMACJEPA

    meta = {
        "state_dim": 8,
        "n_agents": 2,
        "n_actions": 3,
        "n_enemies": 1,
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
    }
    core = SMACJEPA(
        state_dim=8, n_agents=2, n_actions=3, latent_dim=6, hidden_dim=8,
        action_dim=4, num_heads=2, mode="entity", max_agents=2,
        max_enemies=1, max_actions=3, token_dim=5, static_dim=4,
    )
    memory = EntityRolloutGRUMemory(latent_dim=6, memory_dim=7)
    info = JEPACheckpointInfo("synthetic", "0" * 64, meta, {}, "synthetic", False, 6, 7, 3)
    return FrozenJEPAWorldModel(core=core, memory_module=memory, info=info, feature_dim=16)


def _obs(batch=2, time=None):
    shape = (batch, 3, 5) if time is None else (batch, time, 3, 5)
    prefix = (batch,) if time is None else (batch, time)
    return {
        "jepa_entity": torch.randn(*shape),
        "jepa_entity_mask": torch.ones(*prefix, 3),
        "jepa_entity_slot_mask": torch.ones(*prefix, 3),
        "jepa_static_condition": torch.randn(*prefix, 4),
    }


def test_initial_observe_and_imagine_shapes():
    wm = _model()
    z0, d0 = wm.initial(2)
    assert z0.shape == (2, 3, 6)
    assert d0.shape == (2, wm.state_spec.deter_dim)
    enc = wm.encode_obs(_obs(batch=2, time=4))
    actions = torch.zeros(2, 4, 6)
    actions[..., 0] = 1
    z, d = wm.observe(enc, actions, (z0, d0), torch.zeros(2, 4, dtype=torch.bool))
    assert z.shape == (2, 4, 3, 6)
    assert d.shape == (2, 4, wm.state_spec.deter_dim)
    zi, di = wm.imagine_with_action(z[:, 0], d[:, 0], actions[:, :2])
    assert zi.shape == (2, 2, 3, 6)
    assert di.shape == (2, 2, wm.state_spec.deter_dim)
    feat = wm.get_feat(z[:, 0], d[:, 0])
    assert feat.shape == (2, 16)


def test_repeated_obs_step_equals_observe():
    wm = _model()
    z0, d0 = wm.initial(1)
    obs = _obs(batch=1, time=3)
    enc = wm.encode_obs(obs)
    actions = torch.zeros(1, 3, 6)
    actions[..., 0] = 1
    resets = torch.zeros(1, 3, dtype=torch.bool)
    z_seq, d_seq = wm.observe(enc, actions, (z0, d0), resets)
    z, d = z0, d0
    zs, ds = [], []
    for t in range(3):
        z, d = wm.obs_step(z, d, actions[:, t], {k: v[:, t] for k, v in enc.items()}, resets[:, t])
        zs.append(z)
        ds.append(d)
    torch.testing.assert_close(z_seq, torch.stack(zs, 1))
    torch.testing.assert_close(d_seq, torch.stack(ds, 1))


def test_frozen_core_has_no_trainable_parameters():
    wm = _model()
    assert sum(p.numel() for p in wm.parameters_frozen() if p.requires_grad) == 0
    wm.train()
    assert not wm.core.training
    assert not wm.memory_module.training
