import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
JEPA = pathlib.Path("/Users/kialok/Desktop/NUS Mods/smac-jepa-wm-main")
if JEPA.exists():
    sys.path.insert(0, str(JEPA))

pytest.importorskip("torch")

from smacdreamer.jepa.checkpoint import JEPACompatibilityError, load_frozen_jepa_checkpoint


def _make_checkpoint(path):
    from smac_jepa.jepa import SMACJEPA
    from smacdreamer.jepa.memory import EntityRolloutGRUMemory

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
    cfg = {
        "latent_dim": 6,
        "hidden_dim": 8,
        "action_dim": 4,
        "num_heads": 2,
        "rollout_memory_dim": 7,
        "training_regime": "synthetic",
    }
    model = SMACJEPA(**{
        "state_dim": meta["state_dim"],
        "n_agents": meta["n_agents"],
        "n_actions": meta["n_actions"],
        "latent_dim": cfg["latent_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "action_dim": cfg["action_dim"],
        "num_heads": cfg["num_heads"],
        "mode": "entity",
        "max_agents": meta["max_agents"],
        "max_enemies": meta["max_enemies"],
        "max_actions": meta["max_actions"],
        "token_dim": meta["token_dim"],
        "static_dim": meta["static_dim"],
    })
    memory = EntityRolloutGRUMemory(latent_dim=6, memory_dim=7, hidden_dim=None, residual=True)
    torch.save({"model_state": model.state_dict(), "memory_module_state": memory.state_dict(),
                "metadata": meta, "resolved_config": cfg}, path)
    return meta


def test_synthetic_checkpoint_loads_and_freezes(tmp_path):
    if not JEPA.exists():
        pytest.skip("local smac-jepa-wm checkout is not available")
    path = tmp_path / "jepa.pt"
    meta = _make_checkpoint(path)
    model, memory, info = load_frozen_jepa_checkpoint(path, map_location="cpu", live_metadata=meta)
    assert info.sha256
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in memory.parameters())


def test_checkpoint_metadata_mismatch_fails(tmp_path):
    if not JEPA.exists():
        pytest.skip("local smac-jepa-wm checkout is not available")
    path = tmp_path / "jepa.pt"
    meta = _make_checkpoint(path)
    live = dict(meta)
    live["max_actions"] = 9
    with pytest.raises(JEPACompatibilityError, match="max_actions"):
        load_frozen_jepa_checkpoint(path, map_location="cpu", live_metadata=live)
