"""Tests for reward-transfer checkpoint loading (src/smacdreamer/checkpoint_transfer.py).

Uses a lightweight ``nn.Module`` agent with the real top-level state-dict prefixes
(encoder/rssm/actor/cont/avail_head/alive_head/prj retained; reward/value/_slow_value reset).
Verifies that transfer:
  * loads the reward-agnostic modules verbatim,
  * leaves the reset modules at their fresh (destination) init,
  * rebuilds frozen mirrors via clone_and_freeze(),
  * restores NO optimizer/training state (none is even read),
  * fails loudly on incompatible shapes or a missing retained layer,
  * reads both ``latest.pt`` and ``best_val_macro_winrate.pt`` payloads,
  * validates that a missing checkpoint path errors clearly.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from smacdreamer.checkpoint_transfer import (  # noqa: E402
    RETAIN_PREFIXES, RESET_PREFIXES,
    read_agent_state_dict, transfer_reward_load, load_weights_only,
)


class FakeAgent(nn.Module):
    """Mirror of the real Dreamer top-level submodule prefixes (subset, same names)."""

    def __init__(self, actor_out=2):
        super().__init__()
        # retained (reward-agnostic)
        self.encoder = nn.Linear(4, 4)
        self.rssm = nn.Linear(4, 4)
        self.actor = nn.Linear(4, actor_out)
        self.cont = nn.Linear(4, 1)
        self.avail_head = nn.Linear(4, 2)
        self.alive_head = nn.Linear(4, 1)
        self.prj = nn.Linear(4, 4)
        # reset (reward/value dependent)
        self.reward = nn.Linear(4, 1)
        self.value = nn.Linear(4, 1)
        self._slow_value = nn.Linear(4, 1)
        self.clone_calls = 0

    def clone_and_freeze(self):
        self.clone_calls += 1


def _save_ckpt(tmp_path, agent, name="latest.pt", extra=None):
    payload = {"agent_state_dict": agent.state_dict()}
    if extra:
        payload.update(extra)
    path = tmp_path / name
    torch.save(payload, str(path))
    return path


def test_transfer_loads_retained_and_keeps_reset_fresh(tmp_path):
    torch.manual_seed(0)
    src = FakeAgent()
    path = _save_ckpt(tmp_path, src)

    torch.manual_seed(1)
    dst = FakeAgent()
    # snapshot the fresh (destination) reset weights and a retained weight pre-load
    fresh_reward = dst.reward.weight.detach().clone()
    fresh_value = dst.value.weight.detach().clone()
    assert not torch.allclose(dst.encoder.weight, src.encoder.weight)

    result = transfer_reward_load(dst, path, verbose=False)

    # retained modules now equal the checkpoint
    assert torch.allclose(dst.encoder.weight, src.encoder.weight)
    assert torch.allclose(dst.actor.weight, src.actor.weight)
    assert torch.allclose(dst.prj.weight, src.prj.weight)
    # reset modules untouched (still the fresh destination init, NOT the checkpoint's)
    assert torch.allclose(dst.reward.weight, fresh_reward)
    assert torch.allclose(dst.value.weight, fresh_value)
    assert not torch.allclose(dst.reward.weight, src.reward.weight)
    # frozen mirrors rebuilt exactly once
    assert dst.clone_calls == 1
    assert result["loaded"] > 0
    assert result["incompatible"] == []
    assert result["missing_retained"] == []


def test_reads_best_val_payload_format(tmp_path):
    src = FakeAgent()
    path = _save_ckpt(tmp_path, src, name="best_val_macro_winrate.pt",
                      extra={"val_macro_win_rate": 0.5, "val_macro_original_return": 1.2,
                             "step": 1000, "obs_mode": "structured"})
    sd = read_agent_state_dict(path)
    assert "encoder.weight" in sd


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_agent_state_dict(tmp_path / "does_not_exist.pt")


def test_missing_agent_state_dict_key_raises(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"not_the_key": 1}, str(path))
    with pytest.raises(KeyError):
        read_agent_state_dict(path)


def test_incompatible_retained_shape_fails_loudly(tmp_path):
    src = FakeAgent(actor_out=2)
    path = _save_ckpt(tmp_path, src)
    dst = FakeAgent(actor_out=5)   # actor head shape differs -> transfer must refuse
    with pytest.raises(RuntimeError, match="incompatible"):
        transfer_reward_load(dst, path, verbose=False)


def test_missing_retained_layer_fails_loudly(tmp_path):
    src = FakeAgent()
    sd = src.state_dict()
    # drop a retained layer from the checkpoint to simulate a partial/old world model
    for k in [k for k in list(sd) if k.startswith("encoder.")]:
        del sd[k]
    path = tmp_path / "partial.pt"
    torch.save({"agent_state_dict": sd}, str(path))
    dst = FakeAgent()
    with pytest.raises(RuntimeError, match="retained key"):
        transfer_reward_load(dst, path, verbose=False)


def test_load_weights_only_restores_everything(tmp_path):
    torch.manual_seed(0)
    src = FakeAgent()
    path = _save_ckpt(tmp_path, src)
    torch.manual_seed(1)
    dst = FakeAgent()
    load_weights_only(dst, path, verbose=False)
    # full load: even the reward head matches the checkpoint now
    assert torch.allclose(dst.reward.weight, src.reward.weight)
    assert torch.allclose(dst.encoder.weight, src.encoder.weight)
    assert dst.clone_calls == 1


def test_prefix_sets_are_disjoint():
    assert set(RETAIN_PREFIXES).isdisjoint(set(RESET_PREFIXES))
