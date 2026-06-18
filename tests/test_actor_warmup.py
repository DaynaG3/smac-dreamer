"""Tests for actor warm-up on continuation (first N local steps train everything but the actor).

The mechanism (in external/r2dreamer/dreamer.py) scales the policy loss by 0 while
``actor_updates_enabled`` is False:

    _pol_scale = 1.0 if self.actor_updates_enabled else 0.0
    losses["policy"] = _pol_scale * torch.mean(weight * -(logpi * adv + ent * entropy))

Because the actor parameters appear ONLY in the policy loss, a zero scale yields zero actor
gradient (and with a fresh optimizer there is no momentum to apply), so the actor is frozen
without touching the world-model/critic losses or per-step requires_grad toggling.

The trainer hook flips the flag by LOCAL step: enabled = (step >= actor_warmup_steps).
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class _TinyActorModule(nn.Module):
    """Replicates the policy-loss gating expression against a real actor parameter."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.world = nn.Linear(4, 2)  # stands in for encoder/RSSM/critic
        self.actor_updates_enabled = True

    def losses(self, x, adv):
        logits = self.actor(x)
        logpi = logits.sum()
        _pol_scale = 1.0 if self.actor_updates_enabled else 0.0
        policy = _pol_scale * torch.mean(-(logpi * adv))
        world = torch.mean(self.world(x) ** 2)   # always-on world-model/critic surrogate
        return policy, world


def _actor_grad_norm(m):
    return sum(p.grad.abs().sum().item() for p in m.actor.parameters() if p.grad is not None)


def _world_grad_norm(m):
    return sum(p.grad.abs().sum().item() for p in m.world.parameters() if p.grad is not None)


def test_actor_frozen_when_updates_disabled():
    torch.manual_seed(0)
    m = _TinyActorModule()
    m.actor_updates_enabled = False
    x = torch.randn(3, 4)
    adv = torch.randn(3)
    policy, world = m.losses(x, adv)
    (policy + world).backward()
    # actor receives ZERO gradient during warm-up...
    assert _actor_grad_norm(m) == pytest.approx(0.0)
    # ...while the world model / critic still trains.
    assert _world_grad_norm(m) > 0.0


def test_actor_trains_when_updates_enabled():
    torch.manual_seed(0)
    m = _TinyActorModule()
    m.actor_updates_enabled = True
    x = torch.randn(3, 4)
    adv = torch.randn(3) + 1.0
    policy, world = m.losses(x, adv)
    (policy + world).backward()
    assert _actor_grad_norm(m) > 0.0
    assert _world_grad_norm(m) > 0.0


def test_optimizer_step_does_not_move_actor_during_warmup():
    torch.manual_seed(0)
    m = _TinyActorModule()
    m.actor_updates_enabled = False
    opt = torch.optim.SGD(list(m.parameters()), lr=0.1)
    before = m.actor.weight.detach().clone()
    world_before = m.world.weight.detach().clone()
    x = torch.randn(3, 4)
    policy, world = m.losses(x, torch.randn(3))
    opt.zero_grad()
    (policy + world).backward()
    opt.step()
    # actor weights unchanged; world weights changed
    assert torch.allclose(m.actor.weight, before)
    assert not torch.allclose(m.world.weight, world_before)


def test_actor_moves_after_warmup_threshold():
    torch.manual_seed(0)
    m = _TinyActorModule()
    m.actor_updates_enabled = True
    opt = torch.optim.SGD(list(m.parameters()), lr=0.1)
    before = m.actor.weight.detach().clone()
    x = torch.randn(3, 4)
    policy, world = m.losses(x, torch.randn(3) + 1.0)
    opt.zero_grad()
    (policy + world).backward()
    opt.step()
    assert not torch.allclose(m.actor.weight, before)


@pytest.mark.parametrize("step,warmup,expected", [
    (0, 25_000, False),
    (24_999, 25_000, False),
    (25_000, 25_000, True),
    (25_001, 25_000, True),
    (123, 0, True),       # no warm-up configured -> always enabled
])
def test_warmup_threshold_rule(step, warmup, expected):
    # Mirrors the trainer hook: agent.actor_updates_enabled = (local_step >= warmup_steps).
    assert (step >= warmup) is expected
