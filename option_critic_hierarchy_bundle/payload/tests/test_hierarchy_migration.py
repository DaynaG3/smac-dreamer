from __future__ import annotations

import copy
import pathlib
import sys
import types

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

# hierarchical_dreamer imports TensorDict, but migration itself does not use it.
fake_td = types.ModuleType("tensordict")
fake_td.TensorDict = dict
sys.modules.setdefault("tensordict", fake_td)

from hierarchical_dreamer import load_hierarchical_compatible_state  # noqa: E402
from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402
from option_critic import OptionCritic  # noqa: E402
from test_hierarchical_options import cfg  # noqa: E402


class DummyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        settings = cfg(hidden_dim=32)
        self.hierarchical_enabled = True
        self.base = nn.Linear(5, 7)
        self.hierarchical_options = HierarchicalOptionsPolicy(12, 15, settings)
        self.option_critic = OptionCritic(12, settings)
        self._slow_option_critic = copy.deepcopy(self.option_critic)
        self._frozen_hierarchical_options = copy.deepcopy(
            self.hierarchical_options
        )

    def hierarchical_metadata(self):
        metadata = self.hierarchical_options.metadata()
        metadata["enabled"] = True
        return metadata

    def clone_and_freeze(self):
        self._frozen_hierarchical_options = copy.deepcopy(
            self.hierarchical_options
        )
        for parameter in self._frozen_hierarchical_options.parameters():
            parameter.requires_grad_(False)


def source_tactical_state(agent: DummyAgent) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    target = agent.hierarchical_options
    state = {
        "base.weight": torch.randn_like(agent.base.weight),
        "base.bias": torch.randn_like(agent.base.bias),
        "tactical_policy.selector.0.weight": torch.randn_like(
            target.manager[0].weight
        ),
        "tactical_policy.selector.0.bias": torch.randn_like(
            target.manager[0].bias
        ),
        "tactical_policy.selector.2.weight": torch.randn(
            2, target.manager[2].weight.shape[1]
        ),
        "tactical_policy.selector.2.bias": torch.randn(2),
        "tactical_policy.embedding.weight": torch.randn(
            2, target.option_embedding.weight.shape[1]
        ),
        "tactical_policy.residual.0.weight": torch.randn_like(
            target.worker_residual[0].weight
        ),
        "tactical_policy.residual.0.bias": torch.randn_like(
            target.worker_residual[0].bias
        ),
        "tactical_policy.residual.2.weight": torch.randn_like(
            target.worker_residual[2].weight
        ),
        "tactical_policy.residual.2.bias": torch.randn_like(
            target.worker_residual[2].bias
        ),
    }
    # Source checkpoints also contain frozen tactical copies; migration must
    # remove them rather than report unexpected keys.
    for key, value in list(state.items()):
        if key.startswith("tactical_policy."):
            state["_frozen_" + key] = value.clone()
    return state


def test_v1_2_migration_is_controlled_and_explorable():
    agent = DummyAgent()
    source = source_tactical_state(agent)
    result = load_hierarchical_compatible_state(
        agent,
        source,
        tactical_metadata={
            "architecture": "tactical_mixture_v1_2",
            "num_tactics": 2,
        },
    )
    assert result["migrated"] is True
    assert result["target_options"] == 8
    assert result["migration_layout"] == "four_zero_mean_perturbed_copies_per_v1_2_mode"

    target = agent.hierarchical_options
    assert torch.allclose(
        target.manager[0].weight,
        source["tactical_policy.selector.0.weight"],
    )
    source_out_w = source["tactical_policy.selector.2.weight"]
    source_out_b = source["tactical_policy.selector.2.bias"]
    # Four softened manager rows inherit each source mode. Tiny bias offsets
    # sum to zero within each group and only break routing ties.
    for row in range(8):
        source_index = row % 2
        assert torch.allclose(
            target.manager[2].weight[row], 0.25 * source_out_w[source_index]
        )
    for source_index in (0, 1):
        rows = torch.arange(source_index, 8, 2)
        group_bias = target.manager[2].bias[rows]
        assert torch.allclose(
            group_bias.mean(), 0.25 * source_out_b[source_index], atol=1e-7
        )
        assert group_bias.unique().numel() == 4

    # Before manager unimix, summing the four copy probabilities in each
    # group exactly recovers the deliberately softened two-mode source routing.
    feat = torch.randn(13, target.feature_dim)
    hidden = target.manager[1](target.manager[0](feat))
    source_soft = torch.softmax(
        0.25 * torch.nn.functional.linear(hidden, source_out_w, source_out_b),
        dim=-1,
    )
    target_probs = torch.softmax(target.manager[2](hidden), dim=-1)
    grouped = torch.stack(
        [target_probs[:, 0::2].sum(-1), target_probs[:, 1::2].sum(-1)],
        dim=-1,
    )
    assert torch.allclose(grouped, source_soft, atol=2e-7, rtol=1e-6)

    source_embedding = source["tactical_policy.embedding.weight"]
    for source_index in (0, 1):
        rows = torch.arange(source_index, 8, 2)
        group = target.option_embedding.weight[rows]
        # The group mean preserves the source embedding exactly, while every
        # independent row starts distinct so duplicate workers are not a
        # stationary symmetry of the expected gradient.
        assert torch.allclose(group.mean(0), source_embedding[source_index], atol=1e-7)
        assert torch.pdist(group).min() > 0

    # Tiny zero-mean embedding codes preserve the source worker numerically
    # rather than replacing six options with unsafe random policies.
    target.set_training_step(10_000)
    feat = torch.randn(9, 12)
    target_raw = target._all_uncentered_residuals(feat)
    target_center = target.all_residual_logits(feat) / target.worker_scale()

    # Reconstruct the exact two-mode source worker with unperturbed embeddings.
    leading = feat.shape[:-1]
    feat_all = feat.unsqueeze(-2).expand(*leading, 2, target.feature_dim)
    src_emb = source_embedding.view(1, 2, -1).expand(feat.shape[0], 2, -1)
    source_raw = target.worker_residual(torch.cat([feat_all.float(), src_emb], dim=-1))
    cap = target.settings.max_abs_residual_logit
    source_raw = cap * torch.tanh(source_raw / cap)
    source_center = source_raw - source_raw.mean(dim=1, keepdim=True)

    for source_index in (0, 1):
        rows = torch.arange(source_index, 8, 2)
        group_mean = target_center[:, rows].mean(dim=1)
        assert torch.allclose(group_mean, source_center[:, source_index], atol=2e-4, rtol=1e-4)
    assert torch.pdist(target_raw[0]).min() > 0

    assert torch.count_nonzero(agent.option_critic.trunk[-1].weight) == 0
    assert torch.count_nonzero(agent.option_critic.trunk[-1].bias) == 0
    for live, frozen in zip(
        agent.hierarchical_options.parameters(),
        agent._frozen_hierarchical_options.parameters(),
    ):
        assert torch.allclose(live, frozen)
        assert frozen.requires_grad is False


def test_strict_hierarchical_resume_requires_matching_metadata():
    agent = DummyAgent()
    state = agent.state_dict()
    result = load_hierarchical_compatible_state(
        agent,
        state,
        checkpoint_metadata=agent.hierarchical_metadata(),
    )
    assert result == {"migrated": False, "strict": True}

    bad = dict(agent.hierarchical_metadata())
    bad["max_duration"] = 19
    try:
        load_hierarchical_compatible_state(
            agent,
            state,
            checkpoint_metadata=bad,
        )
    except RuntimeError as exc:
        assert "max_duration" in str(exc)
    else:
        raise AssertionError("metadata mismatch should fail closed")
