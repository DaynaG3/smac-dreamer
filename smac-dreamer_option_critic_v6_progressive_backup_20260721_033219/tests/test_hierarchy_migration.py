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
        self._source_hierarchical_options = copy.deepcopy(
            self.hierarchical_options
        )
        for parameter in self._source_hierarchical_options.parameters():
            parameter.requires_grad_(False)

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


def test_v1_2_migration_is_exactly_trajectory_preserving_at_step_zero():
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
    assert result["migration_layout"] == "four_exact_copies_per_v1_2_mode"
    assert result["trajectory_preservation"] == (
        "per_state_reselection_then_gradual_commitment"
    )

    target = agent.hierarchical_options
    source_ref = agent._source_hierarchical_options
    assert target.worker_scale(0) == target.settings.worker_scale_max
    assert target.manager_unimix(0) == 0.0
    assert target.commitment_reselect_probability(0) == 1.0

    assert torch.allclose(
        target.manager[0].weight,
        source["tactical_policy.selector.0.weight"],
    )
    assert torch.allclose(
        target.manager[0].bias,
        source["tactical_policy.selector.0.bias"],
    )
    source_out_w = source["tactical_policy.selector.2.weight"]
    source_out_b = source["tactical_policy.selector.2.bias"]
    for row in range(8):
        source_index = row % 2
        assert torch.equal(target.manager[2].weight[row], source_out_w[source_index])
        assert torch.equal(target.manager[2].bias[row], source_out_b[source_index])

    # Four exact copies per source mode preserve the grouped categorical
    # distribution exactly, without the old 0.25-temperature softening.
    feat = torch.randn(13, target.feature_dim)
    hidden = target.manager[1](target.manager[0](feat))
    source_probs = torch.softmax(
        torch.nn.functional.linear(hidden, source_out_w, source_out_b), dim=-1
    )
    target_probs = target.manager_probs(feat, step=0)
    grouped = torch.stack(
        [target_probs[:, 0::2].sum(-1), target_probs[:, 1::2].sum(-1)], dim=-1
    )
    assert torch.allclose(grouped, source_probs, atol=2e-7, rtol=1e-6)

    source_embedding = source["tactical_policy.embedding.weight"]
    for row in range(8):
        assert torch.equal(
            target.option_embedding.weight[row], source_embedding[row % 2]
        )

    # Every copied option produces exactly its source mode's primitive residual.
    feat = torch.randn(9, target.feature_dim)
    target_residual = target.all_residual_logits(feat, step=0)
    leading = feat.shape[:-1]
    feat_all = feat.unsqueeze(-2).expand(*leading, 2, target.feature_dim)
    src_emb = source_embedding.view(1, 2, -1).expand(feat.shape[0], 2, -1)
    source_raw = target.worker_residual(
        torch.cat([feat_all.float(), src_emb], dim=-1)
    )
    cap = target.settings.max_abs_residual_logit
    source_raw = cap * torch.tanh(source_raw / cap)
    source_center = source_raw - source_raw.mean(dim=1, keepdim=True)
    # Expanding each source mode four times changes the eight-option centering
    # by the same two-mode mean, so each copy remains exact.
    for row in range(8):
        assert torch.allclose(
            target_residual[:, row] / target.worker_scale(0),
            source_center[:, row % 2],
            atol=2e-6,
            rtol=1e-6,
        )

    # At preservation step zero every eligible transition reselects, recovering
    # the source selector's per-state decision semantics rather than imposing a
    # 3-20 step commitment immediately.
    beta, eligible, forced_continue, forced_terminate = (
        target.effective_termination_probability(
            feat, torch.zeros(feat.shape[0], dtype=torch.long),
            torch.ones(feat.shape[0], dtype=torch.long), step=0
        )
    )
    assert eligible.all()
    assert not forced_continue.any()
    assert not forced_terminate.any()
    assert torch.equal(beta, torch.ones_like(beta))

    assert torch.count_nonzero(agent.option_critic.trunk[-1].weight) == 0
    assert torch.count_nonzero(agent.option_critic.trunk[-1].bias) == 0
    for live, frozen, source_copy in zip(
        agent.hierarchical_options.parameters(),
        agent._frozen_hierarchical_options.parameters(),
        source_ref.parameters(),
    ):
        assert torch.equal(live, frozen)
        assert torch.equal(live, source_copy)
        assert frozen.requires_grad is False
        assert source_copy.requires_grad is False


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


class TwoOptionDummyAgent(DummyAgent):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        settings = cfg(
            num_options=2,
            min_effective_options=1.0,
            source_manager_group_count=2,
            max_duration=8,
        )
        self.hierarchical_enabled = True
        self.base = nn.Linear(5, 7)
        self.hierarchical_options = HierarchicalOptionsPolicy(12, 15, settings)
        self.option_critic = OptionCritic(12, settings)
        self._slow_option_critic = copy.deepcopy(self.option_critic)
        self._frozen_hierarchical_options = copy.deepcopy(self.hierarchical_options)
        self._source_hierarchical_options = copy.deepcopy(self.hierarchical_options)
        for parameter in self._source_hierarchical_options.parameters():
            parameter.requires_grad_(False)


def test_v1_2_migration_to_two_options_is_exact_and_has_no_duplicate_slots():
    agent = TwoOptionDummyAgent()
    source = source_tactical_state(agent)
    result = load_hierarchical_compatible_state(
        agent,
        source,
        tactical_metadata={
            "architecture": "tactical_mixture_v1_2",
            "num_tactics": 2,
        },
    )
    assert result["target_options"] == 2
    assert result["migration_layout"] == "1_exact_copies_per_v1_2_mode"
    target = agent.hierarchical_options
    feat = torch.randn(32, 12)
    source_logits = torch.nn.functional.linear(
        torch.nn.functional.elu(torch.nn.functional.linear(
            feat,
            source["tactical_policy.selector.0.weight"],
            source["tactical_policy.selector.0.bias"],
        )),
        source["tactical_policy.selector.2.weight"],
        source["tactical_policy.selector.2.bias"],
    )
    assert torch.allclose(
        target.manager_probs(feat, 0),
        source_logits.softmax(-1),
        atol=1e-6,
    )


def test_v1_2_migration_to_two_options_preserves_worker_residuals_exactly():
    agent = TwoOptionDummyAgent()
    source = source_tactical_state(agent)
    load_hierarchical_compatible_state(
        agent,
        source,
        tactical_metadata={
            "architecture": "tactical_mixture_v1_2",
            "num_tactics": 2,
        },
    )
    target = agent.hierarchical_options
    feat = torch.randn(19, target.feature_dim)
    migrated = target.all_residual_logits(feat, step=0)
    source_embedding = source["tactical_policy.embedding.weight"]
    feat_all = feat.unsqueeze(-2).expand(feat.shape[0], 2, target.feature_dim)
    emb = source_embedding.view(1, 2, -1).expand(feat.shape[0], 2, -1)
    source_raw = target.worker_residual(torch.cat([feat_all.float(), emb], dim=-1))
    cap = target.settings.max_abs_residual_logit
    source_raw = cap * torch.tanh(source_raw / cap)
    source_centered = source_raw - source_raw.mean(dim=1, keepdim=True)
    expected = target.worker_scale(0) * source_centered
    assert torch.allclose(migrated, expected, atol=2e-6, rtol=1e-6)


def test_v1_2_migration_to_two_options_reselects_every_state_at_step_zero():
    agent = TwoOptionDummyAgent()
    source = source_tactical_state(agent)
    load_hierarchical_compatible_state(
        agent,
        source,
        tactical_metadata={
            "architecture": "tactical_mixture_v1_2",
            "num_tactics": 2,
        },
    )
    target = agent.hierarchical_options
    feat = torch.randn(32, target.feature_dim)
    option = torch.randint(0, 2, (32,))
    age = torch.ones(32, dtype=torch.long)
    beta, eligible, forced_continue, forced_terminate = (
        target.effective_termination_probability(feat, option, age, step=0)
    )
    assert eligible.all()
    assert not forced_continue.any()
    assert not forced_terminate.any()
    assert torch.equal(beta, torch.ones_like(beta))
