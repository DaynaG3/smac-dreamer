from __future__ import annotations

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402


def cfg(**overrides):
    base = {
        "enabled": True,
        "num_options": 8,
        "option_embedding_dim": 16,
        "age_embedding_dim": 8,
        "hidden_dim": 32,
        "min_duration": 3,
        "max_duration": 20,
        "initial_termination_probability": 0.10,
        "termination_warmup_steps": 50,
        "termination_full_steps": 150,
        "termination_max_probability_during_ramp": 0.30,
        "termination_max_probability_final": 0.80,
        "termination_cap_full_steps": 250,
        "termination_margin_normalized": 0.02,
        "termination_loss_scale": 0.05,
        "termination_entropy_scale": 0.0,
        "termination_collapse_scale": 0.05,
        "termination_mean_min": 0.02,
        "termination_mean_max": 0.60,
        "termination_advantage_clip": 1.0,
        "termination_min_advantage_magnitude": 0.01,
        "termination_max_target_disagreement": 0.25,
        "termination_unimix": 0.02,
        "eval_sample_termination": False,
        "eval_termination_hazard_threshold": 1.0,
        "manager_unimix_initial": 0.1,
        "manager_unimix_final": 0.02,
        "manager_unimix_decay_steps": 200,
        "manager_pg_scale": 1.0,
        "manager_pg_warmup_steps": 25,
        "manager_pg_full_steps": 100,
        "manager_entropy_scale": 1e-4,
        "manager_collapse_scale": 0.05,
        "manager_mi_target_normalized": 0.10,
        "manager_mi_scale": 0.02,
        "max_usage_target": 0.75,
        "min_effective_options": 3.0,
        "worker_pg_scale": 1.0,
        "worker_entropy_scale": 0.0,
        "worker_scale_warmup_steps": 25,
        "worker_scale_full_steps": 100,
        "worker_scale_max": 0.25,
        "max_abs_residual_logit": 2.0,
        "max_residual_to_base": 0.25,
        "residual_guard_scale": 0.05,
        "base_kl_target": 0.02,
        "base_kl_scale": 0.1,
        "action_diversity_target": 0.002,
        "action_diversity_scale": 0.05,
        "residual_cosine_target": 0.95,
        "residual_cosine_scale": 0.01,
        "max_diversity_states": 64,
        "max_diversity_pairs": 8,
        "option_critic_scale": 1.0,
        "hierarchy_value_scale": 0.5,
        "slow_target_update": 1,
        "slow_target_fraction": 0.005,
        "freeze_base_actor": True,
        "freeze_feature_adapter": True,
    }
    base.update(overrides)
    return base


def policy(**overrides):
    torch.manual_seed(0)
    return HierarchicalOptionsPolicy(12, 15, cfg(**overrides))


def test_centered_residuals_sum_to_zero():
    model = policy()
    model.set_training_step(100)
    feat = torch.randn(7, 12)
    residual = model.all_residual_logits(feat)
    assert residual.shape == (7, 8, 15)
    assert torch.allclose(
        residual.sum(dim=-2), torch.zeros(7, 15), atol=2e-6, rtol=0
    )


def test_worker_scale_schedule():
    model = policy()
    assert model.worker_scale(0) == 0.0
    assert math.isclose(model.worker_scale(25), 0.0625)
    assert math.isclose(model.worker_scale(100), 0.25)
    assert math.isclose(model.worker_scale(1000), 0.25)


def test_termination_blend_schedule():
    model = policy()
    assert model.termination_blend(0) == 0.0
    assert model.termination_blend(50) == 0.0
    assert math.isclose(model.termination_blend(100), 0.5)
    assert model.termination_blend(150) == 1.0


def test_minimum_duration_is_never_violated():
    model = policy()
    model.set_training_step(1_000)
    feat = torch.randn(4, 12)
    option = torch.tensor([0, 1, 2, 3])
    has = torch.ones(4, dtype=torch.bool)
    first = torch.zeros(4, dtype=torch.bool)
    for age_value in (0, 1, 2):
        step = model.step_option(
            feat,
            option,
            torch.full((4,), age_value),
            has,
            first,
            deterministic=False,
            termination_uniform=torch.zeros(4),
        )
        assert not step.option_terminated.any()
        assert torch.equal(step.option, option)


def test_maximum_duration_always_terminates():
    model = policy()
    model.set_training_step(1_000)
    feat = torch.randn(4, 12)
    previous = torch.tensor([0, 1, 2, 3])
    step = model.step_option(
        feat,
        previous,
        torch.full((4,), 20),
        torch.ones(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
        deterministic=False,
        termination_uniform=torch.ones(4),
        manager_uniform=torch.tensor([0.01, 0.2, 0.4, 0.8]),
    )
    assert step.option_terminated.all()
    assert step.option_started.all()
    assert torch.equal(step.action_age, torch.zeros(4, dtype=torch.long))
    assert torch.equal(step.carry_age, torch.ones(4, dtype=torch.long))


def test_same_option_reselection_is_allowed():
    model = policy()
    with torch.no_grad():
        model.manager[-1].weight.zero_()
        model.manager[-1].bias.fill_(-10)
        model.manager[-1].bias[3] = 10
    feat = torch.randn(1, 12)
    step = model.step_option(
        feat,
        torch.tensor([3]),
        torch.tensor([20]),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        deterministic=True,
    )
    assert step.option_terminated.item()
    assert step.option.item() == 3
    assert step.option_started.item()


def test_new_episode_selects_fresh_option_and_resets_age():
    model = policy()
    feat = torch.randn(3, 12)
    step = model.step_option(
        feat,
        torch.tensor([4, 4, 4]),
        torch.tensor([9, 9, 9]),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        deterministic=True,
    )
    assert step.option_started.all()
    assert not step.option_terminated.any()
    assert torch.equal(step.action_age, torch.zeros(3, dtype=torch.long))


def test_eligible_probability_uses_fixed_hazard_during_warmup():
    model = policy()
    feat = torch.randn(5, 12)
    option = torch.zeros(5, dtype=torch.long)
    age = torch.full((5,), 5)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=0
    )
    assert eligible.all()
    assert torch.allclose(beta, torch.full_like(beta, 0.1), atol=1e-6)


def test_action_statistics_are_finite_and_masked():
    model = policy()
    model.set_training_step(100)
    feat = torch.randn(16, 12)
    base = torch.randn(16, 15)
    option = torch.arange(16) % 8
    mask = torch.ones(16, 3, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    active = torch.ones(16, 3, dtype=torch.bool)
    stats = model.behaviour_statistics(
        feat, base, option, mask, active, (5, 5, 5), None, 100
    )
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert stats["base_kl_mean"] >= 0
    assert 0 <= stats["action_flip_rate"] <= 1


def test_bfloat16_path_is_finite():
    model = policy().eval()
    model.set_training_step(100)
    feat = torch.randn(8, 12)
    base = torch.randn(8, 15, dtype=torch.bfloat16)
    option = torch.arange(8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model.combine_logits(base, feat, option)
        beta = model.learned_termination_probability(
            feat, option, torch.arange(8) % 5
        )
    assert torch.isfinite(logits.float()).all()
    assert torch.isfinite(beta.float()).all()


def test_centered_residuals_are_hard_bounded_after_projection():
    model = policy(max_abs_residual_logit=0.5, worker_scale_max=0.25)
    model.set_training_step(1000)
    with torch.no_grad():
        model.worker_residual[-1].weight.mul_(1000)
    feat = torch.randn(9, 12)
    residual = model.all_residual_logits(feat)
    assert residual.abs().max() <= 0.1250001
    assert torch.allclose(residual.sum(dim=-2), torch.zeros(9, 15), atol=2e-6)


def test_eval_uses_deterministic_cumulative_hazard():
    model = policy(eval_sample_termination=False)
    model.set_training_step(0)
    feat = torch.randn(1, 12)
    option = torch.tensor([1])
    age = torch.tensor([3])
    has = torch.ones(1, dtype=torch.bool)
    first = torch.zeros(1, dtype=torch.bool)
    hazard = torch.zeros(1)
    # Fixed beta=0.1 should not terminate immediately and should terminate
    # deterministically after roughly ten eligible decisions.
    seen = None
    for index in range(12):
        step = model.step_option(
            feat, option, age, has, first, deterministic=True,
            termination_hazard=hazard,
        )
        if step.option_terminated.item():
            seen = index + 1
            break
        hazard = step.carry_termination_hazard
        age = step.carry_age
    assert seen == 10


def test_behavior_statistics_honors_zero_state_weights():
    model = policy()
    model.set_training_step(100)
    feat = torch.randn(4, 12)
    base = torch.randn(4, 15)
    option = torch.arange(4)
    mask = torch.ones(4, 3, 5, dtype=torch.bool)
    active = torch.ones(4, 3, dtype=torch.bool)
    weighted = model.behaviour_statistics(
        feat, base, option, mask, active, (5, 5, 5),
        torch.tensor([1.0, 0.0, 0.0, 0.0]), 100
    )
    single = model.behaviour_statistics(
        feat[:1], base[:1], option[:1], mask[:1], active[:1],
        (5, 5, 5), torch.ones(1), 100
    )
    assert torch.allclose(weighted["base_kl_mean"], single["base_kl_mean"], atol=1e-6)


def test_replay_predecision_state_reproduces_the_current_option_decision():
    """Replay must store the state entering act(), not post-action carry age."""
    model = policy()
    model.set_training_step(1000)
    feat = torch.randn(3, 12)
    before_option = torch.tensor([1, 2, 3])
    before_age = torch.tensor([3, 7, 20])
    before_has = torch.ones(3, dtype=torch.bool)
    first = torch.zeros(3, dtype=torch.bool)
    termination_u = torch.tensor([0.0, 1.0, 1.0])
    manager_u = torch.tensor([0.1, 0.5, 0.9])

    real = model.step_option(
        feat,
        before_option,
        before_age,
        before_has,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    replay_reconstructed = model.step_option(
        feat,
        before_option,
        before_age,
        before_has,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    for left, right in zip(real, replay_reconstructed):
        assert torch.equal(left, right)

    # Starting from carry_age would incorrectly make the same posterior state
    # one primitive action older, and can trigger an early boundary.
    wrong = model.step_option(
        feat,
        real.option,
        real.carry_age,
        real.has_option,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    assert not torch.equal(real.previous_age, wrong.previous_age)


def test_manager_mi_guard_detects_state_independent_uniform_routing():
    model = policy()
    probs = torch.full((16, 8), 1.0 / 8.0)
    sampled = torch.arange(16) % 8
    boundary = torch.ones(16, dtype=torch.bool)
    stats = model.manager_statistics(probs, sampled, boundary)
    assert stats["mutual_information_normalized"] < 1.0e-6
    assert stats["mi_shortfall_loss"] > 0


def test_manager_mi_guard_is_inactive_for_state_dependent_routing():
    model = policy()
    probs = torch.nn.functional.one_hot(torch.arange(16) % 8, 8).float()
    sampled = torch.arange(16) % 8
    boundary = torch.ones(16, dtype=torch.bool)
    stats = model.manager_statistics(probs, sampled, boundary)
    assert stats["mutual_information_normalized"] > 0.99
    assert stats["mi_shortfall_loss"] == 0


def test_manager_pg_blend_starts_only_after_worker_warmup():
    model = policy()
    assert model.manager_pg_blend(0) == 0.0
    assert model.manager_pg_blend(25) == 0.0
    assert math.isclose(model.manager_pg_blend(62.5), 0.5)
    assert model.manager_pg_blend(100) == 1.0


def test_termination_embedding_is_gradient_isolated_from_worker():
    model = policy()
    model.set_training_step(1000)
    feat = torch.randn(6, 12)
    option = torch.arange(6) % 8
    worker_loss = model.residual_logits(feat, option).square().sum()
    worker_loss.backward()
    worker_grad = model.option_embedding.weight.grad
    term_grad = model.termination_option_embedding.weight.grad
    assert worker_grad is not None and torch.count_nonzero(worker_grad) > 0
    assert term_grad is None or torch.count_nonzero(term_grad) == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.termination[-1].weight.normal_(0.0, 0.1)
    beta = model.learned_termination_probability(
        feat, option, torch.full((6,), 5)
    )
    beta.sum().backward()
    worker_grad = model.option_embedding.weight.grad
    term_grad = model.termination_option_embedding.weight.grad
    assert term_grad is not None and torch.count_nonzero(term_grad) > 0
    assert worker_grad is None or torch.count_nonzero(worker_grad) == 0


def test_invalid_action_logits_cannot_fake_residual_magnitude_or_diversity():
    import types

    model = policy()
    model.set_training_step(1000)
    n = 4
    feat = torch.randn(n, 12)
    base = torch.zeros(n, 15)
    option = torch.arange(n) % 8
    # Only action 0 is valid for each of three agents.
    mask = torch.zeros(n, 3, 5, dtype=torch.bool)
    mask[..., 0] = True
    active = torch.ones(n, 3, dtype=torch.bool)

    def fake_all(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15)
        # Large option-specific changes exist only in invalid action slots.
        for k in range(8):
            shaped = out[:, k].reshape(local_feat.shape[0], 3, 5)
            shaped[..., 1:] = float(k + 1)
        return out.to(local_feat.device)

    model.all_residual_logits = types.MethodType(fake_all, model)
    stats = model.behaviour_statistics(
        feat, base, option, mask, active, (5, 5, 5), torch.ones(n), 1000
    )
    assert stats["base_kl_mean"] == 0
    assert stats["js_mean"] == 0
    assert stats["residual_rms"] == 0
    assert stats["residual_cosine_mean"] == 0


def test_diversity_hinge_penalizes_duplicate_pairs_even_when_mean_js_is_high():
    import types

    model = policy(
        action_diversity_target=0.001,
        max_diversity_pairs=28,
    )
    model.set_training_step(1000)
    n = 6
    feat = torch.randn(n, 12)
    base = torch.zeros(n, 15)
    selected = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n, 3, 5, dtype=torch.bool)
    active = torch.ones(n, 3, dtype=torch.bool)

    def fake_all(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15, device=local_feat.device)
        # Options 0 and 1 are exact duplicates. The remaining options are made
        # strongly distinct, so a hinge on only mean JS would miss the duplicate.
        for k in range(2, 8):
            shaped = out[:, k].reshape(local_feat.shape[0], 3, 5)
            shaped[..., k % 5] = 4.0
        return out

    model.all_residual_logits = types.MethodType(fake_all, model)
    stats = model.behaviour_statistics(
        feat, base, selected, mask, active, (5, 5, 5), torch.ones(n), 1000
    )
    assert stats["js_mean"] > model.settings.action_diversity_target
    assert stats["js_shortfall_fraction"] > 0
    assert stats["diversity_loss"] > 0


def test_termination_execution_cap_relaxes_continuously_without_full_blend_jump():
    model = policy()
    # Learned-beta blend becomes full at 150, but the execution cap remains at
    # 0.30 there and only reaches 0.80 at cap_full=250.
    assert math.isclose(model.termination_probability_cap(149), 0.30)
    assert math.isclose(model.termination_probability_cap(150), 0.30)
    assert math.isclose(model.termination_probability_cap(200), 0.55)
    assert math.isclose(model.termination_probability_cap(250), 0.80)

    with torch.no_grad():
        model.termination[-1].weight.zero_()
        model.termination[-1].bias.fill_(20.0)
    feat = torch.randn(4, 12)
    option = torch.zeros(4, dtype=torch.long)
    age = torch.full((4,), 5)
    before, *_ = model.effective_termination_probability(feat, option, age, 149)
    at_full, *_ = model.effective_termination_probability(feat, option, age, 150)
    later, *_ = model.effective_termination_probability(feat, option, age, 200)
    assert before.max() <= 0.300001
    assert at_full.max() <= 0.300001
    assert later.max() <= 0.550001


def test_initialized_learned_termination_matches_fixed_hazard_after_unimix():
    model = policy()
    feat = torch.randn(5, 12)
    option = torch.arange(5) % 8
    age = torch.full((5,), 5)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=150
    )
    assert eligible.all()
    assert torch.allclose(beta, torch.full_like(beta, 0.10), atol=1e-6)


def test_executed_termination_probability_has_exact_warmup_and_cap_gradients():
    model = policy()
    feat = torch.zeros(1, 12)
    option = torch.zeros(1, dtype=torch.long)
    age = torch.full((1,), 5, dtype=torch.long)
    with torch.no_grad():
        model.termination[-1].weight.zero_()
        model.termination[-1].bias.fill_(0.0)  # raw sigmoid = 0.5

    model.set_training_step(0)
    beta_warm, eligible, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    assert eligible.all()
    beta_warm.sum().backward()
    warm_grad = model.termination[-1].bias.grad
    assert warm_grad is not None and warm_grad.item() == 0.0

    model.zero_grad(set_to_none=True)
    model.set_training_step(100)  # blend = 0.5, cap = 0.30
    # raw beta=0.2 stays below the cap after 2% unimix.
    raw_beta = 0.2
    model.termination[-1].bias.data.fill_(math.log(raw_beta / (1.0 - raw_beta)))
    beta_mid, _, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    beta_mid.sum().backward()
    expected = 0.5 * (1.0 - model.settings.termination_unimix) * raw_beta * (1.0 - raw_beta)
    assert torch.allclose(
        model.termination[-1].bias.grad, torch.tensor(expected), atol=1e-7
    )

    model.zero_grad(set_to_none=True)
    model.set_training_step(200)  # full blend, execution cap=0.55
    model.termination[-1].bias.data.fill_(20.0)
    beta_capped, _, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    beta_capped.sum().backward()
    capped_grad = model.termination[-1].bias.grad
    assert capped_grad is not None and capped_grad.item() == 0.0
