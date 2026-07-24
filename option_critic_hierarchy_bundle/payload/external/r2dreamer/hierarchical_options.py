"""Call-and-return hierarchical option policy for JEPA-backed R2-Dreamer.

The module is deliberately independent from JEPA and replay. It owns only:

* a high-level manager over discrete options;
* option-conditioned, zero-mean residuals around an inherited primitive actor;
* a learned termination function with hard duration guards;
* the real/imagination option state machine;
* option-collapse and behaviour-diversity diagnostics.

The option index is never an environment action. Primitive action masking remains
outside this module and must be applied after option-conditioned logits are built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class HierarchicalOptionSettings:
    enabled: bool = False
    num_options: int = 8
    option_embedding_dim: int = 16
    age_embedding_dim: int = 8
    hidden_dim: int = 128

    min_duration: int = 3
    max_duration: int = 20
    initial_termination_probability: float = 0.10
    termination_warmup_steps: int = 100_000
    termination_full_steps: int = 300_000
    termination_max_probability_during_ramp: float = 0.30
    termination_max_probability_final: float = 0.80
    termination_cap_full_steps: int = 500_000
    termination_margin_normalized: float = 0.02
    termination_loss_scale: float = 0.05
    termination_entropy_scale: float = 0.0
    termination_collapse_scale: float = 0.05
    termination_mean_min: float = 0.02
    termination_mean_max: float = 0.60
    termination_advantage_clip: float = 1.0
    termination_min_advantage_magnitude: float = 0.01
    termination_max_target_disagreement: float = 0.25
    termination_unimix: float = 0.02
    eval_sample_termination: bool = False
    eval_termination_hazard_threshold: float = 1.0

    manager_unimix_initial: float = 0.10
    manager_unimix_final: float = 0.02
    manager_unimix_decay_steps: int = 200_000
    manager_pg_scale: float = 1.0
    manager_pg_warmup_steps: int = 25_000
    manager_pg_full_steps: int = 100_000
    manager_entropy_scale: float = 1.0e-4
    manager_collapse_scale: float = 0.05
    manager_mi_target_normalized: float = 0.10
    manager_mi_scale: float = 0.02
    max_usage_target: float = 0.75
    min_effective_options: float = 3.0

    worker_pg_scale: float = 1.0
    worker_entropy_scale: float = 0.0
    worker_scale_warmup_steps: int = 25_000
    worker_scale_full_steps: int = 100_000
    worker_scale_max: float = 0.25
    max_abs_residual_logit: float = 2.0
    max_residual_to_base: float = 0.25
    residual_guard_scale: float = 0.05
    base_kl_target: float = 0.02
    base_kl_scale: float = 0.10

    action_diversity_target: float = 0.002
    action_diversity_scale: float = 0.05
    residual_cosine_target: float = 0.95
    residual_cosine_scale: float = 0.01
    max_diversity_states: int = 128
    max_diversity_pairs: int = 12

    option_critic_scale: float = 1.0
    hierarchy_value_scale: float = 0.5
    slow_target_update: int = 1
    slow_target_fraction: float = 0.005

    freeze_base_actor: bool = True
    freeze_feature_adapter: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "HierarchicalOptionSettings":
        return cls(**{
            field: _cfg_get(cfg, field, getattr(cls(), field))
            for field in cls.__dataclass_fields__
        })

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.num_options < 2:
            raise ValueError("hierarchical_options.num_options must be >= 2")
        if self.option_embedding_dim <= 0 or self.age_embedding_dim <= 0:
            raise ValueError("option and age embedding dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hierarchical_options.hidden_dim must be positive")
        if not 1 <= self.min_duration < self.max_duration:
            raise ValueError("require 1 <= min_duration < max_duration")
        if self.max_duration > 255:
            raise ValueError("max_duration > 255 is intentionally unsupported")
        if not 0.0 < self.initial_termination_probability < 1.0:
            raise ValueError("initial_termination_probability must be in (0, 1)")
        if not 0 <= self.termination_warmup_steps < self.termination_full_steps:
            raise ValueError("termination warmup/full steps are inconsistent")
        if not 0.0 < self.termination_max_probability_during_ramp <= 1.0:
            raise ValueError("termination ramp probability cap must be in (0, 1]")
        if not (
            self.termination_max_probability_during_ramp
            <= self.termination_max_probability_final
            <= 1.0
        ):
            raise ValueError("termination probability caps are inconsistent")
        if self.termination_cap_full_steps <= self.termination_full_steps:
            raise ValueError(
                "termination_cap_full_steps must exceed termination_full_steps"
            )
        if not 0.0 <= self.manager_unimix_final <= self.manager_unimix_initial < 1.0:
            raise ValueError("manager unimix schedule is invalid")
        if self.manager_unimix_decay_steps <= 0:
            raise ValueError("manager_unimix_decay_steps must be positive")
        if not 0 <= self.manager_pg_warmup_steps < self.manager_pg_full_steps:
            raise ValueError("manager PG warmup/full steps are inconsistent")
        if not 1.0 / self.num_options <= self.max_usage_target <= 1.0:
            raise ValueError("max_usage_target must be between uniform usage and 1")
        if not 1.0 <= self.min_effective_options <= self.num_options:
            raise ValueError("min_effective_options must be in [1, num_options]")
        if not 0.0 < self.worker_scale_max <= 1.0:
            raise ValueError("worker_scale_max must be in (0, 1]")
        if not 0 < self.max_abs_residual_logit:
            raise ValueError("max_abs_residual_logit must be positive")
        if not 0 < self.max_residual_to_base:
            raise ValueError("max_residual_to_base must be positive")
        if not 0.0 <= self.termination_unimix < 1.0:
            raise ValueError("termination_unimix must be in [0, 1)")
        termination_floor = 0.5 * self.termination_unimix
        termination_ceiling = 1.0 - termination_floor
        if not termination_floor < self.initial_termination_probability < termination_ceiling:
            raise ValueError(
                "initial_termination_probability must lie inside the termination "
                "unimix support"
            )
        if not math.isfinite(self.eval_termination_hazard_threshold) or self.eval_termination_hazard_threshold <= 0.0:
            raise ValueError("eval_termination_hazard_threshold must be finite and positive")
        if not 0.0 <= self.termination_mean_min < self.termination_mean_max <= 1.0:
            raise ValueError("termination mean bounds are invalid")
        if not self.termination_advantage_clip > 0:
            raise ValueError("termination_advantage_clip must be positive")
        if not 0.0 <= self.termination_min_advantage_magnitude <= self.termination_advantage_clip:
            raise ValueError("termination_min_advantage_magnitude is invalid")
        if not self.termination_max_target_disagreement > 0:
            raise ValueError("termination_max_target_disagreement must be positive")
        if not 0.0 <= self.manager_mi_target_normalized <= 1.0:
            raise ValueError("manager_mi_target_normalized must be in [0, 1]")
        if not 0.0 <= self.residual_cosine_target <= 1.0:
            raise ValueError("residual_cosine_target must be in [0, 1]")
        if not 0 < self.base_kl_target:
            raise ValueError("base_kl_target must be positive")
        if self.max_diversity_states <= 0 or self.max_diversity_pairs <= 0:
            raise ValueError("diversity subsampling limits must be positive")
        if not 0 < self.slow_target_fraction <= 1.0:
            raise ValueError("slow_target_fraction must be in (0, 1]")
        for name in (
            "termination_margin_normalized",
            "termination_loss_scale",
            "termination_entropy_scale",
            "termination_collapse_scale",
            "manager_pg_scale",
            "manager_entropy_scale",
            "manager_collapse_scale",
            "manager_mi_target_normalized",
            "manager_mi_scale",
            "worker_pg_scale",
            "worker_scale_max",
            "worker_entropy_scale",
            "residual_guard_scale",
            "base_kl_scale",
            "action_diversity_target",
            "action_diversity_scale",
            "residual_cosine_target",
            "residual_cosine_scale",
            "option_critic_scale",
            "hierarchy_value_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"hierarchical_options.{name} must be finite and non-negative")


class OptionStep(NamedTuple):
    option: torch.Tensor
    action_age: torch.Tensor
    carry_age: torch.Tensor
    has_option: torch.Tensor
    option_started: torch.Tensor
    option_terminated: torch.Tensor
    termination_eligible: torch.Tensor
    termination_probability: torch.Tensor
    previous_option: torch.Tensor
    previous_age: torch.Tensor
    manager_log_prob: torch.Tensor
    manager_entropy: torch.Tensor
    carry_termination_hazard: torch.Tensor


class HierarchicalOptionsPolicy(nn.Module):
    """Manager, option worker residuals, and learned termination head."""

    ARCHITECTURE = "dreamer_option_critic_v2"
    SCHEMA_VERSION = 2

    def __init__(self, feature_dim: int, action_logit_dim: int, config: Any) -> None:
        super().__init__()
        self.settings = HierarchicalOptionSettings.from_config(config)
        self.settings.validate()
        self.feature_dim = int(feature_dim)
        self.action_logit_dim = int(action_logit_dim)
        if self.feature_dim <= 0 or self.action_logit_dim <= 0:
            raise ValueError("feature_dim and action_logit_dim must be positive")

        s = self.settings
        self.manager = nn.Sequential(
            nn.Linear(self.feature_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, s.num_options),
        )
        self.option_embedding = nn.Embedding(s.num_options, s.option_embedding_dim)
        # Termination has its own option embedding so termination gradients cannot
        # silently rewrite the worker policy through a shared representation.
        self.termination_option_embedding = nn.Embedding(
            s.num_options, s.option_embedding_dim
        )
        self.age_embedding = nn.Embedding(s.max_duration + 1, s.age_embedding_dim)
        self.worker_residual = nn.Sequential(
            nn.Linear(self.feature_dim + s.option_embedding_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, self.action_logit_dim),
        )
        self.termination = nn.Sequential(
            nn.Linear(
                self.feature_dim + s.option_embedding_dim + s.age_embedding_dim,
                s.hidden_dim,
            ),
            nn.ELU(),
            nn.Linear(s.hidden_dim, 1),
        )
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("diversity_calls", torch.zeros((), dtype=torch.long))
        # Schedules and pair rotation are host-side control decisions. Mirror
        # their checkpointed buffers with Python integers to avoid repeated
        # accelerator synchronizations from Tensor.item() in the hot path.
        self._training_step_int = 0
        self._diversity_calls_int = 0
        self.reset_parameters()

    @property
    def num_options(self) -> int:
        return self.settings.num_options

    def reset_parameters(self) -> None:
        # The inherited primitive actor remains the initial policy. Option
        # residuals start tiny but non-identical so diversity gradients do not
        # sit at an exact symmetric fixed point.
        nn.init.normal_(
            self.manager[-1].weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(float(self.settings.hidden_dim)),
        )
        nn.init.zeros_(self.manager[-1].bias)
        nn.init.normal_(
            self.worker_residual[-1].weight,
            mean=0.0,
            std=1.0e-2 / math.sqrt(float(self.settings.hidden_dim)),
        )
        nn.init.zeros_(self.worker_residual[-1].bias)
        nn.init.zeros_(self.termination[-1].weight)
        # Account for termination unimix so the learned branch itself evaluates
        # to the exact fixed warm-up hazard at initialization.
        term_eps = float(self.settings.termination_unimix)
        initial_raw = (
            self.settings.initial_termination_probability - 0.5 * term_eps
        ) / max(1.0 - term_eps, 1.0e-8)
        nn.init.constant_(self.termination[-1].bias, _logit(initial_raw))

    def set_training_step(self, step: int | torch.Tensor) -> None:
        value = int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step)
        value = max(value, 0)
        self._training_step_int = value
        self.training_step.fill_(value)

    def set_diversity_calls(self, calls: int) -> None:
        value = max(int(calls), 0)
        self._diversity_calls_int = value
        self.diversity_calls.fill_(value)

    def _step_float(self, step: int | torch.Tensor | None = None) -> float:
        if step is None or step is self.training_step:
            return float(self._training_step_int)
        if torch.is_tensor(step):
            # Internal callers pass the registered training_step buffer. External
            # tensor steps are rare control-plane inputs and may synchronize.
            return float(step.detach().cpu().item())
        return float(step)

    def manager_unimix(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        fraction = min(max(self._step_float(step) / s.manager_unimix_decay_steps, 0.0), 1.0)
        return s.manager_unimix_initial + fraction * (
            s.manager_unimix_final - s.manager_unimix_initial
        )

    def worker_scale(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= 0:
            return 0.0
        if x < s.worker_scale_warmup_steps:
            return 0.25 * s.worker_scale_max * x / max(float(s.worker_scale_warmup_steps), 1.0)
        if x < s.worker_scale_full_steps:
            fraction = (x - s.worker_scale_warmup_steps) / max(
                float(s.worker_scale_full_steps - s.worker_scale_warmup_steps), 1.0
            )
            return s.worker_scale_max * (0.25 + 0.75 * fraction)
        return s.worker_scale_max

    def manager_pg_blend(self, step: int | torch.Tensor | None = None) -> float:
        """Keep manager task-PG off until option workers have causal effect."""
        s = self.settings
        x = self._step_float(step)
        if x <= s.manager_pg_warmup_steps:
            return 0.0
        if x >= s.manager_pg_full_steps:
            return 1.0
        return (x - s.manager_pg_warmup_steps) / max(
            float(s.manager_pg_full_steps - s.manager_pg_warmup_steps), 1.0
        )

    def termination_blend(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.termination_warmup_steps:
            return 0.0
        if x >= s.termination_full_steps:
            return 1.0
        return (x - s.termination_warmup_steps) / max(
            float(s.termination_full_steps - s.termination_warmup_steps), 1.0
        )

    def termination_probability_cap(
        self, step: int | torch.Tensor | None = None
    ) -> float:
        """Smoothly relax the execution cap after learned beta is active."""
        s = self.settings
        x = self._step_float(step)
        if x <= s.termination_full_steps:
            return s.termination_max_probability_during_ramp
        if x >= s.termination_cap_full_steps:
            return s.termination_max_probability_final
        fraction = (x - s.termination_full_steps) / max(
            float(s.termination_cap_full_steps - s.termination_full_steps), 1.0
        )
        return s.termination_max_probability_during_ramp + fraction * (
            s.termination_max_probability_final
            - s.termination_max_probability_during_ramp
        )

    def manager_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError("manager feature dimension mismatch")
        return self.manager(feat.float())

    def manager_probs(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = self.manager_logits(feat).softmax(dim=-1)
        unimix = self.manager_unimix(step)
        return (1.0 - unimix) * probs + unimix / self.num_options

    def manager_dist(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> Categorical:
        return Categorical(probs=self.manager_probs(feat, step))

    def _all_uncentered_residuals(self, feat: torch.Tensor) -> torch.Tensor:
        leading = feat.shape[:-1]
        k = self.num_options
        feat_all = feat.unsqueeze(-2).expand(*leading, k, self.feature_dim)
        ids = torch.arange(k, device=feat.device, dtype=torch.long)
        ids = ids.view(*([1] * len(leading)), k).expand(*leading, k)
        emb = self.option_embedding(ids)
        raw = self.worker_residual(torch.cat([feat_all.float(), emb], dim=-1))
        cap = self.settings.max_abs_residual_logit
        return cap * torch.tanh(raw / cap)

    def all_residual_logits(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self._all_uncentered_residuals(feat)
        centered = raw - raw.mean(dim=-2, keepdim=True)
        # Project every state/action slice onto an L-infinity ball without
        # disturbing the exact zero-mean option invariant. A second tanh after
        # centering would break zero mean; multiplicative projection does not.
        cap = float(self.settings.max_abs_residual_logit)
        max_abs = centered.abs().amax(dim=-2, keepdim=True).clamp_min(1.0e-8)
        projection = torch.clamp(cap / max_abs, max=1.0)
        bounded = centered * projection
        return self.worker_scale(step) * bounded

    def residual_logits(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        option = option.to(device=feat.device, dtype=torch.long)
        if option.shape != feat.shape[:-1]:
            raise ValueError("option shape must match feature leading shape")
        all_residual = self.all_residual_logits(feat, step)
        index = option.unsqueeze(-1).unsqueeze(-1).expand(
            *option.shape, 1, self.action_logit_dim
        )
        return all_residual.gather(-2, index).squeeze(-2)

    def combine_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return base_logits + self.residual_logits(feat, option, step).to(base_logits.dtype)

    def learned_termination_probability(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        option = option.to(device=feat.device, dtype=torch.long)
        age = age.to(device=feat.device, dtype=torch.long).clamp(
            0, self.settings.max_duration
        )
        oemb = self.termination_option_embedding(option)
        aemb = self.age_embedding(age)
        logits = self.termination(torch.cat([feat.float(), oemb, aemb], dim=-1))
        return logits.squeeze(-1).sigmoid()

    def effective_termination_probability(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.settings
        age = age.to(device=feat.device, dtype=torch.long)
        learned = self.learned_termination_probability(feat, option, age)
        eps = float(self.settings.termination_unimix)
        learned = (1.0 - eps) * learned + 0.5 * eps
        blend = self.termination_blend(step)
        fixed = torch.full_like(learned, s.initial_termination_probability)
        learned_for_execution = learned.clamp(
            max=self.termination_probability_cap(step)
        )
        mixed = fixed + blend * (learned_for_execution - fixed)
        forced_continue = age < s.min_duration
        forced_terminate = age >= s.max_duration
        eligible = (~forced_continue) & (~forced_terminate)
        effective = torch.where(
            forced_continue,
            torch.zeros_like(mixed),
            torch.where(forced_terminate, torch.ones_like(mixed), mixed),
        )
        return effective, eligible, forced_continue, forced_terminate

    def step_option(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        has_option: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool,
        step: int | torch.Tensor | None = None,
        termination_uniform: torch.Tensor | None = None,
        manager_uniform: torch.Tensor | None = None,
        termination_hazard: torch.Tensor | None = None,
    ) -> OptionStep:
        """Apply the call-and-return state transition before choosing an action.

        ``age`` is the number of primitive actions already executed under the
        carried option. The returned ``carry_age`` is the age after the current
        primitive action and is what must be carried to the next environment
        state and stored in replay.
        """
        leading = feat.shape[:-1]
        option = option.to(device=feat.device, dtype=torch.long).reshape(leading)
        age = age.to(device=feat.device, dtype=torch.long).reshape(leading)
        has_option = has_option.to(device=feat.device, dtype=torch.bool).reshape(leading)
        is_first = is_first.to(device=feat.device, dtype=torch.bool).reshape(leading)
        if termination_hazard is None:
            termination_hazard = torch.zeros(leading, device=feat.device, dtype=torch.float32)
        else:
            termination_hazard = termination_hazard.to(
                device=feat.device, dtype=torch.float32
            ).reshape(leading).clamp_min(0.0)

        reset = is_first | (~has_option)
        safe_option = option.clamp(0, self.num_options - 1)
        safe_age = age.clamp(0, self.settings.max_duration)
        beta, eligible, _, forced_terminate = self.effective_termination_probability(
            feat, safe_option, safe_age, step
        )
        beta = torch.where(reset, torch.zeros_like(beta), beta)
        eligible = eligible & (~reset)

        if deterministic and not self.settings.eval_sample_termination:
            # Deterministic cumulative-hazard evaluation. For constant beta,
            # termination occurs after approximately 1 / beta eligible decisions,
            # matching the timescale of the stochastic training policy without
            # adding validation RNG noise or using the incorrect beta >= 0.5 rule.
            hazard_increment = -torch.log1p(-beta.clamp(max=1.0 - 1.0e-6))
            candidate_hazard = torch.where(
                eligible, termination_hazard + hazard_increment, termination_hazard
            )
            sampled_termination = (
                candidate_hazard >= self.settings.eval_termination_hazard_threshold
            )
        else:
            candidate_hazard = torch.zeros_like(beta)
            if termination_uniform is None:
                termination_uniform = torch.rand_like(beta)
            sampled_termination = termination_uniform < beta
        terminated = (~reset) & (forced_terminate | sampled_termination)
        boundary = reset | terminated
        carry_hazard = torch.where(
            boundary, torch.zeros_like(candidate_hazard), candidate_hazard
        )

        dist = self.manager_dist(feat, step)
        if deterministic:
            proposed = dist.probs.argmax(dim=-1)
        elif manager_uniform is None:
            proposed = dist.sample()
        else:
            cdf = dist.probs.cumsum(dim=-1)
            proposed = (manager_uniform.unsqueeze(-1) > cdf).sum(dim=-1)
            proposed = proposed.clamp_max(self.num_options - 1)

        selected = torch.where(boundary, proposed, safe_option)
        action_age = torch.where(boundary, torch.zeros_like(safe_age), safe_age)
        carry_age = action_age + 1
        selected_log_prob = dist.log_prob(selected)
        manager_log_prob = torch.where(
            boundary, selected_log_prob, torch.zeros_like(selected_log_prob)
        )
        manager_entropy = torch.where(
            boundary, dist.entropy(), torch.zeros_like(selected_log_prob)
        )

        return OptionStep(
            option=selected,
            action_age=action_age,
            carry_age=carry_age,
            has_option=torch.ones_like(has_option),
            option_started=boundary,
            option_terminated=terminated,
            termination_eligible=eligible,
            termination_probability=beta,
            previous_option=safe_option,
            previous_age=safe_age,
            manager_log_prob=manager_log_prob,
            manager_entropy=manager_entropy,
            carry_termination_hazard=carry_hazard,
        )

    @staticmethod
    def _masked_probs(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        active_mask: torch.Tensor,
        actor_shape: Sequence[int],
        unimix_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = len(tuple(actor_shape))
        c = int(tuple(actor_shape)[0])
        shaped = logits.float().reshape(*logits.shape[:-1], a, c)
        mask = action_mask.to(dtype=torch.bool, device=logits.device).reshape(shaped.shape)
        active = active_mask.to(dtype=torch.bool, device=logits.device).reshape(
            *shaped.shape[:-1], 1
        )
        # NOOP must be legal whenever the predicted mask is empty.
        empty = ~mask.any(dim=-1, keepdim=True)
        noop = torch.zeros_like(mask)
        noop[..., 0] = True
        mask = torch.where(empty, noop, mask)
        masked_logits = shaped.masked_fill(~mask, -1.0e9)
        probs = masked_logits.softmax(dim=-1)
        unimix = float(unimix_ratio)
        if unimix:
            uniform = mask.float() / mask.float().sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            probs = (1.0 - unimix) * probs + unimix * uniform
        return probs, active

    def behaviour_statistics(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
        selected_option: torch.Tensor,
        action_mask: torch.Tensor,
        active_mask: torch.Tensor,
        actor_shape: Sequence[int],
        state_weights: torch.Tensor | None = None,
        step: int | torch.Tensor | None = None,
        *,
        unimix_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # Subsample deterministically across the complete flattened sequence.
        # This auxiliary path must not allocate all K pairs for the full B*T
        # batch, but it must also not bias diversity checks toward a prefix.
        flat_feat = feat.reshape(-1, feat.shape[-1])
        flat_base = base_logits.reshape(-1, base_logits.shape[-1])
        flat_option = selected_option.reshape(-1)
        flat_mask = action_mask.reshape(-1, action_mask.shape[-2], action_mask.shape[-1])
        flat_active = active_mask.reshape(-1, active_mask.shape[-1])
        if state_weights is None:
            flat_state_weight = torch.ones(
                flat_feat.shape[0], device=feat.device, dtype=torch.float32
            )
        else:
            flat_state_weight = torch.broadcast_to(
                state_weights.float(), feat.shape[:-1]
            ).reshape(-1)
        total_states = flat_feat.shape[0]
        n = min(total_states, self.settings.max_diversity_states)
        if n < total_states:
            sample_index = torch.linspace(
                0, total_states - 1, n, device=feat.device
            ).round().long()
            flat_feat = flat_feat.index_select(0, sample_index)
            flat_base = flat_base.index_select(0, sample_index)
            flat_option = flat_option.index_select(0, sample_index)
            flat_mask = flat_mask.index_select(0, sample_index)
            flat_active = flat_active.index_select(0, sample_index)
            flat_state_weight = flat_state_weight.index_select(0, sample_index)
        flat_state_weight = flat_state_weight.clamp_min(0.0)

        all_logits = flat_base.unsqueeze(-2) + self.all_residual_logits(flat_feat, step)
        selected_logits = all_logits.gather(
            -2,
            flat_option[:, None, None].expand(n, 1, self.action_logit_dim),
        ).squeeze(-2)
        base_probs, active = self._masked_probs(
            flat_base, flat_mask, flat_active, actor_shape, unimix_ratio
        )
        selected_probs, _ = self._masked_probs(
            selected_logits, flat_mask, flat_active, actor_shape, unimix_ratio
        )
        eps = 1.0e-8
        base_kl_per_agent = (
            selected_probs.clamp_min(eps)
            * (
                selected_probs.clamp_min(eps).log()
                - base_probs.clamp_min(eps).log()
            )
        ).sum(-1)
        active_float = active.squeeze(-1).float()
        active_weight = active_float * flat_state_weight.unsqueeze(-1)
        denominator = active_weight.sum().clamp_min(1.0)
        base_kl_mean = (base_kl_per_agent * active_weight).sum() / denominator
        base_kl_max = torch.where(
            active_weight > 0.0,
            base_kl_per_agent,
            torch.zeros_like(base_kl_per_agent),
        ).max()
        base_mode = base_probs.argmax(dim=-1)
        selected_mode = selected_probs.argmax(dim=-1)
        action_flip_rate = (
            (base_mode != selected_mode).float() * active_weight
        ).sum() / denominator

        all_probs = []
        for option_index in range(self.num_options):
            option_probs, _ = self._masked_probs(
                all_logits[:, option_index], flat_mask, flat_active, actor_shape,
                unimix_ratio,
            )
            all_probs.append(option_probs)
        pair_js = []
        pairs = []
        for i in range(self.num_options):
            for j in range(i + 1, self.num_options):
                pairs.append((i, j))
        # Rotate the sampled subset by training step so all pairs receive
        # diagnostics over time without evaluating an excessive number.
        if len(pairs) > self.settings.max_diversity_pairs:
            # Replay count caps at buffer capacity, so using environment step
            # alone would permanently freeze the pair subset late in training.
            # Rotate by a checkpointed per-update counter instead.
            offset = self._diversity_calls_int % len(pairs)
            self.set_diversity_calls(self._diversity_calls_int + 1)
            pairs = (pairs[offset:] + pairs[:offset])[: self.settings.max_diversity_pairs]
        for i, j in pairs:
            p = all_probs[i].clamp_min(eps)
            q = all_probs[j].clamp_min(eps)
            m = 0.5 * (p + q)
            js = 0.5 * (
                (p * (p.log() - m.log())).sum(-1)
                + (q * (q.log() - m.log())).sum(-1)
            )
            pair_js.append((js * active_weight).sum() / denominator)
        js_tensor = torch.stack(pair_js) if pair_js else torch.zeros(1, device=feat.device)
        js_mean = js_tensor.mean()
        js_target = torch.as_tensor(
            self.settings.action_diversity_target,
            device=js_mean.device,
            dtype=js_mean.dtype,
        )
        # Penalize every sampled duplicate pair rather than only the average.
        # A mean-only hinge can be satisfied by a few highly distinct pairs while
        # other options remain exact duplicates. Pair rotation gives all K choose
        # 2 pairs coverage over successive updates.
        pairwise_js_shortfall = torch.relu(js_target - js_tensor)
        diversity_loss = pairwise_js_shortfall.mean()

        # Magnitude and direction safeguards are computed only on actions that
        # can actually affect the environment. Invalid actions and padded/dead
        # agents must not inflate the residual guard or provide fake diversity.
        actor_count = len(tuple(actor_shape))
        action_count = int(tuple(actor_shape)[0])
        valid_mask = flat_mask.bool().reshape(n, actor_count, action_count)
        active_agents = flat_active.bool().reshape(n, actor_count, 1)
        empty = ~valid_mask.any(dim=-1, keepdim=True)
        noop = torch.zeros_like(valid_mask)
        noop[..., 0] = True
        valid_mask = torch.where(empty, noop, valid_mask)
        effective_action_mask = valid_mask & active_agents
        entry_weight = (
            effective_action_mask.float() * flat_state_weight[:, None, None]
        )
        entry_denominator = entry_weight.sum().clamp_min(1.0)

        residual = (selected_logits - flat_base).float().reshape(
            n, actor_count, action_count
        )
        base_shaped = flat_base.float().reshape(n, actor_count, action_count)
        residual_rms = (
            (residual.square() * entry_weight).sum() / entry_denominator
        ).sqrt()
        base_rms = (
            (base_shaped.square() * entry_weight).sum() / entry_denominator
        ).sqrt().clamp_min(1.0e-6)
        residual_ratio = residual_rms / base_rms

        # Collapse-only directional guard. Positive cosine near one means two
        # options are learning the same valid-action residual direction. Opposite
        # directions are intentionally not penalized because they are distinct.
        residual_all = self.all_residual_logits(flat_feat, step).float().reshape(
            n, self.num_options, actor_count, action_count
        )
        residual_all = (
            residual_all * effective_action_mask[:, None].float()
        ).reshape(n, self.num_options, -1)
        normalized = F.normalize(residual_all, dim=-1, eps=1.0e-8)
        cosine = torch.einsum("nkd,njd->nkj", normalized, normalized)
        upper = torch.triu(
            torch.ones(
                self.num_options, self.num_options,
                device=cosine.device, dtype=torch.bool
            ), diagonal=1
        )
        cosine_pairs = cosine[:, upper]
        cosine_weight = flat_state_weight[:, None]
        cosine_excess = torch.relu(
            cosine_pairs - float(self.settings.residual_cosine_target)
        )
        residual_cosine_loss = (
            cosine_excess.square() * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        weighted_cosine_mean = (
            cosine_pairs * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        residual_duplicate_fraction = (
            (cosine_pairs > float(self.settings.residual_cosine_target)).float()
            * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        residual_guard = torch.relu(
            residual_ratio
            - torch.as_tensor(
                self.settings.max_residual_to_base,
                device=residual_ratio.device,
                dtype=residual_ratio.dtype,
            )
        ).square()
        base_kl_loss = torch.relu(
            base_kl_mean
            - torch.as_tensor(
                self.settings.base_kl_target,
                device=base_kl_mean.device,
                dtype=base_kl_mean.dtype,
            )
        ).square()
        return {
            "base_kl_mean": base_kl_mean,
            "base_kl_max": base_kl_max,
            "base_kl_loss": base_kl_loss,
            "action_flip_rate": action_flip_rate,
            "js_mean": js_mean,
            "js_min": js_tensor.min(),
            "js_max": js_tensor.max(),
            "diversity_loss": diversity_loss,
            "js_shortfall_fraction": (js_tensor < js_target).float().mean(),
            "residual_rms": residual_rms,
            "base_rms": base_rms,
            "residual_ratio": residual_ratio,
            "residual_guard_loss": residual_guard,
            "residual_cosine_loss": residual_cosine_loss,
            "residual_cosine_mean": weighted_cosine_mean,
            "residual_duplicate_fraction": residual_duplicate_fraction,
            "duplicate_pair_fraction": (js_tensor < 1.0e-4).float().mean(),
        }

    def manager_statistics(
        self,
        manager_probs: torch.Tensor,
        sampled_option: torch.Tensor,
        boundary_mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        probs = manager_probs.float()
        boundary = boundary_mask.float()
        if weights is None:
            w = boundary
        else:
            w = boundary * torch.broadcast_to(weights.float(), boundary.shape)
        denominator = w.sum().clamp_min(1.0)
        marginal = (probs * w.unsqueeze(-1)).reshape(-1, self.num_options).sum(0)
        marginal = marginal / marginal.sum().clamp_min(1.0e-8)
        marginal_entropy = -(
            marginal.clamp_min(1.0e-8) * marginal.clamp_min(1.0e-8).log()
        ).sum()
        conditional_per_state = -(
            probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        conditional_entropy = (
            conditional_per_state * w
        ).sum() / denominator
        mutual_information = (marginal_entropy - conditional_entropy).clamp_min(0.0)
        normalized_mutual_information = mutual_information / max(
            math.log(float(self.num_options)), 1.0e-8
        )
        effective = marginal_entropy.exp()
        usage_max = marginal.max()
        max_excess = torch.relu(
            usage_max
            - torch.as_tensor(
                self.settings.max_usage_target,
                device=probs.device,
                dtype=probs.dtype,
            )
        )
        effective_shortfall = torch.relu(
            torch.as_tensor(
                self.settings.min_effective_options,
                device=probs.device,
                dtype=probs.dtype,
            )
            - effective
        )
        collapse_loss = max_excess.square() + (
            effective_shortfall / max(float(self.num_options), 1.0)
        ).square()
        mi_shortfall_loss = torch.relu(
            torch.as_tensor(
                self.settings.manager_mi_target_normalized,
                device=probs.device, dtype=probs.dtype,
            ) - normalized_mutual_information
        ).square()
        sampled = F.one_hot(
            sampled_option.long().clamp(0, self.num_options - 1),
            self.num_options,
        ).float()
        sampled_usage = (sampled * w.unsqueeze(-1)).reshape(-1, self.num_options).sum(0)
        sampled_usage = sampled_usage / sampled_usage.sum().clamp_min(1.0)
        return {
            "marginal": marginal,
            "sampled_usage": sampled_usage,
            "effective_count": effective,
            "marginal_entropy": marginal_entropy,
            "conditional_entropy": conditional_entropy,
            "mutual_information": mutual_information,
            "mutual_information_normalized": normalized_mutual_information,
            "usage_max": usage_max,
            "collapse_loss": collapse_loss,
            "mi_shortfall_loss": mi_shortfall_loss,
            "boundary_count": boundary.sum(),
        }

    def metadata(self) -> dict[str, Any]:
        s = self.settings
        return {
            "schema_version": self.SCHEMA_VERSION,
            "architecture": self.ARCHITECTURE,
            "enabled": bool(s.enabled),
            "num_options": s.num_options,
            "option_embedding_dim": s.option_embedding_dim,
            "age_embedding_dim": s.age_embedding_dim,
            "hidden_dim": s.hidden_dim,
            "min_duration": s.min_duration,
            "max_duration": s.max_duration,
            "feature_dim": self.feature_dim,
            "action_logit_dim": self.action_logit_dim,
            "freeze_base_actor": s.freeze_base_actor,
            "freeze_feature_adapter": s.freeze_feature_adapter,
            "worker_scale_max": s.worker_scale_max,
            "eval_sample_termination": s.eval_sample_termination,
            "eval_termination_hazard_threshold": s.eval_termination_hazard_threshold,
        }
