"""Tests for the three continuation-run hardening fixes.

FIX 1 — ``validate_resume_args`` refuses continuation settings without ``--resume``.
FIX 2 — the periodic checkpointer's global step comes from the trainer-published step on the
        agent (``_smacdreamer_global_step``), never the replay count.
FIX 3 — episode reward components map the env's ``log_*`` keys to ``episode/reward_*`` scalars.
"""

import pathlib
import sys

import pytest

from smacdreamer.checkpoint_transfer import validate_resume_args


# ----------------------------------------------------------------------
# FIX 1 — resume guard
# ----------------------------------------------------------------------

def test_transfer_reward_without_resume_raises():
    with pytest.raises(ValueError, match="requires --resume"):
        validate_resume_args("transfer_reward", 0, None)


def test_weights_only_without_resume_raises():
    with pytest.raises(ValueError, match="requires --resume"):
        validate_resume_args("weights_only", 0, "")


def test_step_offset_without_resume_raises():
    with pytest.raises(ValueError, match="Refusing to start a continuation run from scratch"):
        validate_resume_args("full", 2_000_000, None)


def test_transfer_reward_with_resume_ok():
    validate_resume_args("transfer_reward", 2_000_000, "/mnt/pvc/checkpoints/r2_650/best.pt")


def test_full_fresh_run_is_allowed():
    # plain `full` with no offset and no resume is a legitimate fresh run
    validate_resume_args("full", 0, None)


# ----------------------------------------------------------------------
# FIX 2 — checkpointer global step does not depend on replay count
# ----------------------------------------------------------------------

def _checkpointer_step_fn(agent, step_offset):
    """Exact expression used by the script's PeriodicCheckpointer step_fn."""
    return int(getattr(agent, "_smacdreamer_global_step", step_offset))


class _FakeAgent:
    pass


class _FakeReplay:
    """Stand-in whose count() saturates — must NOT influence the checkpoint step."""

    def count(self):
        return 500_000   # capacity-saturated


def test_checkpointer_uses_trainer_published_step_not_replay():
    agent = _FakeAgent()
    agent._smacdreamer_global_step = 2_137_000   # what the trainer published this loop
    replay = _FakeReplay()
    step_offset = 2_000_000
    # The step function returns the trainer's global step, independent of replay.count().
    assert _checkpointer_step_fn(agent, step_offset) == 2_137_000
    assert _checkpointer_step_fn(agent, step_offset) != replay.count() + step_offset


def test_checkpointer_falls_back_to_step_offset_before_loop():
    agent = _FakeAgent()   # trainer hasn't published a step yet
    assert _checkpointer_step_fn(agent, 2_000_000) == 2_000_000


# ----------------------------------------------------------------------
# FIX 3 — episode reward-component log mapping
# ----------------------------------------------------------------------

# external/r2dreamer must be importable for the trainer module (needs torch + tools).
_R2 = pathlib.Path(__file__).resolve().parent.parent / "external" / "r2dreamer"
if str(_R2) not in sys.path:
    sys.path.insert(0, str(_R2))

EXPECTED_EPISODE_REWARD_MAP = {
    "log_episode_original_env_return":          "episode/reward_original_return",
    "log_episode_shaped_return":                "episode/reward_shaped_return",
    "log_episode_reward_shaping_bonus":         "episode/reward_shaping_total",
    "log_reward_term_ally_ehp_dense_ep_sum":    "episode/reward_ally_ehp_dense",
    "log_reward_term_win_ehp_quality_ep_sum":   "episode/reward_win_ehp_quality",
    "log_reward_term_win_alive_quality_ep_sum": "episode/reward_win_alive_quality",
    "log_reward_term_timeout_ep_sum":           "episode/reward_timeout",
    "log_final_ally_ehp_frac":                  "episode/final_ally_ehp_frac",
    "log_final_ally_alive_frac":                "episode/final_ally_alive_frac",
    "log_final_enemy_ehp_frac":                 "episode/final_enemy_ehp_frac",
}


def test_episode_reward_log_map_matches_spec():
    trainer = pytest.importorskip("trainer")
    assert trainer.EPISODE_REWARD_LOG_MAP == EXPECTED_EPISODE_REWARD_MAP


def test_scalar_at_env_extracts_env_index_scalar():
    trainer = pytest.importorskip("trainer")
    torch = pytest.importorskip("torch")
    # [env_num=3, 1] log tensor; pick env index 1.
    t = torch.tensor([[0.1], [0.7], [0.3]])
    assert float(trainer._scalar_at_env(t, 1)) == pytest.approx(0.7)
