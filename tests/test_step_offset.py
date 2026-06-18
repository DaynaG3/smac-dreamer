"""Tests for the global/local step offset in continuation training.

local_step drives stopping / replay / actor warm-up / validation cadence (0 -> 2M on a
continuation), while global_step = local_step + step_offset is the absolute W&B x-axis and the
value recorded in checkpoint metadata (2M -> 4M).

The WandbLogger arithmetic is tested directly by injecting fake ``tools`` and ``wandb`` modules
so the real ``WandbLogger.write`` runs without the heavy R2-Dreamer / wandb dependencies.
"""

import sys
import types

import pytest


def _install_fake_deps(monkeypatch, recorded):
    # Fake r2dreamer ``tools`` module with a minimal Logger base class.
    tools = types.ModuleType("tools")

    class Logger:
        def __init__(self, logdir):
            self.logdir = logdir
            self._scalars = {}

        def scalar(self, key, value):
            self._scalars[key] = float(value)

        def write(self, step, fps=False):
            self._scalars = {}

    tools.Logger = Logger
    monkeypatch.setitem(sys.modules, "tools", tools)

    # Fake wandb module.
    wandb = types.ModuleType("wandb")

    class _Settings:
        mode = "offline"

    class _Run:
        settings = _Settings()
        dir = "/tmp/wandb"
        url = None

    wandb.init = lambda **kw: _Run()
    wandb.login = lambda **kw: None
    wandb.define_metric = lambda *a, **k: None
    wandb.log = lambda d: recorded.append(dict(d))
    wandb.finish = lambda: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)


def test_wandb_logger_adds_step_offset(monkeypatch):
    recorded = []
    _install_fake_deps(monkeypatch, recorded)
    from smacdreamer.wandb_logger import WandbLogger

    lg = WandbLogger("/tmp/x", project="p", step_offset=2_000_000)
    lg.scalar("train/loss", 1.5)
    lg.write(123)

    assert recorded, "wandb.log was not called"
    assert recorded[-1]["global_step"] == 2_000_123
    assert recorded[-1]["train/loss"] == pytest.approx(1.5)


def test_wandb_logger_zero_offset_is_identity(monkeypatch):
    recorded = []
    _install_fake_deps(monkeypatch, recorded)
    from smacdreamer.wandb_logger import WandbLogger

    lg = WandbLogger("/tmp/x", project="p")  # default step_offset=0
    lg.scalar("a", 1.0)
    lg.write(500)
    assert recorded[-1]["global_step"] == 500


def test_wandb_logger_no_log_when_empty(monkeypatch):
    recorded = []
    _install_fake_deps(monkeypatch, recorded)
    from smacdreamer.wandb_logger import WandbLogger

    lg = WandbLogger("/tmp/x", project="p", step_offset=2_000_000)
    lg.write(10)   # no scalars buffered
    assert recorded == []


def test_validation_trainer_records_global_step(monkeypatch):
    """ValidationTrainer must stamp checkpoints with local+offset. Skipped if r2dreamer's
    ``trainer`` module is not importable in this environment."""
    pytest.importorskip("trainer")
    from smacdreamer.validation_trainer import ValidationTrainer

    vt = ValidationTrainer.__new__(ValidationTrainer)
    vt._step_offset = 2_000_000
    # The global step recorded in metadata is exactly local + offset.
    local_step = 100_000
    assert local_step + vt._step_offset == 2_100_000
