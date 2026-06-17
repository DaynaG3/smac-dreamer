import pathlib
import sys
import os
import signal

import numpy as np
import pytest
import torch
from gymnasium import spaces


ROOT = pathlib.Path(__file__).resolve().parent.parent
R2 = ROOT / "external" / "r2dreamer"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from envs.parallel import ParallelEnv


class TinyEnv:
    def __init__(self, slot, generation, episode_len=2):
        self.slot = int(slot)
        self.generation = int(generation)
        self.pid = __import__("os").getpid()
        self.episode_len = int(episode_len)
        self.t = 0
        self.reset_count = 0
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
        })
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def _obs(self, first=False, last=False):
        return {
            "state": np.asarray([self.slot, self.generation], dtype=np.float32),
            "is_first": np.asarray(first),
            "is_last": np.asarray(last),
            "is_terminal": np.asarray(last),
        }

    def reset(self):
        self.t = 0
        self.reset_count += 1
        return self._obs(first=True)

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return self._obs(last=done), float(self.generation), done, {}

    def close(self):
        pass

    def die(self):
        __import__("os")._exit(7)


def make_tiny(slot, generation=0):
    return lambda: TinyEnv(slot, generation)


def _step(envs, done):
    action = torch.zeros(envs.env_num, 1)
    return envs.step(action, torch.as_tensor(done, dtype=torch.bool))


def test_worker_restarts_at_episode_boundary_and_other_slots_unchanged():
    envs = ParallelEnv(make_tiny, 2, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
    try:
        _, done = _step(envs, [True, True])
        pids0 = [i["pid"] for i in envs.worker_infos()]
        _, done = _step(envs, [False, False])
        assert not done.any()
        assert [i["pid"] for i in envs.worker_infos()] == pids0
        trans, done = _step(envs, [False, False])
        assert done.all()
        assert trans["state"].shape == (2, 2)
        _, done = _step(envs, done)
        pids1 = [i["pid"] for i in envs.worker_infos()]
        assert pids1[0] != pids0[0]
        assert pids1[1] != pids0[1]
        assert [i["generation"] for i in envs.worker_infos()] == [1, 1]
    finally:
        envs.close()


def test_single_slot_restart_does_not_restart_other_slot():
    envs = ParallelEnv(make_tiny, 2, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
    try:
        _step(envs, [True, True])
        pids0 = [i["pid"] for i in envs.worker_infos()]
        _step(envs, [False, False])
        _, done = _step(envs, [False, False])
        assert done.all()
        _step(envs, [True, False])
        pids1 = [i["pid"] for i in envs.worker_infos()]
        assert pids1[0] != pids0[0]
        assert pids1[1] == pids0[1]
    finally:
        envs.close()


def test_generation_sequence_is_deterministic():
    def run_once():
        envs = ParallelEnv(make_tiny, 1, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
        states = []
        try:
            done = torch.tensor([True])
            for _ in range(5):
                trans, done = _step(envs, done)
                states.append(tuple(trans["state"][0].tolist()))
                if not bool(done[0]):
                    trans, done = _step(envs, done)
                    states.append(tuple(trans["state"][0].tolist()))
            return states
        finally:
            envs.close()

    assert run_once() == run_once()


def test_unexpected_worker_death_reports_context():
    envs = ParallelEnv(make_tiny, 1, "cpu", shutdown_timeout_seconds=1)
    try:
        _step(envs, [True])
        os.kill(envs.worker_infos()[0]["pid"], signal.SIGKILL)
        with pytest.raises(RuntimeError, match="worker slot=0.*pid=.*phase=step"):
            _step(envs, [False])
    finally:
        envs.close()
