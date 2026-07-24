#!/usr/bin/env python3
"""Install unified adaptive map replay + candidate sequence PER.

The installer backs up every touched existing file and applies marker-checked,
idempotent text patches. It refuses to continue when a required source pattern
is absent or ambiguous.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import shutil
import sys


BUNDLE = pathlib.Path(__file__).resolve().parent
PAYLOAD = BUNDLE / "payload"


def die(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0 and new in text:
        return text
    if count != 1:
        die(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    compiled = re.compile(pattern, flags=re.MULTILINE | re.DOTALL)
    new, count = compiled.subn(lambda _match: replacement, text, count=1)
    if count == 0 and replacement in text:
        return text
    if count != 1:
        die(f"{label}: expected exactly one regex match, found {count}")
    return new


def patch_map_sampler(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text
    text = replace_once(
        text,
        "import pathlib\nimport random\n",
        "import pathlib\nimport random\nimport math\nimport torch\n\n# UNIFIED_PRIORITY_V1\n",
        "map_sampler imports",
    )
    text = replace_once(
        text,
        "        'weighted', 'curriculum',\n",
        "        'weighted', 'curriculum', 'adaptive_priority',\n",
        "map_sampler modes",
    )
    text = replace_once(
        text,
        "    def __init__(self, maps: List[MapEntry], mode: str = 'round_robin', seed: int = 0):\n",
        "    def __init__(\n"
        "        self, maps: List[MapEntry], mode: str = 'round_robin', seed: int = 0,\n"
        "        shared_probabilities=None, shared_version=None,\n"
        "    ):\n",
        "map_sampler init signature",
    )
    text = replace_once(
        text,
        "        self._rng = random.Random(seed)\n",
        "        self._rng = random.Random(seed)\n"
        "        self._shared_probabilities = shared_probabilities\n"
        "        self._shared_version = shared_version\n",
        "map_sampler shared state",
    )
    text = replace_once(
        text,
        "        if mode == 'weighted':\n"
        "            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]\n"
        "        return self.maps[0]\n\n"
        "    def _update_coverage",
        "        if mode == 'weighted':\n"
        "            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]\n"
        "        if mode == 'adaptive_priority':\n"
        "            return self._rng.choices(self.maps, weights=self._adaptive_weights(), k=1)[0]\n"
        "        return self.maps[0]\n\n"
        "    def _adaptive_weights(self):\n"
        "        probs = self._shared_probabilities\n"
        "        if probs is None:\n"
        "            return [1.0] * len(self.maps)\n"
        "        try:\n"
        "            vals = probs.detach().to(dtype=torch.float64, device='cpu').reshape(-1)\n"
        "        except Exception:\n"
        "            return [1.0] * len(self.maps)\n"
        "        if vals.numel() != len(self.maps):\n"
        "            return [1.0] * len(self.maps)\n"
        "        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)\n"
        "        if not torch.isfinite(vals).all() or float(vals.sum()) <= 0:\n"
        "            return [1.0] * len(self.maps)\n"
        "        return vals.tolist()\n\n"
        "    def _update_coverage",
        "map_sampler adaptive helper",
    )
    text = replace_once(
        text,
        "        elif mode == 'weighted':\n"
        "            entry = self._peek_cache\n"
        "            wrapped = False\n"
        "        else:\n",
        "        elif mode == 'weighted':\n"
        "            entry = self._peek_cache\n"
        "            wrapped = False\n"
        "        elif mode == 'adaptive_priority':\n"
        "            entry = self._peek_cache\n"
        "            wrapped = False\n"
        "        else:\n",
        "map_sampler next branch",
    )
    # There are two weighted->return blocks: _compute_peek was already patched
    # above; this one is _compute_next_peek.
    needle = (
        "        if mode == 'weighted':\n"
        "            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]\n"
        "        return self.maps[0]\n\n"
        "    # ------------------------------------------------------------------\n"
        "    # Class-method constructors"
    )
    replacement = (
        "        if mode == 'weighted':\n"
        "            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]\n"
        "        if mode == 'adaptive_priority':\n"
        "            return self._rng.choices(self.maps, weights=self._adaptive_weights(), k=1)[0]\n"
        "        return self.maps[0]\n\n"
        "    # ------------------------------------------------------------------\n"
        "    # Class-method constructors"
    )
    text = replace_once(text, needle, replacement, "map_sampler next peek")
    text = replace_once(
        text,
        "    def from_entries(\n"
        "        cls,\n"
        "        entries: List[MapEntry],\n"
        "        mode: str = 'shuffled_round_robin',\n"
        "        seed: int = 0,\n"
        "    ) -> 'MapSampler':\n",
        "    def from_entries(\n"
        "        cls,\n"
        "        entries: List[MapEntry],\n"
        "        mode: str = 'shuffled_round_robin',\n"
        "        seed: int = 0,\n"
        "        shared_probabilities=None,\n"
        "        shared_version=None,\n"
        "    ) -> 'MapSampler':\n",
        "map_sampler from_entries signature",
    )
    text = replace_once(
        text,
        "        return cls(maps=list(entries), mode=mode, seed=seed)\n",
        "        return cls(\n"
        "            maps=list(entries), mode=mode, seed=seed,\n"
        "            shared_probabilities=shared_probabilities,\n"
        "            shared_version=shared_version,\n"
        "        )\n",
        "map_sampler from_entries call",
    )
    return text


def patch_factory(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text
    text = replace_once(
        text,
        "    worker_generation=0, completed_episode_offset=0, include_jepa_obs=False,\n"
        "    jepa_visibility_config=None,\n"
        "):\n",
        "    worker_generation=0, completed_episode_offset=0, include_jepa_obs=False,\n"
        "    jepa_visibility_config=None,\n"
        "    shared_map_probabilities=None, shared_map_version=None,\n"
        "):\n"
        "    # UNIFIED_PRIORITY_V1\n",
        "factory worker signature",
    )
    text = replace_once(
        text,
        "    sampler_seed = _worker_seed(base_seed, worker_idx, 0)\n"
        "    simulator_seed = _worker_seed(base_seed, worker_idx, worker_generation)\n",
        "    sampler_seed = _worker_seed(\n"
        "        base_seed, worker_idx,\n"
        "        worker_generation if sampling_mode == 'adaptive_priority' else 0,\n"
        "    )\n"
        "    simulator_seed = _worker_seed(base_seed, worker_idx, worker_generation)\n",
        "factory adaptive worker seed",
    )
    text = replace_once(
        text,
        "    sampler = MapSampler.from_entries(entries, mode=sampling_mode, seed=sampler_seed)\n"
        "    sampler.advance(int(completed_episode_offset))\n",
        "    sampler = MapSampler.from_entries(\n"
        "        entries, mode=sampling_mode, seed=sampler_seed,\n"
        "        shared_probabilities=shared_map_probabilities,\n"
        "        shared_version=shared_map_version,\n"
        "    )\n"
        "    if sampling_mode != 'adaptive_priority':\n"
        "        sampler.advance(int(completed_episode_offset))\n",
        "factory worker sampler",
    )
    text = replace_once(
        text,
        "    jepa_visibility_config=None,\n"
        "):\n"
        "    \"\"\"Create multimap train + held-out eval ParallelEnv pools.\n",
        "    jepa_visibility_config=None,\n"
        "    shared_map_probabilities=None,\n"
        "    shared_map_version=None,\n"
        "):\n"
        "    \"\"\"Create multimap train + held-out eval ParallelEnv pools.\n",
        "factory pool signature",
    )
    text = replace_once(
        text,
        "            jepa_visibility_config=jepa_visibility_config,\n"
        "        )\n\n"
        "    # Eval pool:",
        "            jepa_visibility_config=jepa_visibility_config,\n"
        "            shared_map_probabilities=shared_map_probabilities,\n"
        "            shared_map_version=shared_map_version,\n"
        "        )\n\n"
        "    # Eval pool:",
        "factory training worker shared tensors",
    )
    # Evaluation remains deliberately non-adaptive. If caller supplied
    # sampling_mode=adaptive_priority, use uniform_map for held-out workers.
    text = replace_once(
        text,
        "            test_entries, pad_dims, sampling_mode, seed + 10_000, idx,\n",
        "            test_entries, pad_dims,\n"
        "            ('shuffled_round_robin' if sampling_mode == 'adaptive_priority' else sampling_mode),\n"
        "            seed + 10_000, idx,\n",
        "factory validation leakage guard",
    )
    return text


def patch_trainer(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text
    text = replace_once(
        text,
        "        self.steps = int(config.steps)\n",
        "        self.steps = int(config.steps)\n"
        "        self.start_step = int(getattr(config, 'start_step', 0) or 0)\n"
        "        self.current_step = self.start_step\n"
        "        # UNIFIED_PRIORITY_V1\n",
        "trainer start step",
    )
    text = replace_once(
        text,
        "        step = self.replay_buffer.count() * self._action_repeat\n",
        "        step = self.start_step + self.replay_buffer.count() * self._action_repeat\n"
        "        self.current_step = int(step)\n"
        "        if hasattr(self.replay_buffer, 'set_env_step'):\n"
        "            self.replay_buffer.set_env_step(step)\n",
        "trainer begin step",
    )
    text = replace_once(
        text,
        "            step += int((~done).sum()) * self._action_repeat  # step is based on env side\n"
        "            lengths += ~done\n",
        "            step += int((~done).sum()) * self._action_repeat  # step is based on env side\n"
        "            self.current_step = int(step)\n"
        "            if hasattr(self.replay_buffer, 'set_env_step'):\n"
        "                self.replay_buffer.set_env_step(step)\n"
        "            lengths += ~done\n",
        "trainer step propagation",
    )
    text = replace_once(
        text,
        "            self.replay_buffer.add_transition(trans.detach())\n",
        "            if hasattr(self.replay_buffer, 'record_collection'):\n"
        "                self.replay_buffer.record_collection(trans, env_step=step)\n"
        "            self.replay_buffer.add_transition(trans.detach())\n",
        "trainer collection feedback",
    )
    text = replace_once(
        text,
        "            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:\n",
        "            # Resume uses a new replay, so warm-up must depend on replay contents,\n"
        "            # never on the restored absolute environment step.\n"
        "            if self.replay_buffer.count() // envs.env_num > self.batch_length + 1:\n",
        "trainer replay refill warmup",
    )
    text = replace_once(
        text,
        "                for name, value in train_metrics.items():\n",
        "                if hasattr(self.replay_buffer, 'metrics'):\n"
        "                    train_metrics.update(self.replay_buffer.metrics())\n"
        "                for name, value in train_metrics.items():\n",
        "trainer priority metrics",
    )
    return text


def patch_checkpointer(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text
    text = replace_once(
        text,
        "    def __init__(self, agent, logdir, interval_seconds, step_fn=None, keep_snapshots=False):\n",
        "    def __init__(\n"
        "        self, agent, logdir, interval_seconds, step_fn=None,\n"
        "        keep_snapshots=False, extra_state_fn=None,\n"
        "    ):\n"
        "        # UNIFIED_PRIORITY_V1\n",
        "checkpointer signature",
    )
    text = replace_once(
        text,
        "        self._keep_snapshots = bool(keep_snapshots)\n",
        "        self._keep_snapshots = bool(keep_snapshots)\n"
        "        self._extra_state_fn = extra_state_fn\n",
        "checkpointer extra state field",
    )
    text = replace_once(
        text,
        "        latest = self._logdir / \"latest.pt\"\n",
        "        if self._extra_state_fn is not None:\n"
        "            extra = self._extra_state_fn()\n"
        "            if extra:\n"
        "                overlap = set(payload).intersection(extra)\n"
        "                if overlap:\n"
        "                    raise KeyError(f'extra checkpoint state overwrites keys: {sorted(overlap)}')\n"
        "                payload.update(extra)\n"
        "        latest = self._logdir / \"latest.pt\"\n",
        "checkpointer payload extension",
    )
    return text


def patch_dreamer(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text

    text = replace_once(
        text,
        "        data, index, initial = replay_buffer.sample()\n",
        "        data, sample_info, initial = replay_buffer.sample()\n"
        "        importance_weights = getattr(sample_info, 'importance_weights', None)\n"
        "        self._priority_sequence_weights = (\n"
        "            importance_weights.to(self.device) if importance_weights is not None else None\n"
        "        )\n"
        "        self._priority_sequence_priorities = None\n"
        "        self._priority_map_feedback = None\n"
        "        # UNIFIED_PRIORITY_V1\n",
        "dreamer sample metadata",
    )
    text = replace_once(
        text,
        "        # update latent vectors in replay buffer\n"
        "        replay_buffer.update(index, stoch.detach(), deter.detach())\n"
        "        return metrics\n",
        "        # update latent vectors in replay buffer\n"
        "        transition_indices = getattr(sample_info, 'transition_indices', sample_info)\n"
        "        replay_buffer.update(transition_indices, stoch.detach(), deter.detach())\n"
        "        if (\n"
        "            hasattr(sample_info, 'sequence_uids')\n"
        "            and self._priority_sequence_priorities is not None\n"
        "            and hasattr(replay_buffer, 'update_priorities')\n"
        "        ):\n"
        "            replay_buffer.update_priorities(\n"
        "                sample_info.sequence_uids,\n"
        "                self._priority_sequence_priorities,\n"
        "            )\n"
        "        if self._priority_map_feedback is not None:\n"
        "            controller = getattr(replay_buffer, 'priority_controller', None)\n"
        "            if controller is not None:\n"
        "                map_ids, errors, valid = self._priority_map_feedback\n"
        "                controller.record_critic_feedback(\n"
        "                    map_ids, errors, valid,\n"
        "                    env_step=(replay_buffer.current_env_step()\n"
        "                              if hasattr(replay_buffer, 'current_env_step') else None),\n"
        "                )\n"
        "        return metrics\n",
        "dreamer priority updates",
    )

    marker = "    def _cal_grad_jepa(self, data, initial):\n"
    next_marker = "    def _predicted_action_mask"
    if text.count(marker) != 1 or text.count(next_marker) != 1:
        die("dreamer JEPA method markers are missing or ambiguous")
    prefix, remainder = text.split(marker, 1)
    jepa_body, suffix = remainder.split(next_marker, 1)

    jepa_body = replace_once(
        jepa_body,
        "        B, T = data.shape\n"
        "        encoded = self.jepa_world_model.encode_obs(data)\n",
        "        B, T = data.shape\n"
        "        _seq_is = self._priority_sequence_weights\n"
        "        if _seq_is is None:\n"
        "            _seq_is = torch.ones(B, dtype=torch.float32, device=self.device)\n"
        "        _seq_is = _seq_is.to(device=self.device, dtype=torch.float32).reshape(B)\n"
        "        def _priority_weighted_mean(term):\n"
        "            if term.shape[0] == B:\n"
        "                per_sequence = term.reshape(B, -1).mean(-1)\n"
        "            elif term.shape[0] == B * T:\n"
        "                per_sequence = term.reshape(B, T, -1).mean((1, 2))\n"
        "            else:\n"
        "                raise RuntimeError(\n"
        "                    f'cannot align PER weights: leading dim {term.shape[0]}, B={B}, T={T}'\n"
        "                )\n"
        "            return (per_sequence * _seq_is).sum() / _seq_is.sum().clamp_min(1e-8)\n"
        "        encoded = self.jepa_world_model.encode_obs(data)\n",
        "dreamer weighted mean helper",
    )
    jepa_body = replace_once(
        jepa_body,
        '        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))\n'
        '        cont = 1.0 - to_f32(data["is_terminal"])\n'
        '        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))\n',
        '        losses["rew"] = _priority_weighted_mean(\n'
        '            -self.reward(feat).log_prob(to_f32(data["reward"]))\n'
        '        )\n'
        '        cont = 1.0 - to_f32(data["is_terminal"])\n'
        '        losses["con"] = _priority_weighted_mean(-self.cont(feat).log_prob(cont))\n',
        "dreamer reward continuation IS",
    )
    jepa_body = replace_once(
        jepa_body,
        '            losses["avail"] = torch.mean(-self.avail_head(feat).log_prob(to_f32(data["avail_actions"])))\n'
        '            losses["alive"] = torch.mean(-self.alive_head(feat).log_prob(to_f32(data["agent_alive_mask"])))\n',
        '            losses["avail"] = _priority_weighted_mean(\n'
        '                -self.avail_head(feat).log_prob(to_f32(data["avail_actions"]))\n'
        '            )\n'
        '            losses["alive"] = _priority_weighted_mean(\n'
        '                -self.alive_head(feat).log_prob(to_f32(data["agent_alive_mask"]))\n'
        '            )\n',
        "dreamer auxiliary IS",
    )
    jepa_body = replace_once(
        jepa_body,
        '        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))\n',
        '        losses["policy"] = _priority_weighted_mean(\n'
        '            weight[:, :-1].detach()\n'
        '            * -(logpi * adv.detach() + self.act_entropy * entropy)\n'
        '        )\n',
        "dreamer policy IS",
    )
    jepa_body = replace_once(
        jepa_body,
        '        losses["value"] = torch.mean(\n'
        '            weight[:, :-1].detach()\n'
        '            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[\n'
        '                :, :-1\n'
        '            ].unsqueeze(-1)\n'
        '        )\n',
        '        losses["value"] = _priority_weighted_mean(\n'
        '            weight[:, :-1].detach()\n'
        '            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[\n'
        '                :, :-1\n'
        '            ].unsqueeze(-1)\n'
        '        )\n',
        "dreamer value IS",
    )
    jepa_body = replace_once(
        jepa_body,
        '        losses["repval"] = torch.mean(\n'
        '            weight_replay[:, :-1]\n'
        '            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(-1)\n'
        '        )\n',
        '        losses["repval"] = _priority_weighted_mean(\n'
        '            weight_replay[:, :-1]\n'
        '            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[\n'
        '                :, :-1\n'
        '            ].unsqueeze(-1)\n'
        '        )\n'
        '        _priority_value = value_dist.mode()\n'
        '        _priority_error = (ret_replay.detach() - _priority_value[:, :-1].detach()).abs()\n'
        '        _priority_valid = weight_replay[:, :-1].detach()\n'
        '        _priority_num = (_priority_error * _priority_valid).reshape(B, -1).sum(-1)\n'
        '        _priority_den = _priority_valid.reshape(B, -1).sum(-1).clamp_min(1.0)\n'
        '        self._priority_sequence_priorities = _priority_num / _priority_den\n'
        '        if "log_map_id" in data:\n'
        '            _map_ids = data["log_map_id"][:, :_priority_error.shape[1]].detach()\n'
        '            _map_feedback_valid = (\n'
        '                _priority_valid * _seq_is.reshape(B, 1, 1)\n'
        '            )\n'
        '            self._priority_map_feedback = (\n'
        '                _map_ids, _priority_error, _map_feedback_valid\n'
        '            )\n'
        '            metrics["priority/map_feedback_is_weight_mean"] = _seq_is.mean()\n'
        '        metrics["priority/critic_error_mean"] = _priority_error.mean()\n'
        '        metrics["priority/sequence_priority_mean"] = self._priority_sequence_priorities.mean()\n'
        '        metrics["priority/sequence_priority_max"] = self._priority_sequence_priorities.max()\n',
        "dreamer replay critic priority",
    )

    return prefix + marker + jepa_body + next_marker + suffix


def patch_training_script(text: str) -> str:
    if "# UNIFIED_PRIORITY_V1" in text:
        return text
    text = replace_once(
        text,
        "import os\nimport pathlib\nimport sys\n",
        "import os\nimport pathlib\nimport random\nimport sys\n\nimport numpy as np\n",
        "training RNG imports",
    )
    text = replace_once(
        text,
        '    ap.add_argument("--resume", default=None, help="checkpoint path to resume model/training state")\n',
        '    ap.add_argument("--resume", default=None, help="checkpoint path to resume model/training state")\n'
        '    ap.add_argument(\n'
        '        "--resume-start-step", type=int, default=None,\n'
        '        help=("trusted absolute environment step for an old checkpoint; "\n'
        '              "overrides checkpoint[\'step\'] when supplied"),\n'
        '    )\n',
        "training resume-start-step CLI",
    )
    text = replace_once(
        text,
        "from buffer import Buffer\n"
        "from dreamer import Dreamer\n",
        "from adaptive_buffer import AdaptiveBuffer\n"
        "from dreamer import Dreamer\n"
        "from smacdreamer.adaptive_priority import AdaptivePriorityController\n"
        "# UNIFIED_PRIORITY_V1\n",
        "training imports",
    )
    text = replace_once(
        text,
        "    config.buffer.scratch_dir = str(scratch_path)\n",
        "    config.buffer.scratch_dir = str(scratch_path)\n"
        "    _adaptive_cfg = cfg.get('adaptive_priority') or OmegaConf.create({})\n"
        "    config.buffer.adaptive_priority = OmegaConf.create(\n"
        "        OmegaConf.to_container(_adaptive_cfg, resolve=True)\n"
        "        if _adaptive_cfg else {}\n"
        "    )\n",
        "training buffer adaptive config",
    )
    text = replace_once(
        text,
        "    tools.set_seed_everywhere(int(cfg.seed))\n\n"
        "    # --- Train envs ONLY",
        "    tools.set_seed_everywhere(int(cfg.seed))\n"
        "    priority_controller = AdaptivePriorityController.from_entries(\n"
        "        train_entries, _adaptive_cfg\n"
        "    )\n\n"
        "    # --- Train envs ONLY",
        "training controller construction",
    )
    text = replace_once(
        text,
        "        jepa_visibility_config=jepa_visibility_config,\n"
        "    )\n",
        "        jepa_visibility_config=jepa_visibility_config,\n"
        "        shared_map_probabilities=priority_controller.shared_probabilities,\n"
        "        shared_map_version=priority_controller.shared_version,\n"
        "    )\n",
        "training shared probabilities",
    )
    text = replace_once(
        text,
        '        "world_model": config.model.world_model,\n'
        "    })\n",
        '        "world_model": config.model.world_model,\n'
        '        "adaptive_priority": _adaptive_cfg,\n'
        "    })\n",
        "training run config",
    )
    text = replace_once(
        text,
        "        replay_buffer = Buffer(config.buffer)\n",
        "        replay_buffer = AdaptiveBuffer(config.buffer, priority_controller)\n",
        "training adaptive buffer",
    )
    text = replace_once(
        text,
        "        if args.resume:\n"
        "            ckpt = torch.load(args.resume, map_location=str(cfg.device), weights_only=False)\n",
        "        resume_step = 0\n"
        "        if args.resume:\n"
        "            ckpt = torch.load(args.resume, map_location=str(cfg.device), weights_only=False)\n"
        "            checkpoint_step = int(ckpt.get('step', 0))\n"
        "            resume_step = int(\n"
        "                args.resume_start_step\n"
        "                if args.resume_start_step is not None\n"
        "                else checkpoint_step\n"
        "            )\n"
        "            if resume_step < 0:\n"
        "                raise ValueError(f'resume start step must be non-negative, got {resume_step}')\n"
        "            if args.resume_start_step is not None:\n"
        "                print(\n"
        "                    f' [resume] trusted step override: {resume_step:,} ' \n"
        "                    f'(checkpoint stored {checkpoint_step:,})'\n"
        "                )\n"
        "            config.trainer.start_step = resume_step\n"
        "            replay_buffer.set_env_step(resume_step)\n",
        "training resume step",
    )
    text = replace_once(
        text,
        "            print(\"  [resume] replay memmap restore is not implemented; replay will refill before updates.\")\n",
        "            _rng = ckpt.get('rng_state')\n"
        "            if _rng:\n"
        "                if _rng.get('python') is not None:\n"
        "                    random.setstate(_rng['python'])\n"
        "                if _rng.get('numpy') is not None:\n"
        "                    np.random.set_state(_rng['numpy'])\n"
        "                if _rng.get('torch') is not None:\n"
        "                    torch.set_rng_state(_rng['torch'])\n"
        "                if torch.cuda.is_available() and _rng.get('torch_cuda') is not None:\n"
        "                    torch.cuda.set_rng_state_all(_rng['torch_cuda'])\n"
        "                print('  [resume] restored Python/NumPy/Torch RNG state')\n"
        "            if ckpt.get('adaptive_priority_state') is not None:\n"
        "                priority_controller.load_state_dict(\n"
        "                    ckpt['adaptive_priority_state'], strict=True\n"
        "                )\n"
        "                print(' [resume] restored adaptive map-priority state')\n"
        "            else:\n"
        "                print(' [resume] old checkpoint has no adaptive state; maps start uniform')\n"
        "            print(' [resume] replay is intentionally new; sequence priorities refill with it.')\n"
        "            print(f' [resume] absolute environment step restored to {resume_step:,}')\n",
        "training adaptive resume",
    )
    # Move trainer construction before checkpointer and always emit a complete
    # final checkpoint, even when periodic checkpointing is disabled.
    new_block = (
        "        # --- Trainer + checkpointing -------------------------------------------\n"
        "        trainer = ValidationTrainer(\n"
        "            config.trainer, replay_buffer, logger, logdir, train_envs,\n"
        "            validation_entries=val_entries, pad_dims=pad_dims, seeds=val_seeds,\n"
        "            device=str(cfg.device), gamma=float(cfg.gamma),\n"
        "            max_episode_steps=int(cfg.max_episode_steps), obs_mode=obs_mode,\n"
        "            run_at_start=val_run_at_start,\n"
        "            shutdown_timeout_seconds=float(env_lifecycle.get(\"shutdown_timeout_seconds\", 5.0)),\n"
        "        )\n"
        "        def _extra_checkpoint_state():\n"
        "            return {\n"
        "                'adaptive_priority_schema': 1,\n"
        "                'adaptive_priority_state': priority_controller.state_dict(),\n"
        "            }\n"
        "        checkpointer = PeriodicCheckpointer(\n"
        "            agent, logdir,\n"
        "            interval_seconds=max(1.0, float(cfg.get('checkpoint_every_minutes', 0) or 0) * 60.0),\n"
        "            step_fn=lambda: int(trainer.current_step),\n"
        "            extra_state_fn=_extra_checkpoint_state,\n"
        "        )\n"
        "        if float(cfg.get('checkpoint_every_minutes', 0) or 0) > 0:\n"
        "            attach_checkpointing(agent, checkpointer)\n"
        "            print(f\"  Checkpoints : every {cfg.checkpoint_every_minutes:g} min -> {logdir/'latest.pt'}\")\n\n"
        "        # --- Train -------------------------------------------------------------\n"
        "        print(f\"\\nStarting multimap training ({steps} absolute env steps; resume={resume_step})...\\n\")\n"
        "        trainer.begin(agent)\n"
        "        checkpointer.save(final=True)\n"
    )
    text = regex_once(
        text,
        r"        # --- Checkpointing -+\n.*?(?=        print\(\"\\nMultimap training complete\.\"\))",
        new_block,
        "training trainer/checkpointer block",
    )
    return text


PATCHERS = {
    "src/smacdreamer/envs/map_sampler.py": patch_map_sampler,
    "src/smacdreamer/r2dreamer_factory.py": patch_factory,
    "src/smacdreamer/checkpointing.py": patch_checkpointer,
    "external/r2dreamer/trainer.py": patch_trainer,
    "external/r2dreamer/dreamer.py": patch_dreamer,
    "scripts/train_r2dreamer_smaclite_multimap.py": patch_training_script,
}


def make_config(repo: pathlib.Path, base_config: str) -> None:
    source = pathlib.Path(base_config).expanduser()
    if not source.is_absolute():
        source = repo / source
    target = repo / "configs/r2_2100_jepa_unified_priority.yaml"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if (
            "sampling_mode: adaptive_priority" in existing
            and "adaptive_priority:" in existing
            and "candidate_multiplier:" in existing
        ):
            print(f"[OK] unified-priority config already present: {target}")
            return
        die(
            f"target config already exists but is not a recognised unified-priority "
            f"config: {target}. Move it aside and rerun."
        )
    if not source.exists():
        die(f"base config missing: {source}")
    text = source.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^sampling_mode:\s*\S+\s*$",
        "sampling_mode: adaptive_priority",
        text,
        count=1,
    )
    if "sampling_mode: adaptive_priority" not in text:
        die("could not replace sampling_mode in base config")
    text += """

# Unified automatic map collection + candidate sequence PER.
# No human map labels or event definitions are used.
adaptive_priority:
  enabled: true
  map:
    enabled: true
    error_ema_decay: 0.99
    uniform_floor: 0.10
    staleness_mix: 0.20
    update_every_feedbacks: 32
    minimum_feedback: 4
    initial_error: 1.0
  sequence:
    enabled: true
    candidate_multiplier: 4
    alpha: 0.60
    beta_start: 0.40
    beta_end: 1.00
    beta_anneal_env_steps: 2000000
    eps: 1.0e-6
    min_priority: 1.0e-3
    max_priority: 100.0
    cache_max_entries: 500000
    seed: 0
"""
    target.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--base-config",
        default="configs/r2_2100_jepa_reward_shaped.yaml",
        help=(
            "Exact YAML used by the trained run. A new unified-priority YAML "
            "is derived from it without changing reward/model/horizon settings."
        ),
    )
    args = parser.parse_args()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    import subprocess
    try:
        git_root = pathlib.Path(
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        die(f"repo is not inside a Git working tree: {repo}")
    print(f"[INFO] Git root: {git_root}")
    print(f"[INFO] Target subtree: {repo}")
    if not (repo / "external/r2dreamer/dreamer.py").exists():
        die(f"wrong repo root: {repo}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = repo.parent / f"{repo.name}_unified_priority_installer_backup_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    for relative in PATCHERS:
        source = repo / relative
        if not source.exists():
            die(f"required source missing: {relative}")
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    if args.dry_run:
        for relative, patcher in PATCHERS.items():
            patcher((repo / relative).read_text(encoding="utf-8"))
        print(f"[OK] dry-run patches matched. Backup staged at {backup}")
        shutil.rmtree(backup)
        return

    for relative, patcher in PATCHERS.items():
        path = repo / relative
        original = path.read_text(encoding="utf-8")
        patched = patcher(original)
        path.write_text(patched, encoding="utf-8")

    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(PAYLOAD)
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == ".sh" or target.name.endswith(".py") and target.parent.name == "scripts":
            target.chmod(target.stat().st_mode | 0o111)

    make_config(repo, args.base_config)

    (backup / "RESTORE_COMMAND.txt").write_text(
        "Run from the repository parent:\n"
        f"rsync -a '{backup}/' '{repo}/'\n",
        encoding="utf-8",
    )
    print(f"[OK] installed unified priority integration into {repo}")
    print(f"[OK] installer backup: {backup}")
    print("[NEXT] run scripts/static_audit_unified_priority.sh")


if __name__ == "__main__":
    main()
