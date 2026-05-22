import numpy as np
import gymnasium as gym

import embodied
import elements


class SMACliteDreamerEnv(embodied.Env):
    """Centralised-control adapter: one DreamerV3 agent drives all allied SMAClite units."""

    def __init__(self, scenario: str, max_episode_steps: int = 200, seed: int = 0):
        import smaclite  # noqa: registers smaclite/* gymnasium IDs

        self._env = gym.make(f"smaclite/{scenario}-v0")
        uw = self._env.unwrapped

        self._scenario = scenario
        self._max_episode_steps = max_episode_steps
        self._seed = seed

        self.n_agents: int = uw.n_agents
        self.n_enemies: int = uw.n_enemies
        self.n_actions: int = uw.n_actions
        self.obs_size: int = uw.obs_size

        # Validate that get_avail_actions() returns uniform-length arrays for this scenario.
        obs_tuple, _ = self._env.reset(seed=seed)
        avail = uw.get_avail_actions()
        avail_lens = [len(a) for a in avail]
        if len(set(avail_lens)) != 1 or avail_lens[0] != self.n_actions:
            raise ValueError(
                f"Non-uniform avail_actions lengths {avail_lens} for scenario '{scenario}'. "
                "Phase 1 requires all allied units to share the same n_actions."
            )

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = False

        # Per-episode accumulators (reset on each _reset() call).
        self._ep_step = 0
        self._ep_invalid_count = 0
        self._ep_total_count = 0

        # Lifetime debug counters (not exposed as log/ keys).
        self._lifetime_invalid = 0
        self._lifetime_total = 0

        # Episode metrics carried into obs until overwritten at episode end.
        self._ep_metrics = self._zero_ep_metrics()

    # ------------------------------------------------------------------
    # embodied.Env interface
    # ------------------------------------------------------------------

    @property
    def obs_space(self) -> dict:
        n = self.n_agents
        s = self.obs_size
        a = self.n_actions
        return {
            "state":         elements.Space(np.float32, (n * s,)),
            "avail_actions": elements.Space(np.float32, (n * a,)),
            "is_first":      elements.Space(bool),
            "is_last":       elements.Space(bool),
            "is_terminal":   elements.Space(bool),
            "reward":        elements.Space(np.float32),
            "log/battle_won":                     elements.Space(np.float32),
            "log/episode_invalid_action_count":   elements.Space(np.float32),
            "log/episode_total_action_count":     elements.Space(np.float32),
            "log/episode_invalid_action_rate":    elements.Space(np.float32),
            "log/step_invalid_count":             elements.Space(np.float32),
        }

    @property
    def act_space(self) -> dict:
        space = {"reset": elements.Space(bool)}
        for i in range(self.n_agents):
            space[f"action_{i}"] = elements.Space(np.int32, (), 0, self.n_actions)
        return space

    def step(self, action: dict) -> dict:
        if action["reset"] or self._done:
            return self._reset()

        # Extract per-agent integer actions from the DreamerV3 action dict.
        acts = [int(action[f"action_{i}"]) for i in range(self.n_agents)]

        # Validate and replace invalid actions BEFORE calling env.step().
        avail = self._env.unwrapped.get_avail_actions()
        acts, n_invalid = self._sanitise_actions(acts, avail)

        self._ep_total_count += self.n_agents
        self._ep_invalid_count += n_invalid
        self._lifetime_total += self.n_agents
        self._lifetime_invalid += n_invalid

        obs_tuple, reward, terminated, truncated, info = self._env.step(acts)
        self._ep_step += 1

        if self._ep_step >= self._max_episode_steps:
            truncated = True

        is_last = terminated or truncated
        is_terminal = terminated  # truncation does NOT imply terminal
        self._done = is_last

        if is_last:
            self._ep_metrics = self._compute_ep_metrics(info)

        return self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            reward=np.float32(reward),
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
            step_invalid=n_invalid,
        )

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> dict:
        obs_tuple, _ = self._env.reset()
        avail = self._env.unwrapped.get_avail_actions()

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = False
        self._ep_step = 0
        self._ep_invalid_count = 0
        self._ep_total_count = 0
        self._ep_metrics = self._zero_ep_metrics()

        return self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
            step_invalid=0,
        )

    def _sanitise_actions(
        self, acts: list, avail: list
    ) -> tuple[list, int]:
        """Replace each invalid action with the first valid action for that agent."""
        n_invalid = 0
        sanitised = []
        for i, (act, mask) in enumerate(zip(acts, avail)):
            if act < len(mask) and mask[act]:
                sanitised.append(act)
            else:
                n_invalid += 1
                valid_indices = [j for j, v in enumerate(mask) if v]
                # Fallback: noop (0) is always valid for dead agents; first valid otherwise.
                sanitised.append(valid_indices[0] if valid_indices else 0)
        return sanitised, n_invalid

    def _build_obs(
        self,
        obs_tuple,
        avail,
        reward: np.float32,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        step_invalid: int = 0,
    ) -> dict:
        state = np.concatenate(obs_tuple).astype(np.float32)
        avail_flat = np.concatenate(avail).astype(np.float32)
        return {
            "state":         state,
            "avail_actions": avail_flat,
            "is_first":      np.array(is_first, dtype=bool),
            "is_last":       np.array(is_last, dtype=bool),
            "is_terminal":   np.array(is_terminal, dtype=bool),
            "reward":        np.array(reward, dtype=np.float32),
            "log/step_invalid_count": np.array(float(step_invalid), dtype=np.float32),
            **self._ep_metrics,
        }

    def _zero_ep_metrics(self) -> dict:
        return {
            "log/battle_won":                    np.array(0.0, dtype=np.float32),
            "log/episode_invalid_action_count":  np.array(0.0, dtype=np.float32),
            "log/episode_total_action_count":    np.array(0.0, dtype=np.float32),
            "log/episode_invalid_action_rate":   np.array(0.0, dtype=np.float32),
        }

    def _compute_ep_metrics(self, info: dict) -> dict:
        total = self._ep_total_count
        invalid = self._ep_invalid_count
        rate = (invalid / total) if total > 0 else 0.0
        return {
            "log/battle_won":                    np.array(float(info.get("battle_won", False)), dtype=np.float32),
            "log/episode_invalid_action_count":  np.array(float(invalid), dtype=np.float32),
            "log/episode_total_action_count":    np.array(float(total), dtype=np.float32),
            "log/episode_invalid_action_rate":   np.array(rate, dtype=np.float32),
        }
