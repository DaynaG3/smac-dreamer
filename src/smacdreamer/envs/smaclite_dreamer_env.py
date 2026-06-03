import pathlib

import numpy as np
import gymnasium as gym

import embodied
import elements


class SMACliteDreamerEnv(embodied.Env):
    """Centralised-control adapter: one DreamerV3 agent drives all allied SMAClite units.

    Phase 1: single fixed scenario.
    Phase 2: pass a MapSampler to rotate across same-shape maps each episode.
    Phase 3: pass pad_dims to enable variable-size map padding.

    Invalid-action metric categories
    ---------------------------------
    post_mask_invalid  — action was invalid at step time even after the policy applied
                         _apply_avail_mask(). Subdivided into:
      timing_lag       — action was valid in the last obs mask given to the policy,
                         but invalid in the current step-time mask because unit state
                         changed between observation generation and this step.
      masking_failure  — action was already invalid in the last obs mask. Indicates
                         a failure of _apply_avail_mask() to suppress the logit.
                         Should be exactly 0 with -1e9 suppression.
    """

    def __init__(
        self,
        scenario: str,
        max_episode_steps: int = 200,
        seed: int = 0,
        map_sampler=None,
        pad_dims=None,                    # Optional[PaddingDims] — enables Phase 3 padding
        kill_reward_bonus: float = 0.0,   # legacy flat-param shaping (used when v2 disabled)
        step_penalty: float = 0.0,        # legacy flat-param shaping (used when v2 disabled)
        reward_shaping_config=None,       # Optional[RewardShapingConfig] — v2 shaping system
    ):
        from smacdreamer.envs.reward_shaping import RewardShapingConfig as _RSC
        self._scenario = scenario
        self._max_episode_steps = max_episode_steps
        self._seed = seed
        self._map_sampler = map_sampler
        self._pad_dims = pad_dims
        self._kill_reward_bonus = float(kill_reward_bonus)
        self._step_penalty = float(step_penalty)
        self._prev_n_enemies: int = 0
        self._rs: _RSC = reward_shaping_config if reward_shaping_config is not None else _RSC()
        self._use_new_shaping: bool = (
            reward_shaping_config is not None and reward_shaping_config.enabled
        )
        self._prev_n_allies: int = 0
        self._prev_enemy_hp_total: float = 0.0
        self._ep_original_return: float = 0.0
        self._ep_shaped_return: float = 0.0
        self._ep_shaping_bonus: float = 0.0
        self._ep_enemy_hp_damage: float = 0.0
        self._first_ally_death_step: int = -1
        self._ep_attack_actions: int = 0
        self._ep_move_actions: int = 0
        self._ep_noop_actions: int = 0

        if map_sampler is not None:
            init_entry = map_sampler.peek()
            self._env = self._open_env(init_entry)
            self._current_map_name = init_entry.name
            self._map_id_map = {e.name: i for i, e in enumerate(map_sampler.maps)}
        else:
            import smaclite  # noqa: registers smaclite/* gymnasium IDs
            self._env = gym.make(f"smaclite/{scenario}-v0")
            self._current_map_name = scenario
            self._map_id_map = {scenario: 0}

        uw = getattr(self._env, 'unwrapped', self._env)

        self.n_agents: int = uw.n_agents
        self.n_enemies: int = uw.n_enemies
        self.n_actions: int = uw.n_actions
        self.obs_size: int = uw.obs_size

        # Phase 1/2 shape key for exact-match validation on map switch.
        # Phase 3 uses fit-within-pad-dims check instead.
        self._shape_key = (self.n_agents, self.n_enemies, self.n_actions, self.obs_size)

        obs_tuple, _ = self._env.reset(seed=seed)
        avail = uw.get_avail_actions()
        avail_lens = [len(a) for a in avail]
        if len(set(avail_lens)) != 1 or avail_lens[0] != self.n_actions:
            raise ValueError(
                f"Non-uniform avail_actions lengths {avail_lens} for scenario "
                f"'{self._current_map_name}'. "
                "Phase 1/2 requires all allied units to share the same n_actions."
            )

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = False
        self._last_avail_returned: list = avail

        # Per-episode accumulators (reset on each _reset() call).
        self._ep_step = 0
        self._ep_invalid_count = 0        # total post-mask invalids
        self._ep_total_count = 0          # total unit-action slots checked
        self._ep_timing_lag_count = 0     # invalids due to state change between obs and step
        self._ep_masking_failure_count = 0  # invalids that were already invalid in prev mask

        self._lifetime_invalid = 0
        self._lifetime_total = 0

        self._ep_metrics = self._zero_ep_metrics()
        self._current_map_id: int = self._map_id_map.get(self._current_map_name, 0)

    # ------------------------------------------------------------------
    # embodied.Env interface
    # ------------------------------------------------------------------

    @property
    def obs_space(self) -> dict:
        if self._pad_dims is not None:
            A = self._pad_dims.max_agents
            C = self._pad_dims.max_actions
            O = self._pad_dims.max_obs_size
        else:
            A = self.n_agents
            C = self.n_actions
            O = self.obs_size

        result = {
            "state":         elements.Space(np.float32, (A * O,)),
            "avail_actions": elements.Space(np.float32, (A * C,)),
            "is_first":      elements.Space(bool),
            "is_last":       elements.Space(bool),
            "is_terminal":   elements.Space(bool),
            "reward":        elements.Space(np.float32),
            # --- Episode-level metrics (new canonical names) ---
            "log/battle_won":                          elements.Space(np.float32),
            "log/post_mask_invalid_action_count":      elements.Space(np.float32),
            "log/post_mask_invalid_action_rate":       elements.Space(np.float32),
            "log/total_action_count":                  elements.Space(np.float32),
            "log/timing_lag_invalid_action_count":     elements.Space(np.float32),
            "log/timing_lag_invalid_action_rate":      elements.Space(np.float32),
            "log/masking_failure_count":               elements.Space(np.float32),
            "log/masking_failure_rate":                elements.Space(np.float32),
            # --- Episode-level metrics (old aliases) ---
            "log/episode_invalid_action_count":        elements.Space(np.float32),
            "log/episode_total_action_count":          elements.Space(np.float32),
            "log/episode_invalid_action_rate":         elements.Space(np.float32),
            # --- Step-level metrics (new canonical names) ---
            "log/step_post_mask_invalid_count":        elements.Space(np.float32),
            "log/step_timing_lag_invalid_count":       elements.Space(np.float32),
            "log/step_masking_failure_count":          elements.Space(np.float32),
            "log/step_avail_mask_mismatch_count":      elements.Space(np.float32),
            # --- Legacy flat-param shaping diagnostics (0.0 when v2 is active) ---
            "log/step_kill_bonus":                     elements.Space(np.float32),
            "log/step_step_penalty":                   elements.Space(np.float32),
            # --- V2 shaping components (distinct namespace; 0.0 when v2 disabled) ---
            "log/step_v2_win_bonus":                   elements.Space(np.float32),
            "log/step_v2_loss_penalty":                elements.Space(np.float32),
            "log/step_v2_enemy_kill_bonus":            elements.Space(np.float32),
            "log/step_v2_ally_death_penalty":          elements.Space(np.float32),
            "log/step_v2_ally_survival_bonus":         elements.Space(np.float32),
            "log/step_v2_step_penalty":                elements.Space(np.float32),
            # --- Combat state (always present) ---
            "log/enemy_kills_this_step":               elements.Space(np.float32),
            "log/ally_deaths_this_step":               elements.Space(np.float32),
            "log/enemies_alive":                       elements.Space(np.float32),
            "log/allies_alive":                        elements.Space(np.float32),
            # --- Reward breakdown (always present) ---
            "log/original_env_reward":                 elements.Space(np.float32),
            "log/shaped_reward":                       elements.Space(np.float32),
            "log/reward_shaping_bonus":                elements.Space(np.float32),
            "log/reward_shaping_enabled":              elements.Space(np.float32),
            # --- Episode-level return totals ---
            "log/episode_original_env_return":         elements.Space(np.float32),
            "log/episode_shaped_return":               elements.Space(np.float32),
            "log/episode_reward_shaping_bonus":        elements.Space(np.float32),
            # --- Damage-progress shaping (v2 damage_delta_scale) ---
            "log/enemy_hp_total":                      elements.Space(np.float32),
            "log/enemy_hp_damage_this_step":           elements.Space(np.float32),
            "log/step_v2_damage_progress":             elements.Space(np.float32),
            "log/episode_enemy_hp_damage":             elements.Space(np.float32),
            "log/final_enemy_hp_total":                elements.Space(np.float32),
            # --- Combat behaviour diagnostics ---
            "log/first_ally_death_step":               elements.Space(np.float32),
            "log/episode_attack_action_rate":          elements.Space(np.float32),
            "log/episode_move_action_rate":            elements.Space(np.float32),
            "log/episode_noop_rate":                   elements.Space(np.float32),
            # --- Step-level metrics (old aliases) ---
            "log/step_invalid_count":                  elements.Space(np.float32),
            "log/step_invalid_was_prev_valid_count":   elements.Space(np.float32),
            "log/step_invalid_was_prev_invalid_count": elements.Space(np.float32),
            # --- Map and padding metrics ---
            "log/map_id":                              elements.Space(np.float32),
            "log/num_real_agents":                     elements.Space(np.float32),
            "log/num_real_enemies":                    elements.Space(np.float32),
            "log/padded_agent_count":                  elements.Space(np.float32),
            "log/extra_real_agent_action_slot_count":  elements.Space(np.float32),
            "log/padded_agent_action_slot_count":      elements.Space(np.float32),
            "log/ignored_padded_agent_action_count":   elements.Space(np.float32),
            "log/agent_mask_sum":                      elements.Space(np.float32),
        }
        if self._pad_dims is not None:
            result["agent_mask"]             = elements.Space(np.float32, (A,))
            result["real_agent_action_mask"] = elements.Space(np.float32, (A * C,))
        return result

    @property
    def act_space(self) -> dict:
        n_acts   = self._pad_dims.max_actions if self._pad_dims else self.n_actions
        n_agents = self._pad_dims.max_agents  if self._pad_dims else self.n_agents
        space = {"reset": elements.Space(bool)}
        for i in range(n_agents):
            space[f"action_{i}"] = elements.Space(np.int32, (), 0, n_acts)
        return space

    def step(self, action: dict) -> dict:
        if action["reset"] or self._done:
            return self._reset()

        # Only extract actions for real agents; padded agent action keys are ignored.
        acts = [int(action[f"action_{i}"]) for i in range(self.n_agents)]

        avail = getattr(self._env, 'unwrapped', self._env).get_avail_actions()
        acts, n_invalid, n_was_prev_valid, n_was_prev_invalid = (
            self._sanitise_actions(acts, avail)
        )

        # Verify every sanitised action is valid under the current step-time mask.
        for _i, (_act, _mask) in enumerate(zip(acts, avail)):
            _valid = [j for j, v in enumerate(_mask) if v]
            if _valid and not (_act < len(_mask) and _mask[_act]):
                raise AssertionError(
                    f"Sanitised action is invalid: agent={_i}, action={_act}, "
                    f"current_avail={list(_mask)}, "
                    f"prev_avail={list(self._last_avail_returned[_i]) if _i < len(self._last_avail_returned) else 'N/A'}, "
                    f"map={self._current_map_name}"
                )

        n_mask_mismatch = sum(
            int(bool(c) != bool(p))
            for curr_mask, prev_mask in zip(avail, self._last_avail_returned)
            for c, p in zip(curr_mask, prev_mask)
        )

        self._ep_total_count += self.n_agents
        self._ep_invalid_count += n_invalid
        self._ep_timing_lag_count += n_was_prev_valid
        self._ep_masking_failure_count += n_was_prev_invalid
        self._lifetime_total += self.n_agents
        self._lifetime_invalid += n_invalid

        obs_tuple, reward, terminated, truncated, info = self._env.step(acts)
        self._ep_step += 1

        # --- Reward shaping ---
        # Dead units are removed from uw_post.agents / uw_post.enemies immediately
        # by SMACliteEnv.__update_deaths() after each inner step (verified).
        uw_post = getattr(self._env, 'unwrapped', self._env)
        cur_n_enemies = len(uw_post.enemies)
        cur_n_allies  = len(uw_post.agents)
        kill_delta    = max(0, self._prev_n_enemies - cur_n_enemies)
        ally_deaths   = max(0, self._prev_n_allies  - cur_n_allies)
        self._prev_n_enemies = cur_n_enemies
        self._prev_n_allies  = cur_n_allies

        # Enemy HP tracking (alive enemies only; dead units are already removed).
        cur_enemy_hp_total = float(sum(u.hp for u in uw_post.enemies.values()))
        enemy_hp_damage_this_step = max(0.0, self._prev_enemy_hp_total - cur_enemy_hp_total)
        self._prev_enemy_hp_total = cur_enemy_hp_total
        self._ep_enemy_hp_damage += enemy_hp_damage_this_step

        # First ally death step.
        if ally_deaths > 0 and self._first_ally_death_step < 0:
            self._first_ally_death_step = self._ep_step

        # Action-type breakdown for the real agents this step.
        for _act_idx in acts:
            if _act_idx <= 1:
                self._ep_noop_actions += 1
            elif _act_idx <= 5:
                self._ep_move_actions += 1
            else:
                self._ep_attack_actions += 1

        # Compute is_last here so terminal_loss covers both env termination AND
        # our own step-limit truncation (which would otherwise be set after this block).
        if self._ep_step >= self._max_episode_steps:
            truncated = True
        is_last     = terminated or truncated
        is_terminal = terminated

        base_reward = np.float32(reward)  # raw normalised SMAClite reward

        if self._use_new_shaping:
            rs = self._rs
            # Terminal signals: applied exactly once on the episode's last step.
            # loss_penalty fires on ANY non-winning episode end, including time-limit truncation.
            terminal_win  = is_last and     info.get("battle_won", False)
            terminal_loss = is_last and not info.get("battle_won", False)
            v2_win_bonus    = rs.win_bonus    if terminal_win  else 0.0
            v2_loss_penalty = rs.loss_penalty if terminal_loss else 0.0  # negative in config

            v2_enemy_kill    = float(kill_delta)   * rs.enemy_kill_bonus
            v2_ally_death    = float(ally_deaths)  * rs.ally_death_penalty  # negative in config
            v2_ally_survival = float(cur_n_allies) * rs.ally_survival_bonus
            v2_step_penalty  = rs.step_penalty
            v2_damage_progress = enemy_hp_damage_this_step * rs.damage_delta_scale

            shaping_bonus = (
                v2_win_bonus + v2_loss_penalty
                + v2_enemy_kill + v2_ally_death
                + v2_ally_survival - v2_step_penalty
                + v2_damage_progress
            )
            shaped_reward = base_reward + np.float32(shaping_bonus)

            # Legacy flat-param diagnostics are zero when v2 is active.
            step_kill_bonus   = 0.0
            step_step_penalty = 0.0
        else:
            # Legacy flat-param shaping: kill_reward_bonus / step_penalty.
            # Behaviour identical to pre-v2 code.
            step_kill_bonus   = float(kill_delta) * self._kill_reward_bonus
            step_step_penalty = self._step_penalty
            shaping_bonus     = step_kill_bonus - step_step_penalty
            shaped_reward     = base_reward + np.float32(shaping_bonus)

            v2_win_bonus = v2_loss_penalty = v2_enemy_kill = 0.0
            v2_ally_death = v2_ally_survival = v2_step_penalty = 0.0
            v2_damage_progress = 0.0

        # Episode-level return accumulation.
        self._ep_original_return += float(base_reward)
        self._ep_shaped_return   += float(shaped_reward)
        self._ep_shaping_bonus   += float(shaping_bonus)

        self._done = is_last

        if is_last:
            self._ep_metrics = self._compute_ep_metrics(info)

        return self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            reward=shaped_reward,
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
            step_invalid=n_invalid,
            step_timing_lag=n_was_prev_valid,
            step_masking_failure=n_was_prev_invalid,
            step_mask_mismatch=n_mask_mismatch,
            step_kill_bonus=step_kill_bonus,
            step_step_penalty=step_step_penalty,
            step_v2_win_bonus=v2_win_bonus,
            step_v2_loss_penalty=v2_loss_penalty,
            step_v2_enemy_kill_bonus=v2_enemy_kill,
            step_v2_ally_death_penalty=v2_ally_death,
            step_v2_ally_survival_bonus=v2_ally_survival,
            step_v2_step_penalty=v2_step_penalty,
            step_v2_damage_progress=v2_damage_progress,
            enemy_kills_this_step=kill_delta,
            ally_deaths_this_step=ally_deaths,
            enemies_alive=cur_n_enemies,
            allies_alive=cur_n_allies,
            original_env_reward=float(base_reward),
            shaped_reward_val=float(shaped_reward),
            reward_shaping_bonus=float(shaping_bonus),
            enemy_hp_total=cur_enemy_hp_total,
            enemy_hp_damage_this_step=enemy_hp_damage_this_step,
        )

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_env(self, entry):
        if entry.type == 'builtin':
            import smaclite  # noqa
            return gym.make(f'smaclite/{entry.name}-v0')
        root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
        abs_path = root / entry.path
        from smaclite.env.smaclite import SMACliteEnv as _SMACliteEnv
        return _SMACliteEnv(map_file=str(abs_path))

    def _reset(self) -> dict:
        if self._map_sampler is not None:
            entry = self._map_sampler.next()
            if entry.name != self._current_map_name:
                self._env.close()
                self._env = self._open_env(entry)
                uw = getattr(self._env, 'unwrapped', self._env)
                new_shape = (uw.n_agents, uw.n_enemies, uw.n_actions, uw.obs_size)

                if self._pad_dims is not None:
                    # Phase 3: verify new map fits within padding dims.
                    if (uw.n_agents > self._pad_dims.max_agents or
                            uw.n_actions > self._pad_dims.max_actions or
                            uw.obs_size > self._pad_dims.max_obs_size):
                        raise ValueError(
                            f"Map '{entry.name}' shape {new_shape} exceeds padding dims "
                            f"(max_agents={self._pad_dims.max_agents}, "
                            f"max_actions={self._pad_dims.max_actions}, "
                            f"max_obs_size={self._pad_dims.max_obs_size}). "
                            "Update the 'padding' block in the Phase 3 manifest."
                        )
                    # Update real dims; padded dims stay fixed via _pad_dims.
                    self.n_agents = uw.n_agents
                    self.n_enemies = uw.n_enemies
                    self.n_actions = uw.n_actions
                    self.obs_size = uw.obs_size
                else:
                    # Phase 1/2: shape must match exactly.
                    if new_shape != self._shape_key:
                        raise ValueError(
                            f"Map '{entry.name}' shape {new_shape} != "
                            f"expected {self._shape_key}. "
                            "All Phase 2 maps must share the same "
                            "n_agents/n_enemies/n_actions/obs_size. "
                            f"Expected state shape {(self.n_agents * self.obs_size,)}, "
                            f"avail_actions shape {(self.n_agents * self.n_actions,)}, "
                            f"action keys action_0..action_{self.n_agents - 1}."
                        )
                self._current_map_name = entry.name
                self._current_map_id = self._map_id_map.get(entry.name, 0)

        obs_tuple, _ = self._env.reset()
        uw = getattr(self._env, 'unwrapped', self._env)
        avail = uw.get_avail_actions()
        self._prev_n_enemies = len(uw.enemies)
        self._prev_n_allies  = len(uw.agents)
        self._prev_enemy_hp_total = float(sum(u.hp for u in uw.enemies.values()))

        self._last_obs_tuple = obs_tuple
        self._last_avail = avail
        self._done = False
        self._ep_step = 0
        self._ep_invalid_count = 0
        self._ep_total_count = 0
        self._ep_timing_lag_count = 0
        self._ep_masking_failure_count = 0
        self._ep_original_return = 0.0
        self._ep_shaped_return   = 0.0
        self._ep_shaping_bonus   = 0.0
        self._ep_enemy_hp_damage = 0.0
        self._first_ally_death_step = -1
        self._ep_attack_actions = 0
        self._ep_move_actions   = 0
        self._ep_noop_actions   = 0
        self._ep_metrics = self._zero_ep_metrics()

        return self._build_obs(
            obs_tuple=obs_tuple,
            avail=avail,
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
            step_invalid=0,
            step_timing_lag=0,
            step_masking_failure=0,
            step_mask_mismatch=0,
            step_kill_bonus=0.0,
            step_step_penalty=0.0,
        )

    def _sanitise_actions(
        self, acts: list, avail: list
    ) -> tuple[list, int, int, int]:
        """Replace each invalid action with the first valid fallback.

        Returns (sanitised, n_post_mask_invalid, n_timing_lag, n_masking_failure).

        Action categories
        -----------------
        valid            — action is valid in the current step-time mask; passed through.
        post_mask_invalid — action is invalid at step time despite the policy having
                            applied _apply_avail_mask(). Split into:
          timing_lag     — action was valid in the last obs mask returned to the policy
                           but invalid now because unit state changed between obs
                           generation and this step. Expected source of all invalids.
          masking_failure — action was already invalid in the last obs mask. Indicates
                            that _apply_avail_mask() failed to suppress it. Should be
                            exactly 0 with -1e9 logit suppression.
        """
        n_invalid = 0
        n_timing_lag = 0
        n_masking_failure = 0
        sanitised = []
        for i, (act, mask) in enumerate(zip(acts, avail)):
            if act < len(mask) and mask[act]:
                sanitised.append(act)
            else:
                n_invalid += 1
                prev_mask = (
                    self._last_avail_returned[i]
                    if i < len(self._last_avail_returned)
                    else mask
                )
                if act < len(prev_mask) and prev_mask[act]:
                    n_timing_lag += 1
                else:
                    n_masking_failure += 1
                valid_indices = [j for j, v in enumerate(mask) if v]
                sanitised.append(valid_indices[0] if valid_indices else 0)
        return sanitised, n_invalid, n_timing_lag, n_masking_failure

    def _build_obs(
        self,
        obs_tuple,
        avail,
        reward: np.float32,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        step_invalid: int = 0,
        step_timing_lag: int = 0,
        step_masking_failure: int = 0,
        step_mask_mismatch: int = 0,
        step_kill_bonus: float = 0.0,
        step_step_penalty: float = 0.0,
        step_v2_win_bonus: float = 0.0,
        step_v2_loss_penalty: float = 0.0,
        step_v2_enemy_kill_bonus: float = 0.0,
        step_v2_ally_death_penalty: float = 0.0,
        step_v2_ally_survival_bonus: float = 0.0,
        step_v2_step_penalty: float = 0.0,
        step_v2_damage_progress: float = 0.0,
        enemy_kills_this_step: int = 0,
        ally_deaths_this_step: int = 0,
        enemies_alive: int = 0,
        allies_alive: int = 0,
        original_env_reward: float = 0.0,
        shaped_reward_val: float = 0.0,
        reward_shaping_bonus: float = 0.0,
        enemy_hp_total: float = 0.0,
        enemy_hp_damage_this_step: float = 0.0,
    ) -> dict:
        self._last_avail_returned = avail

        if self._pad_dims is not None:
            from smacdreamer.envs.padding import (
                pad_state, pad_avail, make_agent_mask, make_real_agent_action_mask,
            )
            MA = self._pad_dims.max_agents
            MC = self._pad_dims.max_actions
            state     = pad_state(obs_tuple, self.obs_size, MA, self._pad_dims.max_obs_size)
            avail_flat = pad_avail(avail, self.n_actions, MA, MC)
            agent_mask             = make_agent_mask(self.n_agents, MA)
            real_agent_action_mask = make_real_agent_action_mask(agent_mask, MC)
            n_padded              = MA - self.n_agents
            extra_real_slots      = self.n_agents * (MC - self.n_actions)
            padded_agent_slots    = n_padded * MC
            ignored_pad_actions   = n_padded
            agent_mask_sum        = float(self.n_agents)
        else:
            state      = np.concatenate(obs_tuple).astype(np.float32)
            avail_flat = np.concatenate(avail).astype(np.float32)
            n_padded = extra_real_slots = padded_agent_slots = ignored_pad_actions = 0
            agent_mask_sum = float(self.n_agents)

        _f = lambda x: np.array(float(x), dtype=np.float32)
        result = {
            "state":         state,
            "avail_actions": avail_flat,
            "is_first":      np.array(is_first, dtype=bool),
            "is_last":       np.array(is_last, dtype=bool),
            "is_terminal":   np.array(is_terminal, dtype=bool),
            "reward":        np.array(reward, dtype=np.float32),
            # Step-level metrics (new canonical names)
            "log/step_post_mask_invalid_count":        _f(step_invalid),
            "log/step_timing_lag_invalid_count":       _f(step_timing_lag),
            "log/step_masking_failure_count":          _f(step_masking_failure),
            "log/step_avail_mask_mismatch_count":      _f(step_mask_mismatch),
            # Legacy flat-param shaping diagnostics (0.0 when v2 active)
            "log/step_kill_bonus":                     _f(step_kill_bonus),
            "log/step_step_penalty":                   _f(step_step_penalty),
            # V2 shaping components (distinct namespace; 0.0 when v2 disabled)
            "log/step_v2_win_bonus":                   _f(step_v2_win_bonus),
            "log/step_v2_loss_penalty":                _f(step_v2_loss_penalty),
            "log/step_v2_enemy_kill_bonus":            _f(step_v2_enemy_kill_bonus),
            "log/step_v2_ally_death_penalty":          _f(step_v2_ally_death_penalty),
            "log/step_v2_ally_survival_bonus":         _f(step_v2_ally_survival_bonus),
            "log/step_v2_step_penalty":                _f(step_v2_step_penalty),
            "log/step_v2_damage_progress":             _f(step_v2_damage_progress),
            # Combat state (always present)
            "log/enemy_kills_this_step":               _f(enemy_kills_this_step),
            "log/ally_deaths_this_step":               _f(ally_deaths_this_step),
            "log/enemies_alive":                       _f(enemies_alive),
            "log/allies_alive":                        _f(allies_alive),
            "log/enemy_hp_total":                      _f(enemy_hp_total),
            "log/enemy_hp_damage_this_step":           _f(enemy_hp_damage_this_step),
            # Reward breakdown (always present)
            "log/original_env_reward":                 _f(original_env_reward),
            "log/shaped_reward":                       _f(shaped_reward_val),
            "log/reward_shaping_bonus":                _f(reward_shaping_bonus),
            "log/reward_shaping_enabled":              _f(1.0 if self._use_new_shaping else 0.0),
            # Step-level metrics (old aliases)
            "log/step_invalid_count":                  _f(step_invalid),
            "log/step_invalid_was_prev_valid_count":   _f(step_timing_lag),
            "log/step_invalid_was_prev_invalid_count": _f(step_masking_failure),
            # Map and padding metrics
            "log/map_id":                              _f(self._current_map_id),
            "log/num_real_agents":                     _f(self.n_agents),
            "log/num_real_enemies":                    _f(self.n_enemies),
            "log/padded_agent_count":                  _f(n_padded),
            "log/extra_real_agent_action_slot_count":  _f(extra_real_slots),
            "log/padded_agent_action_slot_count":      _f(padded_agent_slots),
            "log/ignored_padded_agent_action_count":   _f(ignored_pad_actions),
            "log/agent_mask_sum":                      _f(agent_mask_sum),
            # Episode metrics (carried forward; overwritten at episode end)
            **self._ep_metrics,
        }
        if self._pad_dims is not None:
            result["agent_mask"]             = agent_mask
            result["real_agent_action_mask"] = real_agent_action_mask
        return result

    def _zero_ep_metrics(self) -> dict:
        _f = lambda: np.array(0.0, dtype=np.float32)
        return {
            # New canonical names
            "log/battle_won":                          _f(),
            "log/post_mask_invalid_action_count":      _f(),
            "log/post_mask_invalid_action_rate":       _f(),
            "log/total_action_count":                  _f(),
            "log/timing_lag_invalid_action_count":     _f(),
            "log/timing_lag_invalid_action_rate":      _f(),
            "log/masking_failure_count":               _f(),
            "log/masking_failure_rate":                _f(),
            # Old aliases
            "log/episode_invalid_action_count":        _f(),
            "log/episode_total_action_count":          _f(),
            "log/episode_invalid_action_rate":         _f(),
            # Episode-level return totals
            "log/episode_original_env_return":         _f(),
            "log/episode_shaped_return":               _f(),
            "log/episode_reward_shaping_bonus":        _f(),
            # Damage-progress diagnostics
            "log/episode_enemy_hp_damage":             _f(),
            "log/final_enemy_hp_total":                _f(),
            "log/first_ally_death_step":               _f(),
            "log/episode_attack_action_rate":          _f(),
            "log/episode_move_action_rate":            _f(),
            "log/episode_noop_rate":                   _f(),
        }

    def _compute_ep_metrics(self, info: dict) -> dict:
        total   = self._ep_total_count
        invalid = self._ep_invalid_count
        lag     = self._ep_timing_lag_count
        failure = self._ep_masking_failure_count
        inv_rate  = invalid / total if total > 0 else 0.0
        lag_rate  = lag     / total if total > 0 else 0.0
        fail_rate = failure / total if total > 0 else 0.0
        _f = lambda x: np.array(float(x), dtype=np.float32)
        return {
            # New canonical names
            "log/battle_won":                          _f(info.get("battle_won", False)),
            "log/post_mask_invalid_action_count":      _f(invalid),
            "log/post_mask_invalid_action_rate":       _f(inv_rate),
            "log/total_action_count":                  _f(total),
            "log/timing_lag_invalid_action_count":     _f(lag),
            "log/timing_lag_invalid_action_rate":      _f(lag_rate),
            "log/masking_failure_count":               _f(failure),
            "log/masking_failure_rate":                _f(fail_rate),
            # Old aliases
            "log/episode_invalid_action_count":        _f(invalid),
            "log/episode_total_action_count":          _f(total),
            "log/episode_invalid_action_rate":         _f(inv_rate),
            # Episode-level return totals
            "log/episode_original_env_return":         _f(self._ep_original_return),
            "log/episode_shaped_return":               _f(self._ep_shaped_return),
            "log/episode_reward_shaping_bonus":        _f(self._ep_shaping_bonus),
            # Damage-progress diagnostics
            "log/episode_enemy_hp_damage":             _f(self._ep_enemy_hp_damage),
            "log/final_enemy_hp_total":                _f(self._prev_enemy_hp_total),
            "log/first_ally_death_step":               _f(self._first_ally_death_step),
            "log/episode_attack_action_rate":          _f(
                self._ep_attack_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
            "log/episode_move_action_rate":            _f(
                self._ep_move_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
            "log/episode_noop_rate":                   _f(
                self._ep_noop_actions / self._ep_total_count if self._ep_total_count > 0 else 0.0
            ),
        }
