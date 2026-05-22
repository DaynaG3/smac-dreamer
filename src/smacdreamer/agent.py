"""SMAClite DreamerV3 agent with per-step action-availability masking."""

import elements
import jax
import jax.numpy as jnp

from dreamerv3.agent import Agent as DreamerAgent, f32, sample
import embodied.jax.outs as _outs


class SMACliteAgent(DreamerAgent):
    """DreamerV3 agent that masks unavailable actions before sampling.

    Only policy() is overridden; all training/loss/report methods
    are inherited from DreamerAgent unchanged.

    LIMITATION: masking applies to real-rollout sampling only.
    Imagination-rollout policy training (loss() / policyfn) still
    samples without masking — that flaw is addressed separately.
    """

    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        # Numerically sorted so agent 10 follows agent 9, not agent 1.
        self._mask_action_keys = sorted(
            (k for k in act_space if k.startswith('action_')),
            key=lambda k: int(k.split('_')[1]),
        )
        # Cast to int: elements.Space.high may return a numpy scalar.
        self._mask_n_actions = (
            int(act_space[self._mask_action_keys[0]].high)
            if self._mask_action_keys else 0
        )

    def _apply_avail_mask(self, policy_dists, avail_flat):
        """
        Return policy_dists with unavailable action logits suppressed.

        avail_flat : float32 (..., N*A)  1.0 = available, 0.0 = unavailable.

        For each agent i, reads dist.logits (raw network output stored by
        Categorical.__init__ with unimix=0, as constructed by Head.categorical()
        in heads.py line 107), adds -1e9 to unavailable positions, then builds
        Categorical(unimix=0) so no second mixing pass occurs.

        Safeguards:
        - Raises AttributeError if a distribution does not expose .logits.
        - Raises ValueError if avail_flat last dim != n_agents * n_actions.
        - All-zero mask: if an agent's entire mask is zero (e.g. dead unit with
          no valid actions), action 0 (noop) is forced available to prevent
          sampling from a fully-suppressed distribution.
        """
        n = self._mask_n_actions
        n_agents = len(self._mask_action_keys)
        expected = n_agents * n
        actual = avail_flat.shape[-1]
        if actual != expected:
            raise ValueError(
                f"_apply_avail_mask: avail_flat last dim {actual} != "
                f"n_agents*n_actions={n_agents}*{n}={expected}"
            )

        masked = dict(policy_dists)
        for i, key in enumerate(self._mask_action_keys):
            if key not in policy_dists:
                continue
            dist = policy_dists[key]
            if not hasattr(dist, 'logits'):
                raise AttributeError(
                    f"_apply_avail_mask: distribution for '{key}' "
                    f"({type(dist).__name__}) does not expose .logits. "
                    "Only Categorical distributions support logit masking."
                )
            agent_mask = f32(avail_flat[..., i * n : (i + 1) * n])

            # Safeguard: all-zero mask -> force action 0 (noop) available.
            all_zero = (agent_mask.sum(axis=-1, keepdims=True) == 0.0)
            noop_rescue = jnp.zeros_like(agent_mask).at[..., 0].set(1.0)
            agent_mask = jnp.where(all_zero, noop_rescue, agent_mask)

            masked_logits = dist.logits + (1.0 - agent_mask) * f32(-1e9)
            new_dist = _outs.Categorical(masked_logits)  # unimix=0: stored as-is
            # Preserve entropy-range attrs used by imag_loss metrics.
            if hasattr(dist, 'minent'):
                new_dist.minent = dist.minent
                new_dist.maxent = dist.maxent
            masked[key] = new_dist
        return masked

    def policy(self, carry, obs, mode='train'):
        """
        Mirrors dreamerv3.Agent.policy() exactly, with one addition:
        avail_actions is applied as a logit mask before sampling.

        If dreamerv3.Agent.policy() is updated upstream, this override
        must be kept in sync. The only intentional deviation is the
        _apply_avail_mask() call between pol() and sample().
        """
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs['is_first']
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw)
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        if 'avail_actions' in obs:
            policy = self._apply_avail_mask(policy, obs['avail_actions'])
        act = sample(policy)
        out = {}
        out['finite'] = elements.tree.flatdict(jax.tree.map(
            lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
            dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            out.update(elements.tree.flatdict(dict(
                enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
        return carry, act, out
