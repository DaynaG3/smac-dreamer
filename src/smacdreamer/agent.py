"""SMAClite DreamerV3 agent with per-step action-availability masking."""

import elements
import jax
import jax.numpy as jnp

from dreamerv3.agent import (
    Agent as DreamerAgent, f32, sample,
    imag_loss, repl_loss, prefix, concat, sg, isimage,
)
import embodied.jax.outs as _outs


class SMACliteAgent(DreamerAgent):
    """DreamerV3 agent that masks unavailable actions before sampling.

    Overrides:
    - policy(): real-rollout action masking (Phase 1A)
    - loss(): imagination-rollout action masking (Phase 1B)

    All other methods (report, etc.) are inherited from DreamerAgent unchanged.
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

    def loss(self, carry, obs, prevact, training):
        # SYNC WARNING: this method is a verbatim copy of dreamerv3.Agent.loss()
        # with exactly two blocks marked "# SMAClite Phase 1B" changed.
        # If Agent.loss() is updated upstream, this override must be kept in sync.
        # Search for "# SMAClite Phase 1B" to find the two changed blocks.
        #
        # If obs does not contain avail_actions (non-SMAClite use), fall back to
        # the parent implementation unchanged.
        if 'avail_actions' not in obs:
            return super().loss(carry, obs, prevact, training)

        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        B, T = reset.shape
        losses = {}
        metrics = {}

        # World model
        enc_carry, enc_entries, tokens = self.enc(
            enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)
        dec_carry, dec_entries, recons = self.dec(
            dec_carry, repfeat, reset, training)
        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
        con = f32(~obs['is_terminal'])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        B, T = reset.shape
        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        # Imagination
        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)

        # SMAClite Phase 1B: extract start-point avail_actions masks.
        # obs["avail_actions"] has shape (B, T, N*A). Select the last K timesteps
        # (the imagination start points) and flatten the batch dimension to (B*K, N*A).
        # This mask is held constant across the H imagination steps — an approximation
        # because availability can change as units move, die, or change range, but it
        # is the best available proxy without a world-model decoder for avail_actions.
        _img_avail = obs['avail_actions'][:, -K:, :].reshape((B * K, -1))

        # SMAClite Phase 1B: replace unmasked policyfn with masked version.
        # Original: policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
        def policyfn(feat):
            pol_raw = self.pol(self.feat2tensor(feat), 1)
            pol_masked = self._apply_avail_mask(pol_raw, _img_avail)
            return sample(pol_masked)

        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
        first = jax.tree.map(
            lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
        inp = self.feat2tensor(imgfeat)

        # SMAClite Phase 1B: mask the policy distribution passed to imag_loss.
        # Broadcast _img_avail from (B*K, N*A) to (B*K, H+1, N*A) so that
        # _apply_avail_mask can handle the time dimension via [..., i*n:(i+1)*n] slicing.
        # Original: self.pol(inp, 2)
        _img_avail_broadc = jnp.broadcast_to(
            _img_avail[:, None, :], (B * K, H + 1, _img_avail.shape[-1]))
        _pol_dist_masked = self._apply_avail_mask(self.pol(inp, 2), _img_avail_broadc)

        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            _pol_dist_masked,
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm, self.valnorm, self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss)
        losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
        metrics.update(mets)

        # Replay
        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
            boot = imgloss_out['ret'][:, 0].reshape(B, K)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot))
            inp = self.feat2tensor(feat)
            los, reploss_out, mets = repl_loss(
                last, term, rew, boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss)
            losses.update(los)
            metrics.update(prefix(mets, 'reploss'))

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()), sorted(self.scales.keys()))
        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
        return loss, (carry, entries, outs, metrics)
