# Reward-quality continuation training (2M → 4M)

Continue an already-completed 2,000,000-step R2-Dreamer run for another 2,000,000 steps,
changing **only the reward** (`smaclite_default` → `win_quality_v5`). Nothing else moves:
observation representation, action masking, RSSM/actor architecture, KL weights, learning rate,
replay capacity, imagination horizon, the map sampler, and the train/validation map split are
all identical to the base run.

The goal is to bias the policy toward **higher-quality wins** (win with more surviving allied
effective-HP and more units alive) without destabilising a model that already wins, and without
ever optimising the shaped return as the headline metric — checkpoint selection stays on the
**macro held-out original win rate**.

---

## The reward: `win_quality_v5`

Defined in [`src/smacdreamer/envs/reward_registry.py`](../../src/smacdreamer/envs/reward_registry.py)
(existing rewards are untouched). Per step:

```
reward = original_smaclite_reward
       + w_ally_ehp_dense * (γ·Φ'(s') − Φ(s))         # Term 1 — dense, every step
       + w_win_ehp        * ally_ehp_frac    [win]    # Term 2 — terminal WIN only
       + w_win_alive      * ally_alive_frac  [win]    # Term 3 — terminal WIN only
       − w_timeout                           [trunc]  # safeguard — truncation only
```

| weight              | default | meaning                                                        |
|---------------------|--------:|----------------------------------------------------------------|
| `w_ally_ehp_dense`  |   0.25  | dense allied effective-HP (HP+shields) preservation potential  |
| `w_win_ehp`         |   0.50  | terminal bonus scaled by surviving allied EHP fraction (wins)  |
| `w_win_alive`       |   0.25  | terminal bonus scaled by fraction of allies alive (wins)       |
| `w_timeout`         |   0.10  | penalty applied once on time-limit truncation                  |

- **Term 1** is the same shifted potential as `ally_ehp_v4`: `Φ(s) = ally_ehp_frac − 1 ∈ [−1, 0]`,
  with a **true terminal** forcing `Φ'(s') = 0` so a terminal allied wipe is not double-counted.
  A **truncation** does *not* zero the potential (R2-Dreamer may bootstrap truncated transitions).
  In raw reward space this term telescopes to zero over an episode — it shapes *how* the agent
  fights without changing the win/loss objective.
- **Terms 2 & 3** fire **only on a true terminal win** (`terminated and battle_won`) — never on a
  loss, a truncation, or a non-terminal step. They reward winning *cleanly*.
- The **timeout** penalty only applies on truncation; `terminated` and `truncated` are mutually
  exclusive (the env passes a timeout-only `truncated` flag).

The reward returns a per-term breakdown
(`original`, `ally_ehp_dense`, `win_ehp_quality`, `win_alive_quality`, `timeout`, `shaping_total`),
logged under `episode/reward_*`.

---

## Resume modes

`--resume-mode` (or `resume.mode` in config) selects how the checkpoint is loaded:

| mode              | use when                          | what is restored                                          |
|-------------------|-----------------------------------|-----------------------------------------------------------|
| `full`           | resuming the *same* reward run    | weights **and** optimizer/scheduler/scaler/replay state   |
| `weights_only`   | new run, keep all learned weights | full `agent_state_dict`; no optimizer/training state      |
| `transfer_reward`| **this continuation**             | reward-agnostic weights only; reward/critic re-initialised|

### `transfer_reward` (the one used here)

Implemented in
[`src/smacdreamer/checkpoint_transfer.py`](../../src/smacdreamer/checkpoint_transfer.py).

**Retained** (transferred verbatim, top-level state-dict prefixes):
`encoder`, `rssm`, `actor`, `cont`, `avail_head`, `alive_head`, `prj`.

**Reset** (re-initialised, *not* loaded): `reward`, `value`, `_slow_value`, `return_ema`.

**Why reset the critic and return stats?** The value head, slow target critic, and return-EMA
normaliser all estimate the *expected return under the old reward*. With a new reward those
targets are wrong; carrying them over injects a large, biased TD error that can wreck a good
policy. Re-initialising them lets the critic recalibrate to the new return scale from scratch.

**Why not restore the optimizer/scheduler/scaler?** The LaProp moment estimates and the LR
warm-up/AGC state are tied to the old gradient statistics (especially for the now-reset critic
and reward heads). A fresh optimizer avoids stale momentum fighting the recalibration. The
encoder/RSSM/actor weights are kept; only their optimizer history is dropped.

**Frozen mirrors** (`_frozen_*`) are never loaded directly — they are rebuilt from the freshly
loaded trainable weights by `agent.clone_and_freeze()` after the partial load.

The loader **fails loudly**: every retained layer must be present and shape-compatible in the
checkpoint (no silent `strict=False`). It prints exactly what was loaded, skipped, reset, and
not-restored, and refuses to proceed if the world-model/actor architecture does not match.

---

## Actor warm-up

`resume.actor_warmup_steps: 25000`. For the first **25,000 local steps**, train everything
**except the actor** so the freshly-reset critic and reward head can recalibrate against a
stable policy before the actor starts chasing the new value estimates.

Mechanism (no `requires_grad` toggling, no whole-WM freeze): the policy loss is scaled by zero
while warm-up is active.

```python
# external/r2dreamer/dreamer.py
_pol_scale = 1.0 if self.actor_updates_enabled else 0.0
losses["policy"] = _pol_scale * torch.mean(weight[:, :-1].detach() *
                                           -(logpi * adv.detach() + self.act_entropy * entropy))
```

The actor parameters appear only in the policy loss, so a zero scale gives them zero gradient;
with a fresh optimizer (no momentum) the actor does not move. The trainer flips the flag by
local step each iteration:

```python
agent.actor_updates_enabled = (local_step >= actor_warmup_steps)
```

Logged as `train/actor_updates_enabled` (0 during warm-up, 1 after).

---

## Local vs global step

Two clocks are maintained:

- **local_step** (0 → 2,000,000): drives the stopping condition, replay count, actor warm-up
  threshold, and validation cadence. A continuation starts from an empty replay buffer, so the
  local clock begins at 0.
- **global_step = local_step + `resume.step_offset`** (2,000,000 → 4,000,000): the absolute
  W&B x-axis, the value stamped into checkpoint metadata, and the number in prints/filenames.

`resume.step_offset: 2000000`. The global step is **not** derived from the replay count alone —
it is `local_step + offset`. Validation runs every 100k local steps, i.e. at global
2.1M, 2.2M, … 4.0M; the **first validation lands at global_step 2,100,000**.

---

## Validation metrics

Held-out validation (original `smaclite_default` reward, fixed map×seed grid) additionally logs,
over **winning episodes only** (0.0 sentinel when a map/grid has no wins — never NaN):

- `val/{macro,micro}_win_final_ally_ehp_frac` — surviving allied EHP on wins,
- `val/{macro,micro}_win_alive_fraction` — fraction of allies alive on wins.

**Checkpoint selection is unchanged**: best by **macro original win rate**, tie-broken by macro
original return — **never** the shaped return. `best_val_macro_winrate.pt` now also records
`global_step`.

---

## Run it (Kubeflow)

The checkpoint path is supplied on the CLI (never hard-coded in the config) and may be an
absolute, off-repo persistent-volume path:

```bash
python scripts/train_r2dreamer_smaclite_multimap.py \
    --config configs/r2_650_win_quality_continuation.yaml \
    --resume /mnt/pvc/checkpoints/r2_650/best_val_macro_winrate.pt \
    --resume-mode transfer_reward
```

`--resume-mode`, `--step-offset`, and `--actor-warmup-steps` override the `resume:` block in the
config if given.

### Verifying the load at startup

Expect a transfer report like:

```
[transfer_reward] checkpoint: /mnt/pvc/checkpoints/r2_650/best_val_macro_winrate.pt
[transfer_reward] loaded 184 retained keys (modules: encoder, rssm, actor, cont, avail_head, alive_head, prj)
[transfer_reward] intentionally RESET (newly initialised): reward head, critic/value head, slow target critic, return-EMA
[transfer_reward] skipped 46 keys (reset modules + frozen mirrors)
[transfer_reward] NOT restored: optimizer / scheduler / AMP scaler / return stats
```

(The exact key counts depend on the architecture; any shape mismatch or missing retained layer
aborts the run with an explicit error rather than loading partially.)

Then, around global_step 2,100,000:

```
  [val global_step 2100000] macro win_rate=0.62 orig_return=18.40 (best -1.000)
  [val global_step 2100000] NEW BEST macro win_rate=0.62 (orig_return=18.40) -> best_val_macro_winrate.pt
```

`train/actor_updates_enabled` should read 0 until local step 25,000 and 1 thereafter.

---

## Tests

- [`tests/test_win_quality_v5.py`](../../tests/test_win_quality_v5.py) — reward terms, win-only
  bonuses, terminal/truncation semantics, telescoping.
- [`tests/test_checkpoint_transfer.py`](../../tests/test_checkpoint_transfer.py) — retained vs
  reset loading, loud failures, payload formats.
- [`tests/test_step_offset.py`](../../tests/test_step_offset.py) — global = local + offset.
- [`tests/test_actor_warmup.py`](../../tests/test_actor_warmup.py) — actor frozen during warm-up,
  world model still trains, actor moves after threshold.
