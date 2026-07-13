# Full-observability ablation (`observation.sight_range`)

## Hypothesis

R2-Dreamer trains under **partial observability**: SMAClite only reveals to each unit the
enemies/allies within `AGENT_SIGHT_RANGE = 9`. This ablation asks a single question — **how much
is fog-of-war costing the r2_2100 win rate?** — by giving the agent **full map visibility** and
measuring whether the held-out win rate rises. It is an **oracle / upper-bound** experiment: the
result is *headroom*, not necessarily a shippable partial-observation policy (see caveats).

**Metric (what should move):** held-out macro **win rate** first, then micro win rate, timeout
rate, final enemy EHP fraction, and episode length. Checkpoint selection is unchanged — always the
ORIGINAL (unshaped) macro win rate.

## What changes (and what does not)

A new optional knob `observation.sight_range` overrides SMAClite's `AGENT_SIGHT_RANGE`. Setting it
to **24** covers every r2_2100 map (max unit span <=~21), so every unit is always visible.

Unchanged: reward (`finish_trade_v3`), gamma (0.997), `max_episode_steps` (200), model, action
masking, padding, replay, sampler. **Attack availability is unaffected** — it is gated by a
separate constant `AGENT_TARGET_RANGE = 6`, so enlarging sight changes only what the agent *sees*,
not what it can *shoot*. Engagement rules are identical to the partial-obs baseline.

### Accepted coupling caveat

`AGENT_SIGHT_RANGE` is **also** SMAClite's distance-normalization divisor
(`obs[...] = distance / AGENT_SIGHT_RANGE`). Overriding the single constant therefore does two
things at once: full visibility **and** distance/dx/dy features rescaled to 9/24 ~= 0.375x of the
trained scale. We **accept** this coupling and do **not** edit `external/smaclite` to decouple the
cull radius from the divisor. Keeping the features bounded in [0,1] (large divisor) is numerically
gentler for the network than the alternative (leaving newly-visible far units at distance > 1).
The resume run fine-tunes the encoder to the rescaled inputs.

## Why tensor shapes are unchanged

Sight range affects observation *content* (which units are visible, feature magnitudes), never
observation *size*. In structured obs the layout is fixed by `max_agents` / `max_enemies` /
`max_actions` + the global unit-type vocab; `obs_size` and `n_enemies` are layout constants
independent of sight. Padding stays pinned at max_agents=9 / max_enemies=10 / max_actions=16 /
max_obs_size=255 — identical to the finish_trade_v3 checkpoint — so the resumed weights load
without any shape mismatch.

## Why `resume-mode weights_only`

The **reward is unchanged**, so the reward/value heads are still useful — load them. But the
**observation distribution shifts** (hidden units become visible; distance features rescaled), so
the LaProp optimizer momentum tuned to the sight=9 inputs is stale. `weights_only` is exactly this
trade: load the COMPLETE agent weights (`strict=False`) but reset optimizer / scheduler / scaler.
`full` would additionally restore the stale optimizer state; `transfer_reward` would needlessly
reset the still-valid reward/value heads. Start a **new** W&B run — full-vis metrics are not
comparable to the partial-obs axis. With `validation.run_at_start: true` the first validation runs
at step 0, giving a full-vis baseline of the transferred weights before any gradient step.

## Mechanism (propagation)

The trainer exports `SMACLITE_SIGHT_RANGE` in the parent process from `observation.sight_range`,
before any env is built. SMAClite envs run in **spawn** subprocesses that inherit `os.environ`;
each applies the override via `smacdreamer.sight_range.maybe_override_sight_range()`, called from:

- `r2dreamer_factory._ensure_paths()` — covers train workers, validation env children (they load
  the same factory through `isolated_env`), and the parent builder;
- `map_discovery.validate_map()` — covers discovery probes, whose in-process path does not go
  through the factory.

`AGENT_SIGHT_RANGE` is never imported by value anywhere and every use is a runtime module-global
lookup, so a single attribute assignment per process takes effect for all subsequent `get_obs()`.
Each process logs `[sight_range] applied AGENT_SIGHT_RANGE=24 (pid=...)` once. Absent env var ->
no-op. `sight_range` is recorded in `run_meta.json` and the W&B run config.

## Reading the results

Compare against the finish_trade_v3 **partial-obs** run on the same r2_2100 splits:

- **Win rate up under full vis** => partial observability is a real ceiling on r2_2100; the gap is
  the headroom. This does not make the full-vis policy deployable under the standard partial-obs
  eval — it relies on information it would not have there. Treat it as a target to close with
  better memory / representation under partial obs.
- **Win rate flat** => the current bottleneck is not perception (it is credit assignment, reward
  shaping, exploration, or capacity); full observability will not help, redirect effort.

## Run

Pilot (1M steps, `weights_only` from the best finish_trade_v3 checkpoint, new W&B run):

```bash
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_2100_finish_trade_v3_fullobs.yaml \
  --resume logs/r2dreamer/r2_2100_finish_trade_v3_from_best52/best_val_macro_winrate.pt \
  --resume-mode weights_only --steps 1000000 \
  --logdir logs/r2dreamer/r2_2100_finish_trade_v3_fullobs_from_best
```

Mechanism check (after launch): the stdout log should show the parent override line and a
worker-side `[sight_range] applied AGENT_SIGHT_RANGE=24 ...` from the spawned workers.
