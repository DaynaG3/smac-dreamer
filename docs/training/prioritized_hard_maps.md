# Adaptive hard-map curriculum (`prioritized_hard_maps`)

Bias the env map sampler toward maps the current policy performs **poorly** on, measured from the
policy's own **training** rollouts. This is *map-level* prioritisation — **not** transition-level
PER: there are no importance-sampling weights, no Dreamer world-model loss changes, and no
TD-error priorities. The replay buffer and the world-model objective are untouched.

## Sampling: 75 % coverage + 25 % hard maps

The new sampler mode `prioritized_hard_maps` is a mixture on every episode reset:

```
p(map) = (1 - hard_map_probability) * baseline_shuffled_round_robin   # default 75%
       +      hard_map_probability  * hard_priority(∝ hard_score)      # default 25%
```

- The **75 % baseline** is the existing `shuffled_round_robin` cycle — it still visits every map
  once per cycle, so **coverage is preserved** and no map starves.
- The **25 % hard component** samples ∝ `hard_score` among maps with enough data. Hard draws do
  **not** consume the baseline cycle, so the baseline keeps cycling underneath.
- Until hard scores exist (warm-up, or too few episodes), it behaves as pure
  `shuffled_round_robin`.

## Hard score (fixed weights)

Per-map difficulty, from an EMA of training-episode outcomes:

```
hard_score = 0.60 * (1 - win_rate_ema)
           + 0.25 * final_enemy_ehp_frac_ema
           + 0.15 * timeout_rate_ema
```

`original_env_return` is also tracked (EMA, logged / per-family) but is **not** part of the hard
score. All inputs are clipped to [0, 1]; an easy map (always wins, no surviving enemies, no
timeouts) scores 0, a maximally hard map scores 1.

## How the feedback loop works

1. **Aggregate (main process).** `MapPriorityTracker` keeps a per-map EMA of `win_rate`,
   `final_enemy_ehp_frac`, `timeout_rate`, `original_return`, keyed by the env's integer
   `log_map_id`. The trainer folds each finished training episode in via the tracker.
2. **Recompute + broadcast (cadence).** Every `map_priority.every` env steps (after
   `map_priority.warmup`), the trainer recomputes `name -> hard_score` for maps with at least
   `min_episodes` episodes and broadcasts them to **all env workers**.
3. **Apply (workers).** Each worker's `MapSampler.set_hard_scores(...)` updates its hard
   component; the next resets reflect the new scores.
4. **Survive recycling.** `ParallelEnv` caches the latest scores and re-pushes them to any worker
   it restarts (workers recycle every ~25 episodes, far more often than the broadcast), so the
   curriculum is not lost on recycle.

## Validation stays fixed and unbiased

The held-out validation path (`evaluate_heldout`, fixed map×seed grid) is **never** prioritised —
it uses a `fixed` sampler and the original reward, so checkpoint selection (macro held-out win
rate) remains an honest generalisation metric. The legacy worker eval pool is also forced back to
`shuffled_round_robin` if training uses `prioritized_hard_maps`. By default the curriculum uses
**training** episode outcomes only.

## Config

```yaml
sampling_mode: prioritized_hard_maps
map_priority:
  enabled: true
  hard_map_probability: 0.25   # mixture weight on the hard-map component
  every: 100000                # env steps between hard-score recomputes/broadcasts
  warmup: 100000               # pure baseline sampling before the first broadcast
  ema_decay: 0.98              # per-episode EMA smoothing of per-map outcomes
  min_episodes: 5              # episodes a map needs before it joins the hard component
```

`map_priority.enabled: true` **requires** `sampling_mode: prioritized_hard_maps` (the script
errors otherwise). To disable, set `enabled: false` and `sampling_mode: shuffled_round_robin`.
Ready-to-run config: [`configs/r2_650_prioritized_hard_maps.yaml`](../../configs/r2_650_prioritized_hard_maps.yaml).

```bash
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650_prioritized_hard_maps.yaml
```

## Logging (W&B / TensorBoard)

Emitted on each broadcast:

- `sampler/hard_map_probability` — the mixture weight (constant).
- `sampler/map_sample_weight/{max,min,entropy}` — spread of the **effective** per-map sampling
  distribution (low entropy = concentrated curriculum).
- `sampler/hard_score_mean` — mean hard score over eligible maps.
- `sampler/hard_score_top10` — mean of the 10 hardest maps' scores.
- `sampler/family_win_rate/<family>` — per-family mean win rate (when families are known).

## What was NOT changed

- No transition-level importance-sampling weights.
- No changes to Dreamer world-model loss reductions.
- No TD-error / replay-buffer PER.

## Tests

- [`tests/test_map_priority.py`](../../tests/test_map_priority.py) — hard-score formula, mixture
  weights, tracker EMA/cadence/logging.
- [`tests/test_prioritized_sampler.py`](../../tests/test_prioritized_sampler.py) — mixture mode
  coverage, hard-score bias, determinism, peek consistency.
