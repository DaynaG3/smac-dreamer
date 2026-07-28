# Changes

## Modifications to vendored `external/` code

`external/r2dreamer/` (vendored at `3b22567`, 2026-06-10) and `external/smaclite/` (vendored at
`8efbb6b`, 2026-06-03) are upstream libraries. Per CLAUDE.md, `external/smaclite/` is treated as
read-only and `external/r2dreamer/` is editable but changes are kept minimal and guarded. This
section is the authoritative list of every file changed since vendoring — regenerate/verify with
`git diff <vendor-commit> -- external/r2dreamer/` or `external/smaclite/`.

### `external/r2dreamer/` (6 files touched)

- **`dreamer.py`** — largest patch. Adds the action-masking system end-to-end (`config.action_masking`
  gate; `avail_head`/`alive_head` auxiliary BCE heads predicting next-step action availability and
  agent-alive state; `MaskedMultiOneHotDist` used in real inference (`act()`), imagination rollouts
  (`_imagine`), and the policy loss; frozen copies of the mask heads in `clone_and_freeze`; mask
  quality diagnostics `mask_precision`/`mask_recall`/`mask_fpr` and `_horizon_mask_diagnostics`).
  Adds explicit `amp_dtype` handling (bfloat16/float16/float32) replacing the old hardcoded fp16
  autocast — required because fp16 overflows on the large structured SMAClite observation.
  Adds `training_state_dict()`/`load_training_state_dict()` for full optimizer/scheduler/RNG
  checkpoint resume. Adds `actor_updates_enabled` actor warm-up switch (zeroes the policy-loss
  gradient so a transferred world model + fresh critic can recalibrate before the actor moves).
  Fixes `get_lr()` -> `get_last_lr()` (removed in newer PyTorch schedulers).
- **`trainer.py`** — adds `EPISODE_REWARD_LOG_MAP`, a table mapping the adapter's `log_reward_term_*`
  info keys to named W&B scalars (per-reward-term episode breakdown for `smaclite_default`,
  `finish_trade_v1`, and `finish_trade_v2` reward variants). Adds the adaptive hard-map curriculum
  hook (`agent._map_priority_tracker.record()` per finished episode, periodic
  `envs.set_map_hard_scores()` broadcast). Adds `_smacdreamer_local_step`/`_smacdreamer_global_step`
  published onto the agent (so periodic checkpointing and `--resume --step-offset` continuation
  stamp the real step, not the replay count which saturates at capacity). Adds
  `actor_updates_enabled` gating by `actor_warmup_steps`. Adds optional periodic system-metrics
  logging (`system_log_every`, `smacdreamer.system_metrics.log_system_metrics`).
- **`envs/parallel.py`** — second-largest patch. `ParallelEnv` gains worker lifecycle management:
  `max_episodes_per_worker` recycling (kills and respawns a worker subprocess after N episodes to
  bound SMAClite's per-reset native memory leak — see CLAUDE.md gotcha), generation-aware
  reconstruction (`_generations`, `_episodes_since_restart`, `_completed_episodes` tracked per
  worker slot so a respawned worker's map sampler resumes where the old one left off instead of
  restarting the curriculum), `worker_infos()`/`worker_restarts` telemetry, and
  `set_map_hard_scores()` broadcast + re-seeding of restarted workers (so the hard-map curriculum
  survives recycling). Wraps every cross-process call in try/except that raises a `RuntimeError`
  with worker pid/exitcode/generation/phase context (previously a dead worker just hung or raised
  an opaque pipe error — see CLAUDE.md's "no Python traceback = SIGKILL'd" gotcha). Guards
  `.pin_memory()` behind `torch.cuda.is_available()` (pin_memory requires a CUDA device; this
  laptop is CPU-only). Adds `close(timeout=...)` for graceful shutdown with a bounded join before
  SIGKILL fallback.
- **`buffer.py`** — adds a `storage_backend: memmap` option (`LazyMemmapStorage`, disk-backed under
  `buffer.scratch_dir`) alongside the original in-memory `tensor` backend (`LazyTensorStorage`), so
  long CPU/memmap production runs don't hold the whole replay buffer in RAM. Adds `close()` to
  release the memmap storage.
- **`optim/laprop.py`** — mechanical fix only: 5 calls to deprecated positional-scalar
  `addcmul_(scalar, a, b)` / `add_(scalar, a)` Tensor methods (removed in current PyTorch) rewritten
  to the keyword form (`addcmul_(a, b, value=scalar)` / `add_(a, alpha=scalar)`). No behavior change.
- **`tools.py`** — `Logger.write()` rewritten from a single `" / "`-joined print line to a grouped,
  wrapped, ASCII-only console printer (metrics grouped by their `prefix/` before the first `/`,
  wrapped at 4 items/row). Required because W&B wraps stdout with a cp1252 shim on Windows that
  raises `UnicodeEncodeError` on any non-ASCII character, silently killing logging mid-write (see
  CLAUDE.md's ASCII-only print rule) — the original dense one-line-per-step format was also hard to
  read once `EPISODE_REWARD_LOG_MAP` and mask diagnostics added dozens of scalars per step.

### `external/smaclite/` (3 files touched — bug fixes to the read-only simulator)

- **`env/smaclite.py`** — fixes `num_target_actions` sizing for maps with healers. Healer target
  actions are addressed by an ally's `id_in_faction` (not a compacted non-healer index), so the old
  formula under-allocated target slots and caused `index N is out of bounds for axis 0` crashes
  during discovery/observation on valid healer maps.
- **`env/units/targeters/targeter.py`** — hardens `StandardTargeter`, `HealTargeter`,
  `KamikazeTargeter`, and `LaserBeamTargeter` against a `target` that died or went `None` between
  command issue and execution (returns 0 damage / clears `origin.target` instead of raising).
  `LaserBeamTargeter` also fixes a `arctan(-1/(dx/dy))` divide-by-zero (units at the same y-position)
  by switching to `arctan2` and guards an empty-neighbours query.
- **`env/units/unit_command.py`** — `AttackUnitCommand` gets a `_target_alive()` guard applied
  consistently across `clean_up_target`/`prepare_velocity`/`execute`, so a target that dies mid-command
  no longer trips the `assert self.target.hp >= 0` or attacks a stale/`None` reference.

These three SMAClite fixes together are what CLAUDE.md refers to as "Hardened SMAClite stale target
handling for `AttackUnitCommand` and `LaserBeamTargeter`" and "Fixed SMAClite healer target-action
sizing" below.

## Runtime memory lifecycle patch

- Held-out validation now runs SMAClite environments in spawned child processes, one child per validation map, while policy inference stays in the parent process.
- Multimap training workers can be recycled at episode boundaries with generation-aware seeds.
- Replay storage supports `tensor` and TorchRL `memmap` backends; production config uses memmap under the run log directory.
- Validation at step zero is controlled by `validation.run_at_start`.
- Added lightweight process, cgroup, CUDA, replay, and worker lifecycle telemetry.
- `amp_dtype: auto` selects BF16 only when supported and otherwise uses FP32; BF16 never silently falls back to FP16.
- W&B project/entity/mode can be supplied through `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_MODE`; API keys stay outside the repo via `WANDB_API_KEY`.

## Long-run reliability follow-up

- Added CUDA preflight before map discovery; it prints torch/CUDA/GPU/arch details and fails on incompatible CUDA wheels.
- AMP now supports exact `float32`, `float16`, and `bfloat16` semantics. A40 config uses BF16; Kaggle P100/T4 config uses FP32.
- Fixed SMAClite healer target-action sizing so valid healer maps do not fail discovery with target-index out-of-bounds errors.
- Hardened SMAClite stale target handling for `AttackUnitCommand` and `LaserBeamTargeter`.
- Discovery now refuses to start when maps were skipped and reports all skipped filenames and causes together.
- Added Kaggle setup script/docs and runtime dependencies in `pyproject.toml` without listing `torch` directly.
- Checkpoints now include richer Dreamer training state and RNG state. Replay memmap resume is still not implemented; resume logs that replay refills.

## Worker recycling sampler continuity

- Worker recycling no longer changes the logical map stream. Sampler seeds are stable per worker slot; simulator seeds remain generation-dependent.
- Replacement workers receive the slot's completed-episode offset and advance the sampler cursor before the first reset.
- `MapSampler.advance(count)` restores deterministic cursor and coverage state for round-robin, shuffled-round-robin, and RNG-based modes.
- `ParallelEnv` constructor compatibility now uses signature inspection instead of broad `TypeError` fallback.

## Full-observability ablation

- New optional `observation.sight_range` config knob. When set, it overrides SMAClite's `AGENT_SIGHT_RANGE` (default 9) so units see the whole map (oracle / upper-bound experiment). Absent leaves behaviour identical to today (partial observability, sight 9).
- Propagation: the trainer exports `SMACLITE_SIGHT_RANGE` in the parent process; every spawned child (train workers, validation env children, discovery probes) inherits it and applies the override via `smacdreamer.sight_range.maybe_override_sight_range()`, called from `r2dreamer_factory._ensure_paths()` and `map_discovery.validate_map()`. Each process logs the applied value once (`[sight_range] applied AGENT_SIGHT_RANGE=...`). No edits to `external/`.
- Accepted coupling caveat: `AGENT_SIGHT_RANGE` is ALSO SMAClite's distance/dx/dy normalization divisor, so enlarging it (e.g. 9 -> 24) rescales those features (~0.375x). Attack availability uses a separate `AGENT_TARGET_RANGE` (unchanged), so engagement rules are unaffected.
- `sight_range` is recorded in `run_meta.json` and the W&B run config. Reward, gamma, model, action masking, and padding are unchanged.
- Added `configs/r2_2100_finish_trade_v3_fullobs.yaml` (sight_range 24, `validation.run_at_start: true`, resume with `--resume-mode weights_only` from the best finish_trade_v3 checkpoint) and `docs/training/full_observability_ablation.md`.
- Standalone eval (`scripts/evaluate_multimap.py`) now reconstructs the training `sight_range` (priority `run_meta.json` > `cfg.observation.sight_range` > None) and exports `SMACLITE_SIGHT_RANGE` before any env is built, so full-vis checkpoints are evaluated under the same visibility they were trained with; the value is recorded in the eval report as `"sight_range"`. Partial-obs checkpoints (no sight_range) are unaffected.

## Per-run loss-scale overrides

- The multimap training script now accepts an optional `loss_scales:` block in the config; any key under it (e.g. `repval`) is applied onto `config.model.loss_scales` before the agent is built, with an unknown-key guard so typos fail loudly. Absent block -> baseline loss scales unchanged. Overrides are captured in the W&B run config via `config.model` and logged at startup (`[loss_scales] override repval -> ...`).
- Added `configs/r2_2100_finish_trade_v3_repval015.yaml` (repval 0.15 vs the 0.3 baseline; one knob changed, `--resume-mode full` from the best finish_trade_v3 checkpoint).
