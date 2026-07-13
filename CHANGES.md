# Changes

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
