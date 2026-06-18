# Changes

## Frozen JEPA world-model backend

- Added optional `world_model.backend: jepa` alongside the default RSSM backend.
- Added a strict JEPA checkpoint loader that requires `model_state`, `memory_module_state`, metadata, and resolved config.
- Added a frozen JEPA runtime adapter with packed `stoch`/`deter` state, action adaptation, recursive imagination, and a trainable JEPA feature adapter.
- Structured SMAClite envs can expose JEPA token fields only when JEPA mode is selected; RSSM observations remain unchanged.
- Added `configs/r2_650_jepa.yaml`, checkpoint inspection/parity scripts, synthetic JEPA tests, and docs.
- Real dataset/checkpoint parity is intentionally still a release gate because the real `.npz` episodes and checkpoint are not present locally.

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
