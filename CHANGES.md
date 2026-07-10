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

## 2v1-stalker distribution-isolation experiment (finish_trade_v3)

- Added `configs/r2_2v1_stalker_240_finish_trade_v3_from_best556.yaml`: a simplified 2v1-only
  STALKER dataset (`configs/maps/r2_smaclite_simple_2v1_stalker_240_configs`, 160 train / 40
  validation / 40 blind_iid) to isolate whether the best r2_2100 `finish_trade_v3` checkpoint
  (best556) adapts quickly to a simpler distribution without sampler bias.
- Reward, gamma (0.997), `max_episode_steps` (200), model, structured observation, and action
  space are unchanged vs `r2_2100_finish_trade_v3.yaml`, so the run resumes with
  `--resume-mode full`. Only the map distribution, sampler, W&B identity, and logdir differ.
- Uses `sampling_mode: shuffled_round_robin` with `map_priority.enabled: false` (uniform coverage,
  no hard-map bias) and `validation.run_at_start: true`.
- Pins an explicit `padding` block (max_agents=9, max_enemies=10, max_actions=16, max_obs_size=255)
  to the best556 checkpoint shape so the 2v1 set cannot shrink the model dims and break `full`
  resume; every 2v1 map fits (2<=9, 1<=10, 7<=16). Flattened the unpacked dataset's redundant
  double-nested top folder.

## finish_trade_v4 reward

- Added `finish_trade_v4` reward (registry only; v1/v2/v3 untouched), iterating on v3 to reduce two failure modes: post-contact timeout/disengagement and high-enemy-EHP all-allies-dead wipeouts.
- Stall penalty is now gated on **first contact** (`has_dealt_damage_before`) so pre-contact positioning is never punished, and is scaled by remaining enemy EHP (`0.5 + 0.5*enemy_ehp_frac`).
- Timeout penalty split into `timeout_base` + `timeout_enemy` + `timeout_alive` (firmer than v3); all-dead penalty split into `all_dead_base` + `all_dead_enemy` (large remaining-enemy term so healthy-enemy wipeouts hurt most). Win reward kept modest (`w_win_ally_ehp 0.50`) so low-ally-EHP wins aren't discouraged.
- New per-episode W&B logs: `episode/reward_{timeout_base,all_dead_base,all_dead_enemy}` and `episode/has_dealt_damage_before` (env `canonical_terms` + trainer `EPISODE_REWARD_LOG_MAP` extended; existing v1-v3 terms unchanged).
- New config `configs/r2_2100_finish_trade_v4.yaml` (reward-only change vs v3); tests in `tests/test_finish_trade_v4.py`.
