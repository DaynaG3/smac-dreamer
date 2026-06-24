# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Training pipeline for **R2-Dreamer** (a decoder-free DreamerV3-family world-model agent) on the
**SMAClite** simulator, treating multi-agent combat as **single-agent centralised control**: one
Dreamer drives all allied units. Project code is in `src/smacdreamer/`, `scripts/`, `configs/`;
the upstream agent and simulator are vendored in `external/r2dreamer/` and `external/smaclite/`.

The full training-pipeline narrative (discovery → centralised structured-obs envs → masked
collection → CPU/memmap replay → decoder-free world model → imagination actor-critic → held-out
checkpoint selection) lives in [README.md](README.md) — read it before changing pipeline behaviour.

This project **fully migrated off JAX-DreamerV3**: there is no `external/dreamerv3`, and no live
`jax`/`embodied`/`elements` imports (only historical comments). Do not reintroduce them. The test
suite is deliberately JAX-free (`tests/conftest.py` only puts `src` and `external/smaclite` on
`sys.path`).

## Commands

Everything runs in the `smac-r2` conda env (Python 3.11), from the repo root.

```bash
# Tests (pytest). conftest.py wires sys.path; tests needing torch/smaclite/r2dreamer
# importorskip and skip cleanly when those aren't importable.
python -m pytest tests/ -q
python -m pytest tests/test_map_priority.py -v                       # one file
python -m pytest tests/test_map_priority.py::test_hard_score_components -v   # one test

# Train (one YAML drives everything). Set memory/headless env vars first (README Quick start).
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650.yaml

# Evaluate a checkpoint on a blind split (rebuilds the model from run_meta.json)
python scripts/evaluate_multimap.py --config configs/r2_650.yaml \
    --checkpoint logs/r2dreamer/<run>/best_val_macro_winrate.pt --split blind_iid

# Resume (warm-start weights+optimizer; see "Resume" gotcha below)
python scripts/train_r2dreamer_smaclite_multimap.py --config <cfg> \
    --resume logs/.../latest.pt --resume-mode full --logdir logs/.../<new_run>
```

There is no configured linter/formatter and no build step (pure-Python package). On Windows use the
`smac-r2` env's `python.exe`; the training path is normally Linux GPU (Kubeflow/A40).

## Architecture you can't see from one file

**Centralised control.** `SMACliteDreamerEnv` exposes the whole allied team as ONE agent:
factorised multi-one-hot actions (`A` agent slots × `C` actions, padded to `max_agents ×
max_actions`) and structured per-entity observations with fixed semantic positions across maps and
a global unit-type vocabulary. Action masking forces only valid actions (real masks while
collecting; predicted `avail_head`/`alive_head` masks inside imagination).

**The env runs in worker subprocesses, wrapped twice.** `ParallelEnv` (in `external/r2dreamer/
envs/parallel.py`) spawns `env_num` worker processes; each holds a `SMACliteR2DreamerAdapter`
(`src/smacdreamer/envs/r2dreamer_adapter.py`) wrapping a `SMACliteDreamerEnv`. Two consequences that
bite:
- **`gym.Wrapper` does NOT auto-forward arbitrary methods.** Any env method you call from the main
  process across the worker boundary (e.g. `set_map_hard_scores`) must be **explicitly forwarded in
  the adapter** (see how `get_debug_context` / `set_map_hard_scores` are written). Forgetting this
  kills every worker at the first call.
- **`ParallelEnv` discards the gym `info` dict on step.** The adapter merges every `log_*` info key
  into the returned `obs` so it reaches the trainer's `log_`-prefixed aggregation. New per-episode
  signals must be emitted as `log_*` keys to survive.

**SMAClite leaks native memory per reset.** Long runs OOM unless workers recycle
(`env_lifecycle.max_episodes_per_worker`, default 25). On shared/Kubeflow nodes, a worker dying with
`ConnectionResetError` / `Lost connection to worker` and **no Python traceback** = it was
SIGKILL'd (OOM or node pressure), and currently takes down the whole run.

**Adaptive hard-map curriculum** (`sampling_mode: prioritized_hard_maps`, `map_priority` config).
`MapPriorityTracker` (`src/smacdreamer/map_priority.py`) aggregates per-map difficulty from
**training** rollouts (keyed by `log_map_id`), and the trainer broadcasts hard scores to all workers
on a cadence; `ParallelEnv` caches them and re-pushes on recycle. Sampling is a mixture: 75% baseline
`shuffled_round_robin` coverage + 25% ∝ hard-score. **Validation is never prioritised** — it uses a
`fixed` sampler over held-out maps. See `docs/training/prioritized_hard_maps.md`.

**Checkpoint selection** is always **macro held-out win rate**, tie-broken by macro **original**
return — `is_validation_improvement` in `src/smacdreamer/evaluation.py`. Never select on shaped
return. Held-out eval uses the original `smaclite_default` reward regardless of training reward.

**Reward is swappable** via a name→callable registry (`src/smacdreamer/envs/reward_registry.py`).
Add new rewards there without editing existing ones; the env tracks the original return separately
for selection and logs each reward term under `log_reward_term_*`.

## Gotchas / conventions

- **Step counter comes from replay, and replay is not persisted.** The trainer derives its local
  step from `replay_buffer.count() * action_repeat`, so a `--resume` restarts the local step (and
  the validation/warm-up cadence) at 0 with warm weights. Use `--step-offset N` to keep the W&B
  `global_step` x-axis continuous (`global_step = local + offset`), and `--wandb-run-id <id>` to
  append to the same W&B run. `transfer_reward`/`weights_only` resume modes and any non-zero
  `step_offset` REQUIRE `--resume` (guarded in `checkpoint_transfer.validate_resume_args`).
- **`torch.load` needs `weights_only=False`** for our checkpoints — they carry numpy/torch RNG and
  optimizer state that PyTorch ≥2.6's safe loader rejects.
- **AMP must be bf16 on the production path** (`amp_dtype: bfloat16`, A40/A100/L4). fp16 overflows on
  the large structured observation; `cuda_preflight` fails loudly if the GPU lacks BF16. CPU runs
  disable autocast (fp32).
- **memmap replay refuses to overwrite.** A re-run into an existing logdir collides on
  `logs/.../replay/state.memmap`; use a fresh `--logdir` or delete the `replay/` subdir first.
- **Editing `external/`:** `external/r2dreamer/` (the agent + trainer + ParallelEnv) is editable when
  necessary, but keep changes minimal and guarded (e.g. tracker hooks are `getattr`-gated so the
  loop still runs without them). `external/smaclite/` is the upstream simulator — treat as
  read-only.
- **Config knob vs agent discount.** Reward-shaping γ comes from `cfg.gamma`; the agent's actual
  return discount is `1 - 1/config.horizon` (horizon hardcoded at 333 ≈ 0.997). They're meant to
  match but are set independently — keep them consistent if you touch either.
