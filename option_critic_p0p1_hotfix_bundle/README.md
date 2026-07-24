# Option-Critic P0/P1 Hotfix v3

This is a **post-install hotfix** for a repository that already contains the corrected 38-test Option-Critic v2 integration. It does not reinstall Tactical Mixture v1.2 or the original hierarchy bundle.

The new architecture identifier is:

```text
dreamer_option_critic_v3_p0p1
```

## What is fixed

### P0. Episode-start credit

Imagination no longer multiplies episode-start trajectories by zero. `is_first` states are valid manager boundaries; only `is_last` starts are excluded.

### P0. Trajectory-preserving migration

The two Tactical Mixture v1.2 modes are expanded into four exact copies each. At step zero:

- manager temperature is exactly `1.0`;
- manager unimix is `0.0`;
- worker residual scale is already the inherited `0.25`;
- minimum option duration is `1`;
- eligible states reselect with probability `1.0` through 100k.

Thus the grouped manager probabilities and primitive policy match the source policy before temporal commitment is introduced.

### P0. Collection-policy corruption

The old `0.25` migration temperature and `0.20` initial manager unimix are removed. Manager exploration increases only from `0.00` to `0.02`.

### P0. Representation drift under a frozen source actor

The live JEPA world-model gradient scale is fixed at zero for this conservative 2M phase. A pre-optimizer guard clears world-model gradients so optimizer momentum and decoupled weight decay cannot move those parameters.

### P1. Horizon mismatch

Hierarchy imagination uses a checkpointed variable horizon. It cycles through `5–8` initially and gradually moves to `7–10` by 500k. The base Dreamer horizon remains unchanged.

### P1. Task-agnostic option forcing

Manager collapse, manager mutual information, action-JS diversity, residual-cosine, and termination-collapse penalties are disabled. Diagnostics remain logged.

### P1. Termination dead-gradient cap

The executed learned termination probability uses a smooth bounded sigmoid rather than `clamp(max=cap)`. The task loss differentiates through the actual executed probability.

### P1. Rare-state policy damage

The trust region is measured against a permanent copy of the **full migrated Tactical Mixture policy**, not just the primitive actor. It includes:

- mean masked KL;
- top-tail masked KL;
- high-confidence source-action preservation;
- action-flip diagnostics.

### P1. Dense-return versus win-rate drift

The training policy is anchored to the source policy, startup validation is enabled, validation runs every 100k, and `check_option_critic_win_guard.py` compares validation macro win rate with the source checkpoint. Checkpoint selection remains based on macro validation win rate.

## Conservative training schedule

```text
0–100k:
  exact per-state source routing
  manager task PG off
  worker task PG off
  learned termination off
  JEPA world model frozen

100k–300k:
  temporal commitment ramps in
  manager and worker PG ramp in
  learned termination ramps in

After 300k:
  full call-and-return hierarchy
  manager unimix at most 0.02
  JEPA world model remains frozen
```

The hotfix intentionally starts a **fresh phase from the Tactical Mixture v1.2 best checkpoint**. Do not resume the degrading Option-Critic v2 run.
