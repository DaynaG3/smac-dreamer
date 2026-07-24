# Option-Critic P1-Final 1M + Exp45 Pipeline Bundle

This is a post-install patch for the already integrated
`dreamer_option_critic_v3_p0p1` repository. It must not be applied to the old
24-test or 38-test source directly.

## Corrections

1. **Termination cap invariance**
   - Replaces cap-dependent interval rescaling with a differentiable smooth
     approximation to `min(beta, cap)`.
   - Low termination probabilities no longer rise merely because the cap rises.
   - The termination head is initialized by numerical inversion so the executed
     probability starts at exactly the configured fixed hazard.

2. **Source manager-routing trust region**
   - Aggregates option probabilities by Tactical v1.2 source mode:
     even slots are source mode 0 and odd slots are source mode 1.
   - Constrains mean and top-tail group KL against the frozen source selector.
   - Adds high-confidence source-group preservation and flip diagnostics.
   - Duplicate slots remain free to specialize within their original source
     group.

3. **Staged 1M schedule**
   - Worker PG: 20k to 100k.
   - Manager PG and temporal commitment: 100k to 250k.
   - Learned termination: 250k to 400k.
   - Smooth cap relaxation: 400k to 600k.
   - Full hierarchy: 400k to 1M.

4. **Longer variable imagination**
   - Cycles through horizons 7-10 initially.
   - Ramps to horizons 12-15 by 400k.

5. **Evaluation and execution**
   - Startup validation remains enabled.
   - RL validation runs every 200k steps.
   - The RL phase is 1,000,000 fresh environment steps from the original
     Tactical Mixture v1.2 best checkpoint.

6. **Resilient sequential pipeline**
   - Runs the audited 1M RL phase, then Exp45 forecast-JEPA training and both
     evaluations.
   - Starts itself in detached tmux by default.
   - A failed RL stage does not prevent the forecast stage from being attempted.
   - Forecast stage failures are recorded rather than killing the master script.
   - If Exp45 training fails but leaves a checkpoint, evaluation is attempted by
     default.

## Architecture

```text
dreamer_option_critic_v4_p1_final
```

## Validation boundary

The package was validated with CPU mathematical/state-machine tests, installer
fixtures, source/config audits, shell syntax checks, rollback checks, and
failure-continuation tests. The packaging environment cannot run the actual
CUDA/SMACLite or Exp45 training jobs; the real-repository audit must pass before
launch.
