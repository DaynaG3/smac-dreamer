# Tactical Mixture Actor v1 integration bundle

Target local tree:

```text
~/workspace/dreamer/combined-upload/smac-dreamer
```

Expected starting point:

- branch/worktree already contains the unified adaptive-priority integration;
- `dreamer.py` and the multimap runner contain `UNIFIED_PRIORITY_V1` markers;
- the old best R2 checkpoint is weights-only;
- the first tactical experiment starts a fresh 2,000,000-step phase.

## What this installs

The actor becomes a residual mixture policy:

```text
JEPA feature
  ├─ existing actor ────────────────> base primitive logits
  ├─ tactic selector ───────────────> one shared z in {0,1,2,3}
  └─ tactic residual(feature, z) ───> residual primitive logits

final primitive logits = base logits + residual logits
```

The existing real and imagined primitive-action masks are applied **after**
the residual is added. The tactic is never sent into JEPA and never changes the
environment action shape.

Version 1 selects a new tactic every primitive step (`duration: 1`). It does
not add persistent manager state yet.

## Isolation from adaptive priority

The generated tactical config disables:

```text
adaptive_priority.enabled
adaptive_priority.map.enabled
adaptive_priority.sequence.enabled
```

The installer also restores the original `Buffer`/`SliceSampler` path whenever
all adaptive-priority switches are disabled. This matters because merely
setting the candidate-PER flag to false would still pass through the adaptive
buffer implementation and would not be a clean replay ablation.

The adaptive implementation remains installed for later runs; it is only
disabled in the tactical experiment config.

## Legacy checkpoint migration

The old best checkpoint has no tactical parameters. Loading is fail-closed:

- only `tactical_policy.*` and `_frozen_tactical_policy.*` may be missing;
- any missing existing actor/value/JEPA key fails;
- unexpected keys fail;
- the residual output and selector output are zero-initialized;
- the old primitive-action logits are therefore exactly preserved initially;
- a fresh optimizer, scheduler, return EMA, replay and RNG phase are used.

A tactical checkpoint includes architecture metadata and must reload strictly.

## Files replaced

```text
external/r2dreamer/dreamer.py
scripts/train_r2dreamer_smaclite_multimap.py
```

Both are copied to a timestamped installer backup before replacement.

## Files added

```text
external/r2dreamer/tactical_policy.py
scripts/preflight_tactical_mixture.py
scripts/static_audit_tactical_mixture.sh
scripts/run_tactical_mixture_2m.sh
scripts/assert_tactical_metrics.py
tests/test_tactical_policy.py
configs/r2_2100_jepa_tactical_mixture.yaml
```

## Safety properties checked by the bundle

- tactical module is registered before optimizer construction;
- frozen tactical copy exists and shares the current live parameter storage;
- zero residual preserves base logits for every tactic;
- tactic is sampled in both real acting and JEPA imagination;
- JEPA `img_step` still receives only primitive actions;
- tactic and primitive log-probabilities use the same imagined advantage;
- tactical losses inherit terminal/start weighting and any sequence IS weight;
- action-mask construction is untouched;
- duration values other than 1 are rejected;
- adaptive priority is disabled in the generated first-ablation config;
- startup validation is disabled and held-out validation remains every 200k.

## Important scientific limitation

The anti-collapse balance and effect terms are auxiliary heuristics. They make
collapse observable and discourage it, but they do not guarantee that useful
human-interpretable tactics emerge. The relevant W&B metrics are:

```text
train/tactic/entropy
train/tactic/effective_count
train/tactic/usage_0 ... usage_3
train/tactic/effect_js
train/tactic/residual_rms
train/tactic/residual_to_base_ratio
```

At exact zero residual, the JS effect objective itself has zero first-order
gradient. Initial tactic differentiation therefore begins through the sampled
tactic policy-gradient and residual output-layer gradients; the effect hinge
acts after modes begin to differ.

## Installation outline

1. Preserve the current working tree and checkpoint.
2. Run the installer in `--dry-run` mode.
3. Install and inspect the Git diff.
4. Run the static audit and six fast module tests.
5. Launch the full 2M run directly; no long environment smoke is required.

See the response accompanying this ZIP for exact commands.
