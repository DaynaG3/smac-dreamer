# Option-Critic v9: Anchor-Safe Interruptible Eight-Slot H=15 Run

## Supersession

**Do not install v7 or v8.** This v9 bundle replaces both and installs only on
top of the currently integrated v6 repository state.

A second audit of v8 found a critical online-policy bug: its
`_frozen_hierarchical_options` object was a true, stale snapshot. R2-Dreamer's
"frozen" acting/imagination modules are no-gradient views of the current online
parameters; their parameter storage must follow live optimizer updates. Under
v8, the live child and slot-manager networks could learn while real and imagined
collection continued to execute their step-zero copies. That made the actor loss
off-policy with respect to the executed hierarchy and invalidated the proposed
experiment.

v9 separates the two concepts correctly:

- `_frozen_hierarchical_options`: a no-gradient online view whose parameter
  storage tracks the live controller, used for acting and imagination;
- `_source_hierarchical_options`: a permanent non-aliased Tactical-v1.2 copy,
  used only as the source-policy safety reference.

## Why this is the controlled experiment justified by the past runs

The historical curves repeatedly improved while the controller remained
reactive and then regressed as long commitment, learned termination, or unsafe
child duplication became influential. The best interpretable next test is
therefore not unrestricted Option-Critic. It is an interruptible short-duration
SMDP that adds within-tactic capacity while retaining the source selector as a
hard safety router.

### Eight identities

```text
source group 0: options 0, 2, 4, 6
source group 1: options 1, 3, 5, 7
```

- Options 0 and 1 are immutable exact Tactical Mixture v1.2 anchors.
- Options 2–7 are bounded child experts.
- Every child has an exactly zero output layer at migration, so all eight
  identities initially execute their source parent exactly.
- Hidden layers are nonzero, allowing state-dependent learning after the output
  layer receives its first update.
- Child action deltas are bounded at `0.10`.

### Anchor-safe identity exploration

Within each source group, the conditional slot distribution has a fixed 40%
source-anchor floor plus a 1% within-group unimix. At the zero-logit
initialization this is approximately:

```text
anchor:      0.547
child 1:     0.151
child 2:     0.151
child 3:     0.151
```

This keeps every child active from step zero without the unsafe v8 prior where
75% of identity collection came from unproven children. A genuinely useful
child can still become deterministic argmax; the floor does not freeze routing.

### Exact interruptible execution

```text
minimum duration: 1 primitive action
maximum duration: 4 primitive actions
learned termination: disabled
source-group interruption: evaluated at every state
```

A carried option returns control when either:

1. its source group differs from the frozen Tactical selector's current group;
2. it has already executed four primitive actions.

The identical interruption event is used by real collection, imagined
collection, Option-Critic bootstrapping, and group-restricted switch values.
There is no stochastic/deterministic learned-termination mismatch in this run.

### Online, source, and slow objects

- The online no-grad hierarchy view shares live parameter storage by design, so
  acting and imagination immediately observe optimizer updates.
- The permanent Tactical source reference never shares storage.
- The slow option critic is an actual EMA target and does not share storage.
- World-model and source-anchor optimizer gradients are cleared during the
  controlled frozen phase, including protection against momentum and AdamW
  weight decay.

### Learning schedule

```text
0–20k:       option critic and hierarchy value warm-up only
20k–150k:    child worker policy gradient ramps to full
100k–300k:   within-group slot-manager policy gradient ramps to full
all steps:   source-group selector and source anchors frozen
all steps:   learned termination disabled
all steps:   imagination horizon exactly 15
```

Worker trust-region objectives are detached entirely while worker PG is zero.
This prevents mathematically zero but numerically nonzero KL gradients from
moving an exact-zero child output during the critic-only warm-up.

While children are still behaviorally close to their source anchor, an option
critic consistency loss pulls child age-zero values toward the detached anchor
value. It decays as worker PG activates and becomes zero once worker learning is
fully active. This prevents the slot manager from exploiting random
identity-specific critic noise before children have causal action differences.

### Source-policy safety

Safety losses use both real replay posterior states and imagined states:

- forward KL `KL(source || live)`;
- high-confidence source-action preservation;
- residual magnitude guard.

The forward direction penalizes dropping probability from source-supported
actions. Disabled task-independent diversity objectives are not attached to the
autograd graph.

## Comparison settings

```text
new environment steps:       800,000
imagination horizon:         exactly 15
validation:                  startup and every 200,000 steps
source checkpoint:           Tactical Mixture v1.2 best validation checkpoint
expected source SHA-256:     74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e
source training regime:      preserved except explicit hierarchy/run paths
replay:                      fresh, run-local
```

The launcher resets the new phase's environment step to zero while loading the
source weights. It refuses any other checkpoint hash.

## Forecast-first orchestration

The master pipeline runs the Exp45 forecast JEPA install/audit/train/ordinary
evaluation/hidden evaluation sequence before the RL phase. The Exp45 installer
receives the actual workspace `ROOT` explicitly; this is required because the
container account's `$HOME` is not the `/home/jovyan` workspace.

If a forecast failure leaves a live training child process, RL is safely skipped
rather than started concurrently on the same GPU. Otherwise, failures are
recorded and later safe stages continue according to `CONTINUE_ON_FAILURE` and
`STRICT_EXIT`.

The forecast checkpoint is not injected into the inherited Tactical actor. The
two experiments run sequentially but remain checkpoint-independent, avoiding an
uncontrolled actor-facing representation change.

## Validation boundary

The package tests mathematical/state-machine/migration/gradient/installer and
pipeline contracts. It cannot prove that stochastic RL will improve win rate,
and no finite test suite can prove the absence of every runtime defect. The
mandatory real-repository audit and startup/200k validations are the empirical
boundary.
