# Option-Critic v5 Stability Hotfix

This is a fail-closed post-install hotfix for a repository that already has the
Option-Critic v4 P1-final package installed.

It does **not** claim that RL performance must improve monotonically. It fixes
specific implementation/design weaknesses found in the v4 source and adds
regression tests for the corrected contracts.

## Why v4 could peak at 200k and regress afterward

The v4 schedule crossed its largest semantic changes immediately after the
observed peak:

- per-state source routing became full temporal commitment by 250k;
- learned termination became fully active by 400k;
- eight option slots were trained from only two competent source tactics;
- the source action trust region used reverse KL;
- source-preservation losses were hinge-only and imagination-only;
- trust diagnostics subsampled only a small set of states.

The low-level call-and-return bootstrap, lambda-return indexing, duration state
machine, boundary-only manager credit, and forced-termination masks remain in
place. The stability failure was primarily in how quickly and how far the
learned hierarchy was allowed to depart from the source policy.

## v5 changes

- exactly two options, one for each Tactical Mixture v1.2 source mode;
- exact step-zero manager and worker migration;
- one-step source routing is retained through 100k;
- reactive reselection never falls below 0.25;
- maximum option duration reduced from 20 to 8;
- worker PG ramps from 20k to 150k;
- manager PG ramps from 100k to 500k;
- commitment ramps from 100k to 600k;
- learned termination waits until 350k and ramps through 800k;
- termination cap remains fixed at 0.30;
- termination task loss scale reduced to 0.02;
- source action preservation uses forward KL, `KL(source || live)`;
- source action and manager distillation remain active inside the hinge region;
- source trust is evaluated on both real replay posterior states and imagined
  states;
- trust-region state coverage rises to 2048 states per evaluation;
- the JEPA world model remains exactly frozen;
- imagination horizon varies from 7–10 initially to 9–12 by 600k;
- validation runs at startup and every 200k;
- the new phase is a fresh 1M-step run from the original v1.2 best checkpoint;
- the combined RL→Exp45 pipeline records failures and proceeds to the forecast
  stage by default.

## Validation boundary

The package validation covers Python/shell syntax, 59 mathematical,
state-machine, migration, gradient-routing, trust-region, and integration tests,
a fixture dry-run/full installation, generated-config resolution, rollback, and
failure-continuation behavior.

A complete CUDA/SMACLite learner run cannot be executed in the packaging
environment. The included real-repository static audit is mandatory before
launch.
