# Option-Critic Hierarchy v2 for JEPA R2-Dreamer

This bundle installs a call-and-return Option-Critic hierarchy on top of the
currently installed **Tactical Mixture v1.2** code path.

The hierarchy is a replacement for the ordinary Dreamer behaviour objective,
not a second competing actor-critic:

- the original JEPA world model, reward, continuation, availability, and alive
  heads remain;
- the inherited primitive actor becomes a frozen worker backbone;
- a manager selects one of eight option slots only at episode start or an
  option boundary;
- an option-conditioned worker residual selects primitive joint actions;
- a learned termination head decides whether the current option should end;
- an age-conditioned option critic evaluates `(belief, option, age)`;
- the old base-actor policy/value/replay-value imagination losses are removed
  from the optimisation objective whenever the hierarchy is enabled.

## Experiment configuration

- Source: `Tactical Mixture v1.2` best validation checkpoint (`0.375` macro win)
- Option capacity: 8
- Minimum duration: 3 primitive actions
- Maximum duration: 20 primitive actions
- Learned termination warm-up: fixed 0.10 hazard through 100k steps
- Learned termination ramp: 100k to 300k steps
- Worker residual maximum scale: 0.25
- Base actor: frozen
- JEPA feature adapter: frozen
- Adaptive map priority: disabled
- Sequence PER: disabled
- Replay: fresh, uniform, run-local memmap
- Imagination horizon: unchanged at 5
- New training phase: 2,000,000 environment steps
- Validation: every 200,000 steps; disabled at startup

Eight is a capacity, not a forced effective option count. The manager has a
small exploration floor and collapse-only regularisation; it is not trained to
use every option equally.

## Critical safeguards

### Correct replay alignment

Dreamer's replay transition stores the posterior observation at `h_t`, while
`agent.act()` also returns the call-and-return state carried to `h_{t+1}`.
The bundle stores **both**:

- the option state entering the decision at `h_t`;
- the selected option/action age and carry state after the decision.

Imagination starts from the pre-decision fields. Starting from carry age would
decide termination twice at the same posterior state and is explicitly tested.

### No competing behaviour objectives

The ordinary Dreamer actor, imagined value, and replay-value losses are removed
from the base `_cal_grad_jepa` loss dictionary in hierarchy mode. Model/head
losses remain. The hierarchical worker, manager, option critic, termination,
and hierarchy-value losses are the sole behaviour actor-critic objective.

### Policy-collapse controls

- Residuals are centered across all options, so a residual shared by every
  option cancels exactly.
- Same-state, post-mask Jensen-Shannon divergence has a small hinge floor.
- Only a rotating subset of pairs is evaluated per update, with a checkpointed
  call counter so all pairs continue rotating after replay reaches capacity.
- The manager uses exploration unimix and collapse-only guards, not uniform
  usage forcing.
- New options receive near-neutral, non-identical embeddings.
- The inherited v1.2 selector final layer is temperature-softened during
  migration so options 2--7 are not starved immediately.

### Learned-termination controls

- No termination is allowed before age 3.
- Termination is forced at age 20, but the manager may reselect the same option.
- The head uses a fixed 0.10 hazard through 100k steps.
- Termination policy gradients **and termination regularisers** are exactly
  disabled during warm-up.
- Learned termination is blended in through 300k and capped during the ramp.
- Termination uses slow option-value comparisons, a continuation margin,
  clipped advantages, a small Bernoulli unimix, entropy, and mean-probability
  collapse guards.
- Forced continuation and forced maximum termination receive no learned
  termination gradient.

### Environment/world-model contracts

- The option ID is never sent to JEPA.
- JEPA continues to receive only the primitive joint action.
- Existing primitive action masks are applied after option residuals.
- The post-mask invalid-action rate must remain zero.

## Files replaced

The installer backs up and replaces only:

- `external/r2dreamer/dreamer.py`
- `external/r2dreamer/trainer.py`
- `scripts/train_r2dreamer_smaclite_multimap.py`
- `src/smacdreamer/validation_trainer.py`

## Files added

- `external/r2dreamer/hierarchical_options.py`
- `external/r2dreamer/option_critic.py`
- `external/r2dreamer/hierarchical_dreamer.py`
- `configs/r2_2100_jepa_option_critic_8_v2.yaml`
- `scripts/audit_option_critic_hierarchy.py`
- `scripts/static_audit_option_critic_hierarchy.sh`
- `scripts/assert_option_critic_metrics.py`
- `scripts/run_option_critic_2m.sh`
- `tests/test_hierarchical_options.py`
- `tests/test_option_critic_math.py`
- `tests/test_hierarchical_auxiliary.py`
- `tests/test_hierarchy_migration.py`

## Installation

Place `option_critic_hierarchy_bundle.zip` in:

```text
/home/jovyan/workspace/dreamer/combined-upload
```

### 1. Stop any running trainer

```bash
pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
```

Stop the v1.2 process before launching Option-Critic. Keep its run directory and
best checkpoint.

### 2. Extract and define paths

```bash
cd /home/jovyan/workspace/dreamer/combined-upload
unzip option_critic_hierarchy_bundle.zip

export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export BUNDLE="$ROOT/option_critic_hierarchy_bundle"

export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"
export SOURCE_CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"
```

Verify the source:

```bash
readlink -f "$SOURCE_CHECKPOINT"

"$PY" - "$SOURCE_CHECKPOINT" <<'PY'
import sys, torch
p=sys.argv[1]
c=torch.load(p,map_location='cpu',weights_only=False)
print('step:',c.get('step'))
print('macro win:',c.get('val_macro_win_rate'))
print('original return:',c.get('val_macro_original_return'))
print('tactical metadata:',c.get('tactical_mixture_metadata'))
assert float(c.get('val_macro_win_rate',-1)) >= 0.3749
assert (c.get('tactical_mixture_metadata') or {}).get('architecture') == 'tactical_mixture_v1_2'
assert int((c.get('tactical_mixture_metadata') or {}).get('num_tactics',-1)) == 2
assert not any(k.startswith('hierarchical_options.') for k in c['agent_state_dict'])
print('[OK] v1.2 best checkpoint selected')
PY
```

### 3. Preserve the complete current source and checkpoint

```bash
"$BUNDLE/preserve_before_option_critic.sh" \
  "$REPO" \
  "$SOURCE_CHECKPOINT"

export PRESERVE_DIR="$(
  ls -dt "$ROOT"/preserve_before_option_critic_* | head -1
)"

echo "$PRESERVE_DIR"
sha256sum "$SOURCE_CHECKPOINT" "$PRESERVE_DIR/source_checkpoint.pt"
```

The two checkpoint hashes must match.

### 4. Fail-closed dry-run

```bash
"$PY" "$BUNDLE/install_option_critic_hierarchy.py" \
  --repo "$REPO" \
  --source-config configs/r2_2100_jepa_tactical_mixture_v1_2.yaml \
  --dry-run
```

Required ending:

```text
[OK] Option-Critic dry-run matched v1.2 source, parsed all patched ASTs, and resolved the output config
```

### 5. Install

```bash
"$PY" "$BUNDLE/install_option_critic_hierarchy.py" \
  --repo "$REPO" \
  --source-config configs/r2_2100_jepa_tactical_mixture_v1_2.yaml

export OPTION_CRITIC_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_backup_* | head -1
)"

echo "$OPTION_CRITIC_BACKUP"
cat "$OPTION_CRITIC_BACKUP/option_critic_backup_manifest.json"
```

### 6. Run the complete fast audit

```bash
cd "$REPO"

REPO="$REPO" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_option_critic_8_v2.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
bash scripts/static_audit_option_critic_hierarchy.sh
```

Required ending:

```text
38 passed
[OK] Option-Critic hierarchy source/config audit passed
[OK] Option-Critic hierarchy static audit passed
```

No long environment smoke is required.

### 7. Launch the new 2M phase

```bash
tmux new -s r2_option_critic
```

Inside tmux:

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"

source "$ROOT/.venv/bin/activate"
cd "$REPO"

ROOT="$ROOT" \
REPO="$REPO" \
PY="$PY" \
TACTICAL_V12_RUN="$TACTICAL_V12_RUN" \
FINAL_STEP=2000000 \
bash scripts/run_option_critic_2m.sh
```

Detach with `Ctrl-b`, then `d`.

The launcher refuses to:

- use anything except the v1.2 best checkpoint;
- use a source checkpoint below 37.5% macro win rate;
- reuse a non-empty run directory;
- run while another multimap trainer is active;
- resume a checkpoint that already contains hierarchy parameters.

## Monitoring

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export OPTION_RUN="$(cat "$ROOT/CURRENT_OPTION_CRITIC_RUN.txt")"

echo "$OPTION_RUN"
tail -f "$OPTION_RUN/train.log"
```

Once learner metrics appear:

```bash
"$PY" "$REPO/scripts/assert_option_critic_metrics.py" "$OPTION_RUN"
```

Important panels:

```text
train/option/legacy_behavior_losses_disabled
train/option/real_boundary_rate
train/option/real_eligible_termination_rate
train/option/real_mean_action_age
train/option/eligible_learned_beta_mean
train/option/termination_blend
train/option/effective_count
train/option/manager_mutual_information_normalized
train/option/action_js_mean
train/option/duplicate_pair_fraction
train/option/base_kl_mean
train/option/residual_to_base_ratio
train/option/action_flip_rate
train/imag_post_mask_invalid_sample_rate
train/real_post_mask_invalid_sample_rate
val/macro_win_rate
```

Expected hard invariants:

```text
legacy_behavior_losses_disabled = 1
post-mask invalid rates = 0
termination_blend = 0 through 100k, then rises to 1 by 300k
option age never exceeds 20
termination cannot occur before age 3
```

Eight option slots need not all remain active. An effective count around 3--6
can be healthy if behaviours are distinct and validation improves.

## Restore

To undo only this installation:

```bash
PY="$PY" bash "$BUNDLE/restore_option_critic_backup.sh" \
  "$OPTION_CRITIC_BACKUP" \
  "$REPO"
```

The larger source snapshot remains at `$PRESERVE_DIR`.

## Known limitation

The worker, manager, and termination objectives use short JEPA imagined
rollouts initialized from replay-aligned option state. Options may last longer
than the five-step imagination window because option identity and age are
bootstrapped through the value functions. The bundle does not add a separate
trajectory discriminator; option diversity is protected through centered
residuals, same-state post-mask JS floors, exploration, and collapse guards.

A complete CUDA/SMACLite learner run cannot be executed in the packaging
sandbox. The fail-closed dry-run and static audit therefore execute against the
user's actual installed v1.2 source before the 2M launch.
