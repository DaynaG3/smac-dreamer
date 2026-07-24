# Exp45 — Exp40 + Dynamic Power-of-Two Direct LSTM Prediction

## Purpose

This experiment preserves the trusted Exp40 JEPA training objective and adds a
second, direct trajectory-prediction branch.

Exp40's recursive transition still trains and evaluates at horizon 5. The new
branch receives:

- the real current JEPA latent;
- Exp40's anchored, action-conditioned belief memory;
- the recorded future ally joint-action sequence.

It predicts future EMA target latents without feeding an intermediate predicted
world latent back into the same direct block.

The direct branch trains specialized jump heads at:

```text
1, 2, 4, 8, 16 steps
```

It also trains one shared readout at a rotating arbitrary horizon from 1 through
16. Thus the model supports two ways to obtain a non-power horizon such as 9:

```text
Exact direct:  one LSTM action-prefix pass to h=9, then the shared readout
Binary:        trained 8-step jump followed by a trained 1-step jump
```

The binary path has one predicted-state boundary instead of eight boundaries in
nine repeated one-step transitions. Horizons above 16 can reuse the largest
block, for example `25=16+8+1` and `32=16+16`; only the supplied action-sequence
length limits composition.

## Relation to the supplied LSTM encoder-decoder diagram

The useful part of the friend's model is retained: initialize a decoder from
real history and decode the future from a causal sequence representation.

The past-observation LSTM encoder is not duplicated because Exp40 already has an
anchored recurrent belief memory. In this implementation:

```text
Exp40 real encoder + anchored belief memory
                  ↓
     memory-conditioned real start latent
                  ↓
 causal LSTMCell over real joint-action prefix
                  ↓
 horizon-specific power head or shared dynamic head
                  ↓
            future JEPA latent
```

Unlike a decoder driven only by indices `[1,2,...,H]`, every decoder step uses
the actual ordered joint action. A learned step embedding is added only to tell
the shared decoder how far it has advanced.

## Architecture

`PowerOfTwoDirectPredictor` contains:

- per-agent action embeddings;
- agent-identity embeddings, preserving joint-action ordering;
- a joint-action projection;
- a causal `LSTMCell` decoder;
- entity-slot embeddings;
- five specialized readout heads for 1/2/4/8/16;
- one shared readout trained across every horizon 1–16.

The five power predictors share the causal action decoder but have independent
readout heads. This is preferable to five independent LSTMs because it avoids
five copies learning incompatible action dynamics while retaining
horizon-specific capacity.

## Training objective

The total objective is:

```text
L = L_Exp40
  + lambda_power * L_direct_power
  + lambda_exact * L_dynamic_shared
  + lambda_comp * L_binary_composition
```

- `L_Exp40` is the installed Exp40 R2-offline loss, forced to recursive horizon 5.
- `L_direct_power` compares direct predictions at 1/2/4/8/16 against real EMA
  target latents.
- `L_dynamic_shared` supervises one rotating exact horizon per optimizer batch;
  over training all horizons 1–16 receive supervision.
- `L_binary_composition` trains identities such as `2≈1+1`, `4≈2+2`,
  `8≈4+4`, and `16≈8+8` against both the real future target and the direct
  larger jump.

The dataset segment is extended to horizon 16, while the wrapped Exp40 loss is
explicitly kept at horizon 5. This extension is required to expose the real
action and target sequences used by the direct branch.

## Important scientific caveats

1. The LSTM hidden state is recurrent, but no predicted world latent is consumed
   inside one direct block. Therefore it avoids world-state feedback error, not
   all forms of sequence-model error.
2. Binary composition still feeds a predicted latent between blocks. For `9=8+1`
   there is one such boundary rather than eight.
3. Future enemy behaviour remains uncertain because only the available joint
   action tensor can be conditioned on.
4. Direct future prediction does not automatically replace Dreamer's closed-loop
   imagination. Dreamer chooses later actions from imagined later states; the
   direct branch assumes an action sequence has already been supplied.
5. The direct branch is experimental and is deliberately marked
   `dreamer_compatible=False`. The sanitizer creates an Exp40-only view for
   standard evaluators and existing R2 loading; it does not pretend that R2 is
   using the new jump model.
6. Direct natural-hidden metrics are included. Controlled-occlusion evaluation
   remains the trusted recursive H=5 hidden-belief suite; the scripts do not
   falsely label it as controlled evaluation of the direct branch.

## Bundle contents

```text
smac_jepa/pow2_direct_predictor.py
smac_jepa/train_jepa_exp45_pow2_direct.py

tools/make_exp40_eval_checkpoint.py
tools/audit_exp45_pow2_checkpoint.py
tools/eval_pow2_direct.py
tools/eval_rnn_seqmem_dreamer_probe_r2aware_anchored.py

scripts/run_exp45_pow2_direct_train.sh
scripts/smoke_exp45_pow2_direct.sh
scripts/eval_exp45_pow2_ordinary.sh
scripts/eval_exp45_pow2_hidden.sh
scripts/eval_exp45_pow2_all.sh
scripts/static_audit_exp45_pow2.sh

tests/test_pow2_direct_predictor.py
tests/test_pow2_checkpoint_sanitizer.py
```

## Install

From the extracted bundle directory:

```bash
chmod +x install_exp45_pow2_direct.sh
ROOT=~/workspace/dreamer/combined-upload \
./install_exp45_pow2_direct.sh
```

The installer only adds/replaces Exp45-specific files. It does not reinstall or
rewrite the manually patched Exp40/Exp42–44 files. Any replaced Exp45 files are
backed up under a timestamped sibling directory.

## Static/unit audit

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm
./scripts/static_audit_exp45_pow2.sh
```

The bundled unit tests cover:

- power-horizon validation;
- causal no-future-action leakage at h=1;
- integer and one-hot action parity;
- output shape/finiteness;
- action sensitivity;
- `13=8+4+1` decomposition;
- reuse beyond the largest head with `25=16+8+1`;
- arbitrary shared-head output;
- checkpoint sanitization.

This does not replace a CUDA smoke against the live repository and dataset.

## CUDA smoke

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm

EXP40_CHECKPOINT=\
runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt \
SMOKE_DEVICE=cuda \
./scripts/smoke_exp45_pow2_direct.sh
```

The smoke intentionally still uses the complete 1/2/4/8/16 architecture so all
five heads are present in the produced checkpoint.

## Full training

```bash
tmux new -s exp45_pow2

cd ~/workspace/dreamer/combined-upload/smac-jepa-wm
source ../.venv/bin/activate

export EXP40_CHECKPOINT=\
runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt

OUT_DIR=runs/rnn_seqmem_exp45_pow2_direct_1_2_4_8_16_$(date +%Y%m%d_%H%M%S) \
WANDB_NAME=exp45-pow2-direct-1-2-4-8-16 \
./scripts/run_exp45_pow2_direct_train.sh
```

Detach from tmux with `Ctrl-b`, then `d`.

The default full run retains Exp40's five epochs and 50,000 sampled segments per
epoch. The direct branch defaults are:

```text
power direct weight       0.10
shared exact weight       0.10
composition weight        0.05
direct hidden dimension   384
warm-up                    2,000 optimizer batches
```

## Resume an Exp45 run

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm

RESUME=runs/<exp45_run>/checkpoint.pt \
OUT_DIR=runs/<exp45_run> \
./scripts/run_exp45_pow2_direct_train.sh
```

Do not supply `EXP40_CHECKPOINT` as an initialization source when `RESUME` is
set. The wrapper restores the direct-loss warm-up counter from the checkpoint's
global step.

## Ordinary evaluation

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm

CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
OUT_DIR=eval_outputs/<exp45_run>/ordinary \
./scripts/eval_exp45_pow2_ordinary.sh
```

This produces three complementary evaluations:

1. R2-aware corrected recursive H=5 metrics;
2. anchored ordinary H=5 metrics plus the exact probe format required by the
   targeted hidden-belief evaluator;
3. direct power, exact arbitrary-horizon, and binary-composed metrics.

The direct JSON reports `all`, `ally`, `enemy`, `visible`, and
`natural_hidden_enemy` subsets at each horizon. To explicitly test composition
beyond 16:

```bash
python tools/eval_pow2_direct.py \
  --manifest splits/r2_general_2100.json --split eval \
  --checkpoint runs/<exp45_run>/checkpoint.pt \
  --base-evaluator eval_rnn_seqmem_dreamer_probe_r2aware.py \
  --out eval_outputs/<exp45_run>/pow2_h25_h32.json \
  --power-horizons "1 2 4 8 16" \
  --binary-horizons "9 17 25 32" \
  --max-composed-horizon 64 --device cuda
```

## Hidden-belief evaluation

Run ordinary evaluation first so the meaningful-feature probe exists, then:

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm

CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
ORDINARY_OUT_DIR=eval_outputs/<exp45_run>/ordinary \
OUT_DIR=eval_outputs/<exp45_run>/hidden \
./scripts/eval_exp45_pow2_hidden.sh
```

This runs:

- the corrected natural-hidden and controlled-occlusion suite for Exp40's
  recursive H=5 branch;
- direct natural-hidden-enemy metrics at power, exact, and binary horizons.

## Run every evaluation

```bash
CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
./scripts/eval_exp45_pow2_all.sh
```

## Evaluator path overrides

When evaluator files are stored somewhere else:

```bash
export ORDINARY_EVAL=/absolute/path/eval_rnn_seqmem_dreamer_probe_r2aware.py
export ANCHORED_EVAL=/absolute/path/eval_jepa_exp31_exp33_anchored.py
export HIDDEN_EVAL=/absolute/path/eval_jepa_hidden_belief_exp31_exp33.py
```

## Reading the results

The most important comparison for compounding error is:

```text
recursive H=5 error
versus
one-pass direct-power H=4/H=8 error
versus
one-pass exact-shared H=5/H=9 error
versus
binary-composed H=5/H=9 error
```

Useful evidence for the hypothesis would be:

- similar direct and recursive error at h=1;
- a widening direct advantage as horizon grows;
- exact h=9 or binary `8+1` outperforming nine one-step transitions;
- no collapse in action-shuffle sensitivity;
- preserved or improved natural-hidden enemy prediction;
- no meaningful regression in the sanitized Exp40 recursive H=5 metrics.

A lower latent error alone is not enough if the action-conditioned diagnostics
show that the model ignores the supplied joint action sequence.
