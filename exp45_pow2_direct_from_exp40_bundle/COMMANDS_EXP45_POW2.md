# Exp45 Pow2 Commands

## Install

```bash
cd ~/workspace/dreamer/combined-upload
unzip exp45_pow2_direct_from_exp40_bundle.zip -d exp45_pow2_bundle
cd exp45_pow2_bundle
ROOT=~/workspace/dreamer/combined-upload ./install_exp45_pow2_direct.sh
```

## Static audit

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm
./scripts/static_audit_exp45_pow2.sh
```

## CUDA smoke

```bash
cd ~/workspace/dreamer/combined-upload/smac-jepa-wm
source ../.venv/bin/activate

EXP40_CHECKPOINT=runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt \
SMOKE_DEVICE=cuda \
./scripts/smoke_exp45_pow2_direct.sh
```

## Full train from Exp40

```bash
tmux new -s exp45_pow2

cd ~/workspace/dreamer/combined-upload/smac-jepa-wm
source ../.venv/bin/activate

EXP40_CHECKPOINT=runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt \
OUT_DIR=runs/rnn_seqmem_exp45_pow2_direct_$(date +%Y%m%d_%H%M%S) \
WANDB_NAME=exp45-pow2-direct-1-2-4-8-16 \
./scripts/run_exp45_pow2_direct_train.sh
```

## Resume Exp45

```bash
RESUME=runs/<exp45_run>/checkpoint.pt \
OUT_DIR=runs/<exp45_run> \
./scripts/run_exp45_pow2_direct_train.sh
```

## Ordinary evaluation

```bash
CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
OUT_DIR=eval_outputs/<exp45_run>/ordinary \
./scripts/eval_exp45_pow2_ordinary.sh
```

## Hidden evaluation

Ordinary evaluation must finish first because it creates the hidden-compatible
meaningful-feature probe.

```bash
CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
ORDINARY_OUT_DIR=eval_outputs/<exp45_run>/ordinary \
OUT_DIR=eval_outputs/<exp45_run>/hidden \
./scripts/eval_exp45_pow2_hidden.sh
```

## Run both evals

```bash
CHECKPOINT=runs/<exp45_run>/checkpoint.pt \
./scripts/eval_exp45_pow2_all.sh
```

## Optional evaluator path overrides

```bash
export ORDINARY_EVAL=/path/eval_rnn_seqmem_dreamer_probe_r2aware.py
export ANCHORED_EVAL=/path/eval_jepa_exp31_exp33_anchored.py
export HIDDEN_EVAL=/path/eval_jepa_hidden_belief_exp31_exp33.py
```
