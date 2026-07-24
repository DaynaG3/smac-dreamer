# R2-Dreamer × Exp33 JEPA integration fix

This patch fixes two serious integration issues:

1. `FrozenJEPAWorldModel.get_feat()` no longer runs the trainable `feature_adapter` under `torch.no_grad()`.
2. `JEPAFeatureAdapter` is replaced with a slot-preserving adapter instead of masked mean pooling.

It also keeps the earlier belief-mask contract:

```text
belief exposure mask = current visible/exposed OR (anchored-memory seen AND structural slot)
```

## Files

Copy these into the repo:

```text
src/smacdreamer/jepa/world_model.py
src/smacdreamer/jepa/feature_adapter.py
validate_integration_static.py
validate_adapter_grad.py
inspect_r2_run_config.py
make_tmp_eval10k_config.py
```

## Install patch from repo root

```bash
cd ~/workspace/dreamer/combined-upload/smac-dreamer

mkdir -p patch_backups/r2_jepa_integration_$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=$(ls -td patch_backups/r2_jepa_integration_* | head -1)

cp src/smacdreamer/jepa/world_model.py "$BACKUP_DIR/world_model.py.before"
cp src/smacdreamer/jepa/feature_adapter.py "$BACKUP_DIR/feature_adapter.py.before"

cp r2_jepa_integration_fix/world_model.py src/smacdreamer/jepa/world_model.py
cp r2_jepa_integration_fix/feature_adapter.py src/smacdreamer/jepa/feature_adapter.py
cp r2_jepa_integration_fix/validate_integration_static.py .
cp r2_jepa_integration_fix/validate_adapter_grad.py .
cp r2_jepa_integration_fix/inspect_r2_run_config.py .
cp r2_jepa_integration_fix/make_tmp_eval10k_config.py .
```

## Validate files and gradients

```bash
python -m py_compile src/smacdreamer/jepa/world_model.py
python -m py_compile src/smacdreamer/jepa/feature_adapter.py
python -m py_compile validate_integration_static.py
python -m py_compile validate_adapter_grad.py
python -m py_compile inspect_r2_run_config.py
python -m py_compile make_tmp_eval10k_config.py

python validate_integration_static.py
python validate_adapter_grad.py
python inspect_r2_run_config.py \
  --config configs/r2_2100_jepa_local.yaml \
  --jepa-checkpoint checkpoints/jepa/model.pt
```

## Make an explicit debug config

```bash
python make_tmp_eval10k_config.py
cat configs/tmp_r2_2100_jepa_local_modelpt_eval10k.yaml | grep -nE "checkpoint|every|run_at_start|steps|logdir|reward"
```

## Run the integration smoke

```bash
export RUN="logs/r2dreamer/debug_exp33_jepa_beliefmask_slotadapter_grad_eval10k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/tmp_r2_2100_jepa_local_modelpt_eval10k.yaml \
  --jepa-checkpoint checkpoints/jepa/model.pt \
  --steps 50000 \
  --logdir "$RUN" \
  --wandb-mode disabled \
  2>&1 | tee "$RUN/train.log"
```

## Inspect the run

```bash
python inspect_r2_run_config.py \
  --config configs/tmp_r2_2100_jepa_local_modelpt_eval10k.yaml \
  --jepa-checkpoint checkpoints/jepa/model.pt \
  --run "$RUN"

grep -Ei "jepa|checkpoint|adapter|validation|eval|win|reward|maskh0|imag_empty|feature_norm|nan|inf|traceback|error" \
  "$RUN/train.log" | tail -250
```

## Expected smoke success criteria

Do not require high win rate from this short smoke. First require:

```text
- config uses checkpoints/jepa/model.pt, not model_smoke.pt
- validation.every = 10000
- adapter static test passes
- adapter gradient test passes
- training starts without shape errors
- no NaN/Inf/traceback
```
