# Frozen JEPA R2-Dreamer Integration

This branch adds a selectable world-model backend:

```yaml
world_model:
  backend: rssm
```

or:

```yaml
world_model:
  backend: jepa
  jepa:
    checkpoint: checkpoints/jepa/model.pt
    strict_checkpoint: true
    freeze_core: true
```

The RSSM backend remains the default. The JEPA backend is optional and requires
the local `smac-jepa-wm` package:

```bash
python -m pip install -e "<PATH_TO_SMAC_JEPA_REPO>"
python -m pip install -e .
```

## Architecture

JEPA mode loads a separately trained SMAC-JEPA checkpoint, freezes the JEPA core,
and trains a new `JEPAFeatureAdapter` plus the existing R2 downstream modules.

Frozen modules:

- `SMACJEPA.encoder`
- `SMACJEPA.predictor`
- `SMACJEPA.decoder`
- `SMACJEPA.presence_head`
- recurrent memory module

Trainable modules:

- `JEPAFeatureAdapter`
- reward head
- continuation head
- availability head
- alive-agent head
- actor
- critic

The slow critic remains managed by the existing target-update path.

## State

`stoch` is the current per-entity JEPA latent:

```text
[B, E, Z]
```

`deter` is a flat packed tensor containing:

```text
memory[B,E,M] | entity_mask[B,E] | slot_mask[B,E] | static_condition[B,S]
```

All slicing is centralized in `smacdreamer.jepa.state.pack_state` and
`unpack_state`.

## Observation Fields

When JEPA mode is selected, structured SMAClite observations additionally include:

- `jepa_entity`
- `jepa_entity_mask`
- `jepa_entity_slot_mask`
- `jepa_static_condition`

RSSM runs do not receive these fields, preserving existing RSSM encoder behavior.

## Checkpoint Contract

The source JEPA checkpoint must contain:

- `model_state`
- `memory_module_state`
- `metadata`
- `resolved_config` or `config`

The loader validates metadata against the live R2 environment and fails on
mismatches. It never loads the JEPA optimizer or scaler, and it never falls back
to RSSM.

The specified local JEPA checkout currently lacks `smac_jepa.modules.rollout_memory`;
this branch includes runtime-compatible memory implementations under
`smacdreamer.jepa.memory` so checkpoints can still load without importing JEPA
training entry points.

## Deliberate Backend Differences

JEPA mode does not provide:

- categorical RSSM stochastic state
- prior/posterior distributions
- sampled alternative futures
- RSSM KL losses
- RSSM prior/posterior entropy metrics

These are backend differences, not missing loss terms.

## Pending Release Gates

The real dataset episodes and checkpoint are absent, so these remain pending:

- real `.npz` online/offline token parity
- real action parity
- real checkpoint reconstruction
- original-runtime versus R2-wrapper parity
- real multi-step recursive-rollout parity
- real-checkpoint 5,000-step smoke run

Do not start long training until these pass.

## Commands

Inspect a checkpoint:

```bash
python scripts/inspect_jepa_checkpoint.py \
  --checkpoint /path/to/checkpoint.pt \
  --config configs/r2_650_jepa.yaml
```

Token parity:

```bash
python scripts/validate_jepa_token_parity.py \
  --checkpoint /path/to/checkpoint.pt \
  --episode-npz /path/to/episode.npz \
  --step 10 \
  --config configs/r2_650_jepa.yaml
```

Wrapper parity:

```bash
python scripts/validate_jepa_r2_integration.py \
  --checkpoint /path/to/checkpoint.pt \
  --episode-npz /path/to/episode.npz \
  --config configs/r2_650_jepa.yaml \
  --device cpu
```

Short smoke after parity gates:

```bash
python -u scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_650_jepa.yaml \
  --jepa-checkpoint /path/to/checkpoint.pt \
  --steps 5000 \
  --logdir logs/jepa_smoke_5k
```
