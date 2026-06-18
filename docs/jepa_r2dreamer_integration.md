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

The gradient boundary is intentional:

```python
with torch.no_grad():
    conditioned = frozen_memory_module.condition(latent, memory, entity_mask)

feature = trainable_feature_adapter(
    conditioned.detach(),
    memory.detach(),
    entity_mask.detach(),
    static_condition.detach(),
)
```

Losses from reward, continuation, availability, alive prediction and value
learning may update the feature adapter. They must not update the JEPA encoder,
predictor, presence head, decoder, projector, or recurrent memory.

Synthetic unit and Dreamer-level tests now run a real backward pass and optimizer
step to verify:

- `JEPAFeatureAdapter` receives finite, nonzero gradients
- at least one adapter parameter changes after an optimizer step
- frozen JEPA parameters receive no gradients
- frozen JEPA parameters remain bitwise unchanged

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

`jepa_entity_slot_mask` is structural. It marks only entity slots that physically
exist on the current map:

```text
allies:  [0, n_agents)
enemies: [max_agents, max_agents + n_enemies)
```

Padded ally and enemy slots are zero. This is separate from:

- `jepa_entity_mask`: currently visible/present entity tokens
- `agent_alive_mask`: allied slots currently capable of acting
- `avail_actions`: valid actions for each allied slot

During JEPA imagination, predicted presence is always intersected with the
structural slot mask, so padded entities cannot become active.

## Visibility Masking

The selected JEPA training path is visibility-aware. Online token construction
therefore applies the same observation-side masking as
`VisibilityMarkovRolloutSMACJEPADataset`: enemy dynamic features outside allied
sight range are zeroed before tokenization. This never uses offline target/full
state tensors during acting.

Visibility settings are read from checkpoint metadata or resolved config and
passed into training workers and isolated validation children. A checkpoint that
requires visibility masking must match runtime metadata; the loader fails rather
than silently disabling masking.

The restored dataset code treats allied liveness as feature-column 0 > 0, and
enemy presence as a nonzero enemy feature row. Synthetic tests match that source
behavior. Real `.npz` parity is still a release gate.

## Recurrent Memory

Action-conditioned recurrent memory preserves prior memory for masked entities:

```python
new_memory = torch.where(entity_mask[..., None], proposed_new_memory, previous_memory)
```

This distinction matters for temporarily invisible entities. Structurally padded
entities remain disabled through the slot mask and start from zero memory.

## Checkpoint Contract

The source JEPA checkpoint must contain:

- `model_state`
- `memory_module_state`
- `metadata`
- `resolved_config` or `config`

The loader validates metadata against the live R2 environment and fails on
mismatches. It never loads the JEPA optimizer or scaler, and it never falls back
to RSSM.

Validated fields include mode, agent/enemy/action dimensions, token dimensions,
dynamic/static feature dimensions, shield flags, unit-type vocabulary size,
latent dimension, recurrent-memory dimension, action-conditioned-memory setting,
visibility-mask setting, sight range, coordinate indices, and latent
normalization mode. Missing live metadata is treated as an incompatibility.

This branch includes runtime-compatible memory implementations under
`smacdreamer.jepa.memory` so R2 runtime code does not import JEPA training entry
points. Optional installed-source parity tests exist in
`tests/test_jepa_memory_source_parity.py`; they run when the corresponding
`smac_jepa` memory modules are importable.

## Deliberate Backend Differences

JEPA mode does not provide:

- categorical RSSM stochastic state
- prior/posterior distributions
- sampled alternative futures
- RSSM KL losses
- RSSM prior/posterior entropy metrics

These are backend differences, not missing loss terms.

## Readiness States

Implementation-ready means:

- synthetic JEPA unit tests pass
- JEPA-mode Dreamer update tests pass
- replay, worker recycling and validation-isolation regressions pass
- preflight tooling exists

Real-checkpoint validated means:

- `preflight_jepa_training.py` passes on the exact checkpoint and a matching
  real `.npz` episode

Smoke-test passed means:

- the 5,000-step JEPA training command completes with the real checkpoint

Full training approved means:

- real preflight passed
- 5,000-step smoke passed
- no parity or shape/device regressions were observed

## Pending Release Gates

The real dataset episodes and checkpoint are absent, so these remain pending:

- real `.npz` online/offline token parity
- real visibility-mask parity
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

Combined preflight:

```bash
python scripts/preflight_jepa_training.py \
  --checkpoint /path/to/checkpoint.pt \
  --episode-npz /path/to/episode.npz \
  --config configs/r2_650_jepa.yaml \
  --device cpu \
  --rollout-horizon 10 \
  --report-json logs/jepa_preflight_report.json
```

The command must finish with:

```text
JEPA R2-DREAMER PREFLIGHT: PASS
```

Short smoke after parity gates:

```bash
python -u scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_650_jepa.yaml \
  --jepa-checkpoint /path/to/checkpoint.pt \
  --steps 5000 \
  --logdir logs/jepa_smoke_5k
```
