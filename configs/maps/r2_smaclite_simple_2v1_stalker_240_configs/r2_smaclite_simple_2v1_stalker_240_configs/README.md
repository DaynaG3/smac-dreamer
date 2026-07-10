# R2-Dreamer × SMAClite simple 2v1 stalker benchmark

This directory contains **240 deterministic custom SMAClite map JSON files** for a deliberately simple high-winrate target:

- Allied team: exactly `2 × STALKER`
- Enemy team: exactly `1 × STALKER`
- Unit type: stalker only, on both teams
- Difficulty target: theoretically easy 2v1 mirror fight
- Terrain source: existing SMAClite presets only — `SIMPLE`, `NARROW`, `OCTAGON`

## Split

- `configs/train`: 160 maps
- `configs/validation`: 40 maps
- `configs/blind_iid`: 40 maps

There is intentionally no `blind_compositional` split because the composition is fixed by design. The files contain only fields accepted by SMAClite's custom map parser; research metadata is stored separately in `manifest.jsonl` and `manifest.csv`.

## Design choices

1. **Single scenario family:** all maps are `2 STALKER` allies versus `1 STALKER` enemy.
2. **High-winnability bias:** the static allied combat-value ratio is exactly 2.0× in every map.
3. **Existing terrain presets only:** no custom terrain arrays are generated.
4. **Mostly facing layouts:** most maps are left-vs-right or same-side vertical encounters, with small angular/positional variation.
5. **Controlled variation:** spawn positions, ally grouping, formation, engagement distance and terrain are randomized deterministically.
6. **No accidental pathfinding traps:** `NARROW` maps keep both teams on the same side of the wall; far maps avoid `NARROW`.
7. **Global type vocabulary:** every map keeps the same nine-entry `unit_type_ids` mapping used in the previous benchmark format.
8. **Reproducibility:** fixed seed `19062026`; rerun `generate_simple_2v1_stalker_240.py` to recreate the dataset.

## Engagement distribution

- Train: 40 immediate, 72 near, 40 medium, 8 far
- Validation: 10 immediate, 18 near, 10 medium, 2 far
- Blind-IID: 10 immediate, 18 near, 10 medium, 2 far

## Files

- `generate_simple_2v1_stalker_240.py`: self-contained deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted focus-fire policy
- `manifest.jsonl` / `manifest.csv`: per-map split, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: the fixed 2v1 stalker family definition
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: config-file content checksums

## Static validation result

- Files: 240
- Errors: 0
- Unique semantic configs: 240
- Seed: 19062026

## Required dynamic validation

Static checks cannot prove trained-policy win rate. After copying this folder into your repo, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_2v1_stalker/validate_in_smaclite.py \
  --root configs/maps/r2_2v1_stalker \
  --episodes 10 \
  --max-steps 160
```

Use `dynamic_validation.csv` to identify any layouts where even the scripted focus-fire baseline times out or loses. For the Dreamer run, select checkpoints using validation/blind-IID win rate and original SMAClite return.
