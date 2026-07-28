# R2-Dreamer × SMAClite 3v2 basic mixed-unit benchmark

This directory contains **480 deterministic custom SMAClite map JSON files** for the next curriculum step after the 2v1 stalker-only benchmark.

## Scenario rule

Every map is exactly:

- `3` allied units
- `2` enemy units
- basic combat units only: `STALKER`, `ZEALOT`, `MARINE`, `MARAUDER`, `ZERGLING`
- no special/support/static/burst units in generated groups: `BANELING`, `MEDIVAC`, `SPINE_CRAWLER`, `COLOSSUS`
- existing SMAClite terrain presets only: `SIMPLE`, `NARROW`, `OCTAGON`
- theoretically winnable by static combat-value proxy; ally/enemy value ratio is always >= 1.20×

## Split

- `configs/train`: 320 maps
- `configs/validation`: 80 maps
- `configs/blind_iid`: 80 maps

There is intentionally no `blind_compositional` split in this first mixed-unit curriculum step. The validation and blind-IID splits use the same 16 composition families, but with unseen layouts, terrain/formation choices and spawn jitter.

## Composition families

There are 16 basic-combat families. Each family has 20 train variants, 5 validation variants and 5 blind-IID variants.

Examples:

- `3 STALKER` vs `2 STALKER`
- `3 ZEALOT` vs `2 ZEALOT`
- `2 STALKER + 1 ZEALOT` vs `1 STALKER + 1 ZEALOT`
- `3 MARINE` vs `2 MARINE`
- `2 MARINE + 1 MARAUDER` vs `1 MARINE + 1 MARAUDER`
- `3 ZERGLING` vs `2 ZERGLING`
- `2 MARINE + 1 ZERGLING` vs `1 MARINE + 1 ZERGLING`

Each faction is internally shield-homogeneous, so the map-level `ally_has_shields` and `enemy_has_shields` flags remain correct for SMAClite observations.

## Engagement distribution

- Train: 96 immediate, 128 near, 80 medium, 16 far
- Validation: 24 immediate, 32 near, 20 medium, 4 far
- Blind-IID: 24 immediate, 32 near, 20 medium, 4 far

This keeps the task mostly combat-rich while introducing a small amount of longer approach behaviour.

## Randomized dimensions

- spawn positions
- ally and enemy grouping
- formation: `compact`, `close_split`, `wide_split`, `staggered`, `type_split`
- terrain preset
- engagement distance
- facing orientation
- small deterministic jitter

`NARROW` maps keep teams on the same side of the wall/gate and far maps avoid `NARROW`, so the dataset does not accidentally become a pathfinding benchmark.

## Files

- `generate_3v2_basic_mixed_480.py`: deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted focus-fire policy
- `manifest.jsonl` / `manifest.csv`: per-map split, family, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: the 16 composition-family definitions
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: config-file content checksums

## Static validation result

- Files: 480
- Errors: 0
- Unique semantic configs: 480
- Seed: 21072026

## Dynamic validation

Static checks cannot prove trained-policy win rate. After copying this folder into your repo, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite python configs/maps/r2_3v2_basic_mixed/validate_in_smaclite.py   --root configs/maps/r2_3v2_basic_mixed   --episodes 5   --max-steps 180
```

Use `dynamic_validation.csv` to identify layouts where even the scripted focus-fire baseline has issues. For Dreamer, select checkpoints using validation/blind-IID win rate and original SMAClite return.
