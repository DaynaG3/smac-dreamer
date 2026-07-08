# Entity value tables

These tables compare the decoded target token values against reconstruction, one-step prediction, and open-loop prediction.

Important convention:
- `target_timestep = 1` means actual `x_1`, predicted from `x_0` and `action_0`.
- Values are normalized entity-token values, not raw map pixels.
- Allies use dynamic feature order: `['hp', 'cooldown_or_energy', 'dx', 'dy', 'shield', 'unit_type_0', 'unit_type_1', 'unit_type_2', 'unit_type_3', 'unit_type_4', 'unit_type_5', 'unit_type_6', 'unit_type_7', 'unit_type_8']`.
- Enemies use dynamic feature order: `['hp', 'dx', 'dy', 'shield', 'unit_type_0', 'unit_type_1', 'unit_type_2', 'unit_type_3', 'unit_type_4', 'unit_type_5', 'unit_type_6', 'unit_type_7', 'unit_type_8']`.
- `dx` and `dy` are normalized offsets from the map center in the repo's human-readable decoder.
- `observed_at_input_t` indicates whether the entity was visible/observed at the input state used for that transition.
- `observed_at_target_t` indicates whether the entity was visible/observed at the resulting target state.

Files:
- `entity_value_tables/entity_values_tXXXX.csv`: long table, one row per entity-feature.
- `entity_compact_tables/entity_compact_tXXXX.csv/md`: compact per-entity table for hp/shield/cooldown/dx/dy.
- `slot_error_summary.csv`: aggregate errors per entity slot and feature group.
- `feature_error_summary.csv`: aggregate errors per entity slot and individual feature.
- `top_one_step_feature_errors.csv`: largest one-step feature errors.
- `top_open_loop_feature_errors.csv`: largest open-loop feature errors.

Current detailed table feature mode: `important`.
Use `--value-table-features dynamic` to include all dynamic features, or `--value-table-features all` to include static/extra token dimensions too.
