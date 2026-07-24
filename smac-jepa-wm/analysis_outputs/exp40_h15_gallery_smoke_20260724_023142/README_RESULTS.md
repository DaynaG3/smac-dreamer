# Exp40 H15 Rollout Gallery Results

Checkpoint: `/home/jovyan/workspace/dreamer/combined-upload/smac-jepa-wm/runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt`
Held-out rollout examples evaluated: **160**

## Start here

1. Open `selections.csv` or the `Selections` sheet in `rollout_gallery.xlsx`.
2. Inspect `examples/good_eventful/01_*/overview_h1_h5_h15.png` for the first candidate slide.
3. Inspect each failure-category folder for distinct data-grounded failure slides.
4. Upload `UPLOAD_THIS_BACK_TO_CHAT.zip` for final example selection and presentation analysis.

## Category meanings

- **good_eventful**: Eventful rollout with low error at H15; avoids choosing a trivial static clip.
- **late_rollout_drift**: Prediction is relatively accurate early but diverges substantially after H5.
- **position_drift**: Entity positions drift away from the recorded trajectory by H15.
- **health_or_damage_miss**: The rollout misses health/shield changes on transitions where those values change.
- **enemy_tracking_failure**: Enemy-state prediction is substantially worse than allied-state prediction.
- **presence_lifecycle_failure**: The presence head invents an absent entity or removes one that remains present.
- **copying_dynamic_change**: The target changes, but the predicted state remains too close to the rollout start.
- **unstable_overshoot**: The prediction changes more aggressively than the recorded transition or leaves plausible health bounds.
- **visibility_transition_failure**: Enemy error rises around natural visibility changes or while the enemy is hidden.

## Evaluation semantics

The first rollout state is grounded in the held-out trajectory. Every later state is generated recursively from the model's own prediction using recorded allied actions. Future visibility and future ground-truth presence are not supplied to the predictor.
