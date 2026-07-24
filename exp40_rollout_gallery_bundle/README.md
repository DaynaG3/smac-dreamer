# Exp40 rollout-gallery collector

This bundle runs the final Exp40 standalone JEPA checkpoint for **15 autonomous recursive steps** across many held-out rollout starts. It ranks actual examples for a visual-first results section:

- a good, eventful H1/H5/H15 example;
- late rollout drift;
- position drift;
- missed health/damage changes;
- enemy tracking failures;
- presence/lifecycle mistakes;
- copying when the environment changes;
- unstable overshoot;
- failures around natural enemy visibility transitions.

It does **not** run a memory ablation.

## Exact checkpoint used by default

```text
~/workspace/dreamer/combined-upload/smac-jepa-wm/runs/
rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt
```

The script falls back to the R2-installed copy at:

```text
~/workspace/dreamer/combined-upload/smac-dreamer/checkpoints/jepa/model_exp40.pt
```

## Run

Upload/unzip this folder on the A40 machine, then:

```bash
chmod +x run_exp40_rollout_gallery.sh
./run_exp40_rollout_gallery.sh
```

The default evaluates approximately:

```text
80 batches × 16 dataset windows × 20 rollout starts = 25,600 rollouts
```

Each rollout is 15 steps long. The exact number can be slightly smaller on a partial final batch.

For a larger mining pass:

```bash
MAX_BATCHES=150 TOP_K=8 ./run_exp40_rollout_gallery.sh
```

For a faster first check:

```bash
MAX_BATCHES=10 TOP_K=3 ./run_exp40_rollout_gallery.sh
```

## Main output

At completion, the script prints the path to:

```text
UPLOAD_THIS_BACK_TO_CHAT.zip
```

Upload that ZIP. It contains the selected H1/H5/H15 figures, trajectory overlays, detailed entity/feature tables, all-example ranking data, and the horizon curves needed after the qualitative slides.

## Evaluation semantics

- The rollout starts once from a true observation-limited state.
- Recorded allied actions are supplied at each transition.
- Every future latent is produced recursively from the model's own previous prediction.
- Predicted presence is propagated.
- Future ground-truth visibility and presence are never supplied to the rollout.
- H15 is collected as a long-horizon stress test; the checkpoint was trained on H5.
