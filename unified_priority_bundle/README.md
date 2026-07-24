# Unified Adaptive Priority bundle

Target repository/branch:

```text
DaynaG3/smac-dreamer
combined-jepa-dreamer
repository root: .../combined-upload/smac-dreamer
```

This bundle installs the two requested mechanisms together before a trained
Exp40 R2-Dreamer checkpoint is continued:

1. **Automatic adaptive map collection**
   - map scores come only from the agent's per-timestep critic error;
   - scores are combined with staleness and a uniform coverage floor;
   - probabilities are published through a shared CPU tensor to environment workers;
   - held-out validation never writes feedback or receives adaptive probabilities.

2. **Automatic recurrent-sequence prioritisation**
   - TorchRL's existing `SliceSampler` still constructs valid contiguous windows;
   - a larger uniform candidate pool is priority-resampled as complete windows;
   - priorities use masked mean absolute error between replay return targets and
     the **current trainable critic**;
   - importance-sampling weights are applied to replay prediction losses and to
     actor/critic imagination starts.

The sequence method is deliberately described as **candidate-pool sequence
PER**, not mathematically exact global PER. It avoids relying on private
TorchRL 0.9.x slice-index internals while preserving recurrent-window semantics.

## Resume fixes included

The integration also corrects issues that become critical when the old replay is
not restored:

- absolute environment step resumes from `checkpoint["step"]`;
- the 2M stopping target remains absolute;
- checkpoint step reporting no longer restarts from zero;
- PER beta annealing continues from the restored step;
- Python, NumPy, Torch, and CUDA RNG state is restored from the checkpoint;
- learner updates wait until the new replay contains a full recurrent window;
- map priority state is checkpointed and restored;
- sequence-priority state is not restored without the matching replay contents;
- recycled adaptive workers receive generation-specific sampler seeds;
- configuration preflight compares the new YAML with the source run metadata,
  including the JEPA checkpoint SHA-256.

The old branch stored checkpoint `step` from replay size. Once a finite replay
reached capacity, that stored value could become a lower bound rather than the
true environment step. The bundle therefore adds `--resume-start-step`, the
`RESUME_START_STEP` runner variable, and `scripts/infer_resume_step.py`. Verify
the chosen value against the source run's W&B/logs before continuing.

## Preserve first

Run the included preservation script **before** installing:

```bash
BUNDLE=/path/to/unified_priority_bundle
ROOT=~/workspace/dreamer/combined-upload
REPO="$ROOT/smac-dreamer"
CKPT="$REPO/logs/r2dreamer/exp40_jepa_dense_v3_perm_imagmask_100k_2m_20260710_022902/latest.pt"

"$BUNDLE/preserve_before_unified_priority.sh" "$REPO" "$CKPT"
```

It stores Git refs, staged/unstaged patches, untracked code/config files, the
source checkpoint, and source-run metadata. The installer separately backs up
every existing file it edits.

## Install

Use the **exact YAML that created the trained model** as `--base-config`.
The installer copies it to `configs/r2_2100_jepa_unified_priority.yaml` and
changes only sampling mode plus the new adaptive-priority section.

```bash
ROOT=~/workspace/dreamer/combined-upload
REPO="$ROOT/smac-dreamer"
PY="$ROOT/.venv/bin/python"
BUNDLE=/path/to/unified_priority_bundle
BASE_CONFIG=configs/r2_2100_jepa_reward_shaped.yaml

"$PY" "$BUNDLE/install_unified_priority.py" \
  --repo "$REPO" \
  --base-config "$BASE_CONFIG" \
  --dry-run

"$PY" "$BUNDLE/install_unified_priority.py" \
  --repo "$REPO" \
  --base-config "$BASE_CONFIG"
```

The installer is fail-closed: a missing or ambiguous source marker stops the
installation instead of guessing.

## Files patched

```text
src/smacdreamer/envs/map_sampler.py
src/smacdreamer/r2dreamer_factory.py
src/smacdreamer/checkpointing.py
external/r2dreamer/trainer.py
external/r2dreamer/dreamer.py
scripts/train_r2dreamer_smaclite_multimap.py
```

## Files added

```text
src/smacdreamer/adaptive_priority.py
external/r2dreamer/adaptive_buffer.py
configs/r2_2100_jepa_unified_priority.yaml
scripts/preflight_unified_priority.py
scripts/assert_unified_priority_metrics.py
scripts/inspect_unified_priority_checkpoint.py
scripts/infer_resume_step.py
scripts/static_audit_unified_priority.sh
scripts/run_unified_priority_resume_smoke.sh
scripts/run_unified_priority_resume_full.sh
tests/test_adaptive_priority_controller.py
tests/test_adaptive_buffer_math.py
tests/test_adaptive_buffer_torchrl.py
tests/test_adaptive_map_sampler.py
```

## Static and unit checks

```bash
REPO="$ROOT/smac-dreamer" PY="$PY" \
  "$REPO/scripts/static_audit_unified_priority.sh"

PYTHONPATH="$REPO/src:$REPO/external/r2dreamer" "$PY" -m pytest -q \
  "$REPO"/tests/test_adaptive_*.py

# The requested evaluation cadence must remain 200k.
"$PY" - "$REPO/configs/r2_2100_jepa_unified_priority.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
assert int(cfg.validation.every) == 200000, cfg.validation.every
assert str(cfg.sampling_mode) == "adaptive_priority"
assert bool(cfg.adaptive_priority.map.enabled)
assert bool(cfg.adaptive_priority.sequence.enabled)
print("[OK] YAML: adaptive map + sequence priority, validation every 200k")
PY
```

## Determine the trusted continuation step

```bash
SOURCE_RUN="$(dirname "$CKPT")"

"$PY" "$REPO/scripts/infer_resume_step.py" \
  --checkpoint "$CKPT" \
  --run-dir "$SOURCE_RUN"
```

When W&B or the source logs show a larger true environment step than the stored
checkpoint value, export it explicitly for every smoke/full command:

```bash
export RESUME_START_STEP=<TRUSTED_ABSOLUTE_ENV_STEP>
```

Do not guess this value. Using the checkpoint value is conservative and may
repeat training; using an incorrectly larger value can stop training too early.

## Source checkpoint/config compatibility check

```bash
SOURCE_RUN="$(dirname "$CKPT")"

"$PY" "$REPO/scripts/preflight_unified_priority.py" \
  --repo "$REPO" \
  --checkpoint "$CKPT" \
  --config configs/r2_2100_jepa_unified_priority.yaml \
  --source-run-meta "$SOURCE_RUN/run_meta.json"
```

This must pass before the smoke continuation.

## Smoke continuation

The smoke starts from the original trained checkpoint but writes to a new run
directory and a new replay. It advances the absolute step by 5,000.

```bash
SMOKE1="$REPO/logs/r2dreamer/exp40_unified_priority_smoke_1"

ROOT="$ROOT" REPO="$REPO" PY="$PY" \
CHECKPOINT="$CKPT" \
CONFIG=configs/r2_2100_jepa_unified_priority.yaml \
RUN="$SMOKE1" \
SMOKE_ADDITIONAL_STEPS=5000 \
RESUME_START_STEP="${RESUME_START_STEP:-}" \
  "$REPO/scripts/run_unified_priority_resume_smoke.sh"
```

The smoke runner now fails automatically unless it finds finite critic-error,
sequence-PER, importance-weight, and map-priority metrics, and unless both sampling
distributions become non-uniform.

Then test checkpoint/adaptive-state restoration with a second, short restart:

```bash
SMOKE2="$REPO/logs/r2dreamer/exp40_unified_priority_smoke_2"

ROOT="$ROOT" REPO="$REPO" PY="$PY" \
CHECKPOINT="$SMOKE1/latest.pt" \
SOURCE_RUN_META="$SMOKE1/run_meta.json" \
CONFIG=configs/r2_2100_jepa_unified_priority.yaml \
RUN="$SMOKE2" \
SMOKE_ADDITIONAL_STEPS=5000 \
RESUME_START_STEP="" \
  "$REPO/scripts/run_unified_priority_resume_smoke.sh"
```

Do not use either smoke checkpoint for the scientific full run. After both
passes, restart the clean full continuation from the original checkpoint.

## Full continuation to absolute step 2,000,000

```bash
ROOT="$ROOT" REPO="$REPO" PY="$PY" \
CHECKPOINT="$CKPT" \
CONFIG=configs/r2_2100_jepa_unified_priority.yaml \
RESUME_START_STEP="${RESUME_START_STEP:-}" \
FINAL_STEP=2000000 \
  "$REPO/scripts/run_unified_priority_resume_full.sh"
```

This is a forked continuation with a new data-collection and replay distribution.
Keep the original Exp40 run untouched as the baseline. Run the full command inside
`tmux`; monitor `train.log`, `metrics.jsonl`, and `latest.pt` in the new run directory.
