# Tactical Mixture Actor v1.1 — Hardening Bundle

This bundle hardens the already-installed **Tactical Mixture Actor v1**. It is intentionally a patch-on-top bundle: the unified adaptive-priority integration and the first tactical integration must already be present.

## What the current W&B graphs say

The current v1 tactical run should **not** be used as the next source checkpoint:

- held-out `val/micro_win_rate` fell from about **0.351** to **0.341** and then **0.293** by 600k;
- `usage_0` through `usage_3` remain close to 0.25 and `usage_max` remains around 0.25–0.31;
- those usage curves are only marginal selector probabilities, and v1 explicitly penalized deviation from uniform usage, so they do **not** establish that four state-dependent tactics were learned;
- `train/tar` remains noisy without a corresponding held-out improvement.

The hardened run must start again from the **adaptive-PER run's best non-tactical checkpoint**, not from any checkpoint produced by the declining v1 tactical run.

## Main fixes

1. Replaces KL-to-uniform tactic balancing with a collapse-only hinge.
2. Adds conditional entropy, marginal entropy, mutual information, sampled usage and argmax usage diagnostics.
3. Isolates the diversity auxiliary from the inherited actor and JEPA feature adapter.
4. Adds a bounded symmetry break so tactic branches do not begin at an exact zero-gradient symmetry point.
5. Uses a smooth confidence gate during deterministic validation instead of applying arbitrary tactic 0 when the selector is nearly uniform.
6. Freezes the inherited base actor and JEPA feature adapter for the recommended clean residual experiment.
7. Hard-bounds each residual logit to `[-4, 4]` and adds a residual/base guard loss.
8. Fixes uniform `Buffer` compatibility for `set_env_step`.
9. Skips adaptive-priority checkpoint state when adaptive priority is disabled.
10. Saves tactical metadata in `best_val_macro_winrate.pt` and supports safe legacy migration.
11. Makes the effect diagnostic compatible with `torch.compile(fullgraph=True)` by removing data-dependent `nonzero` and Tensor-to-Python branches.
12. Corrects the W&B JEPA adapter count: total adapter parameters and actually trainable adapter parameters are logged separately.
13. Adds source-order checks for masking, world-model action dimensions, optimizer registration and config plumbing.
14. Forces the launch script to select `best_val_macro_winrate.pt` inside `CURRENT_UNIFIED_PRIORITY_RUN.txt`; a stale generic `CHECKPOINT` variable cannot silently override it.
15. Creates an exact pre-hardening backup plus a manifest-driven restore path.
16. Resolves and validates the complete output YAML before writing any repository file.
17. Forces replay scratch storage to the new run's relative `replay/` directory and refuses non-empty run directories.
18. Automatically rolls back from the backup if installation fails during the write phase.
19. Verifies that both the checkpoint and `run_meta.json` come from the selected adaptive run.

See `AUDIT_FINDINGS.md` for the full review.

---

# Installation

Place the ZIP in:

```text
/home/jovyan/workspace/dreamer/combined-upload
```

Then run:

```bash
cd /home/jovyan/workspace/dreamer/combined-upload
unzip tactical_mixture_hardening_bundle.zip

export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export BUNDLE="$ROOT/tactical_mixture_hardening_bundle"

export ADAPTIVE_RUN="$(cat "$ROOT/CURRENT_UNIFIED_PRIORITY_RUN.txt")"
export SOURCE_CHECKPOINT="$ADAPTIVE_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$ADAPTIVE_RUN/run_meta.json"
```

Verify the paths:

```bash
test -x "$PY" || { echo "[FAIL] Python missing: $PY"; exit 1; }
test -d "$REPO" || { echo "[FAIL] repo missing: $REPO"; exit 1; }
test -s "$SOURCE_CHECKPOINT" || {
  echo "[FAIL] adaptive best checkpoint missing: $SOURCE_CHECKPOINT"
  exit 1
}
test -s "$SOURCE_RUN_META" || {
  echo "[FAIL] adaptive run metadata missing: $SOURCE_RUN_META"
  exit 1
}

readlink -f "$SOURCE_CHECKPOINT"
sha256sum "$SOURCE_CHECKPOINT"
```

The resolved checkpoint path must be inside the adaptive-priority run and end with:

```text
best_val_macro_winrate.pt
```

## Stop the declining tactical v1 run

```bash
pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
```

Stop that process through its tmux session before editing or launching. The hardened launch script also refuses to start when another multimap trainer is active.

## Fail-closed dry run

```bash
"$PY" "$BUNDLE/install_tactical_hardening.py" \
  --repo "$REPO" \
  --source-config configs/r2_2100_jepa_tactical_mixture.yaml \
  --dry-run
```

Required ending:

```text
[OK] hardening dry-run matched all source anchors, parsed all ASTs, and resolved the output config
```

Do not bypass a failed anchor.

## Install

```bash
"$PY" "$BUNDLE/install_tactical_hardening.py" \
  --repo "$REPO" \
  --source-config configs/r2_2100_jepa_tactical_mixture.yaml
```

Record the generated backup:

```bash
export HARDENING_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_tactical_hardening_backup_* |
  head -1
)"

echo "$HARDENING_BACKUP"
cat "$HARDENING_BACKUP/hardening_backup_manifest.json"
```

The installer backs up every replaced file and any payload destination that existed before installation.

---

# Static audit

```bash
cd "$REPO"

REPO="$REPO" \
ROOT="$ROOT" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_tactical_mixture_hardened.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
  bash scripts/static_audit_tactical_hardening.sh
```

Then run the source-lineage check that explicitly rejects tactical source checkpoints:

```bash
"$PY" scripts/audit_tactical_hardening.py \
  --repo "$REPO" \
  --config configs/r2_2100_jepa_tactical_mixture_hardened.yaml \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --source-run-meta "$SOURCE_RUN_META" \
  --require-legacy-source
```

Do not launch unless both finish with `[OK]`.

---

# Launch the new 2M run

No hour-long smoke is required. The bundle uses unit tests, AST/source audits, checkpoint-lineage checks and a fullgraph compile test before launch.

```bash
tmux new -s r2_tactical_hardened
```

Inside tmux:

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export ADAPTIVE_RUN="$(cat "$ROOT/CURRENT_UNIFIED_PRIORITY_RUN.txt")"

source "$ROOT/.venv/bin/activate"

ROOT="$ROOT" \
REPO="$REPO" \
PY="$PY" \
ADAPTIVE_RUN="$ADAPTIVE_RUN" \
FINAL_STEP=2000000 \
  bash "$REPO/scripts/run_tactical_hardened_2m.sh"
```

The launch script will:

- use `$ADAPTIVE_RUN/best_val_macro_winrate.pt`;
- reject a checkpoint outside that run;
- reject `latest.pt` or any tactical checkpoint;
- start a fresh phase at step 0;
- refuse a non-empty run directory, preventing stale replay/memmap reuse;
- use fresh optimizer, scheduler, scaler, return EMA and replay;
- disable adaptive map priority and sequence PER for this isolated experiment;
- retain imagination horizon 5;
- validate every 200k with no startup validation.

Detach with `Ctrl-b`, then `d`.

## Monitor

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export RUN="$(cat "$ROOT/CURRENT_TACTICAL_HARDENED_RUN.txt")"

tail -f "$RUN/train.log"
```

Once tactic metrics appear:

```bash
"$PY" "$REPO/scripts/assert_tactical_hardened_metrics.py" "$RUN"
```

The new diagnostics distinguish:

- balanced marginal usage;
- actual sampled usage;
- deterministic argmax usage;
- state-dependent specialization through mutual information;
- whether each tactic changes primitive action distributions;
- whether the residual is overpowering the inherited actor.

---

# Restore

```bash
"$BUNDLE/restore_tactical_hardening_backup.sh" \
  "$HARDENING_BACKUP" \
  "$REPO"
```

The backup directory is intentionally retained after restoration.

---

# Files modified

- `external/r2dreamer/dreamer.py`
- `external/r2dreamer/tactical_policy.py`
- `scripts/train_r2dreamer_smaclite_multimap.py`
- `src/smacdreamer/validation_trainer.py`

# Files added

- `configs/r2_2100_jepa_tactical_mixture_hardened.yaml`
- `scripts/audit_tactical_hardening.py`
- `scripts/assert_tactical_hardened_metrics.py`
- `scripts/static_audit_tactical_hardening.sh`
- `scripts/run_tactical_hardened_2m.sh`
- `tests/test_tactical_policy_hardened.py`

# Files deliberately untouched

- JEPA core and checkpoint
- JEPA transition action dimensionality
- SMACLite environment
- map files
- replay storage implementation
- adaptive-priority implementation
- action-mask construction
- critic/reward/continuation architecture
