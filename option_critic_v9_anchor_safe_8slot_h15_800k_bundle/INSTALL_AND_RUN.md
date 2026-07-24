# Install and run Option-Critic v9

## 1. Stop the current trainer

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"

pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
tmux ls || true
```

Attach to the active tmux session, press `Ctrl-C`, then confirm:

```bash
pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
```

## 2. Verify and extract the bundle

Place the ZIP under `$ROOT`, then:

```bash
cd "$ROOT"
sha256sum option_critic_v9_anchor_safe_8slot_h15_800k_bundle.zip
```

The hash must match `OPTION_CRITIC_V9_ANCHOR_SAFE_SHA256.txt` from the handoff.

```bash
rm -rf "$ROOT/option_critic_v9_anchor_safe_8slot_h15_800k_bundle"
unzip -q \
  "$ROOT/option_critic_v9_anchor_safe_8slot_h15_800k_bundle.zip" \
  -d "$ROOT"
export PATCH="$ROOT/option_critic_v9_anchor_safe_8slot_h15_800k_bundle"
```

## 3. Fail-closed dry-run

```bash
"$PY" "$PATCH/apply_option_critic_v9_anchor_safe_hotfix.py" \
  --repo "$REPO" \
  --dry-run
```

Required ending:

```text
[OK] v9 dry-run matched integrated v6, parsed all payloads, and resolved the Tactical-v1.2-based H=15 config
```

Do not continue if it fails.

## 4. Install transactionally

```bash
"$PY" "$PATCH/apply_option_critic_v9_anchor_safe_hotfix.py" \
  --repo "$REPO"
```

Record the printed backup path. It can also be resolved with:

```bash
export V9_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_v9_anchor_safe_backup_* |
  head -1
)"
echo "$V9_BACKUP"
cat "$V9_BACKUP/option_critic_v9_anchor_safe_backup_manifest.json"
```

## 5. Select the exact source checkpoint

```bash
export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"
export SOURCE_CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"

sha256sum "$SOURCE_CHECKPOINT"
```

Required SHA-256:

```text
74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e
```

## 6. Mandatory real-repository audit

```bash
cd "$REPO"

REPO="$REPO" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
EXPECTED_SOURCE_CHECKPOINT_SHA256=74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e \
bash scripts/static_audit_option_critic_v9_anchor_safe.sh \
  2>&1 | tee "$ROOT/option_critic_v9_anchor_safe_audit.txt"
```

Required ending:

```text
53 passed, 12 deselected
27 passed
[OK] Option-Critic v9 anchor-safe source/config audit passed
[OK] Option-Critic v9 anchor-safe static audit passed
```

Do not launch if any line differs through a failure.

## 7. Run forecast JEPA first, then RL

```bash
cd "$REPO"
bash scripts/run_forecast_first_then_option_critic_v9_800k.sh
```

The wrapper starts a detached tmux session and prints the attach command.

Pipeline order:

```text
Exp45 install/static audit/training
ordinary evaluation
hidden evaluation
then
Option-Critic v9: H=15, 800k new environment steps
```

Default failure policy:

```text
CONTINUE_ON_FAILURE=1
STRICT_EXIT=0
AUTO_TMUX=1
```

A forecast failure is recorded and RL is attempted only when no forecast trainer
is still alive. RL is safely skipped if a failed forecast left a GPU process.

## 8. Monitor forecast and RL

```bash
cat "$ROOT/CURRENT_FORECAST_THEN_OPTION_CRITIC_V9_PIPELINE.txt"
tmux ls
```

When RL starts:

```bash
export OPTION_RUN="$(
  cat "$ROOT/CURRENT_OPTION_CRITIC_V9_ANCHOR_SAFE_8SLOT_800K_RUN.txt"
)"
echo "$OPTION_RUN"
tail -f "$OPTION_RUN/train.log"
```

Validation occurs at startup, 200k, 400k, 600k, and 800k.

After learner metrics appear and after each validation:

```bash
"$PY" "$REPO/scripts/assert_option_critic_v9_metrics.py" \
  "$OPTION_RUN"
```

The assertion rejects stale metrics, source-action safety violations, option
state-machine violations, wrong H, wrong schedules, and broken eight-option
usage totals.

## 9. Rollback

Stop all trainers, then:

```bash
export PATCH="$ROOT/option_critic_v9_anchor_safe_8slot_h15_800k_bundle"
export V9_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_v9_anchor_safe_backup_* |
  head -1
)"

PY="$PY" bash "$PATCH/restore_option_critic_v9_anchor_safe_hotfix.sh" \
  "$V9_BACKUP" \
  "$REPO"
```

Rollback verifies backup hashes, restores all replaced files, and removes every
v9-introduced path.
