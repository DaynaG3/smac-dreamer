# Install, audit, and run the Option-Critic P0/P1 hotfix

## Paths

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export HOTFIX="$ROOT/option_critic_p0p1_hotfix_bundle"
```

## 1. Stop the old hierarchy run

```bash
pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
tmux ls || true
```

Attach to the old session, press `Ctrl-C`, then verify no trainer remains:

```bash
pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
```

Preserve the old run directory; do not delete it.

## 2. Verify and extract the ZIP

```bash
cd "$ROOT"
sha256sum option_critic_p0p1_hotfix_bundle.zip
rm -rf "$HOTFIX"
unzip -q option_critic_p0p1_hotfix_bundle.zip -d "$ROOT"
```

Compare the hash with `HOTFIX_SHA256.txt` supplied alongside the ZIP.

## 3. Run the fail-closed dry-run

```bash
"$PY" "$HOTFIX/apply_option_critic_p0p1_hotfix.py" \
  --repo "$REPO" \
  --dry-run
```

Required ending:

```text
[OK] P0/P1 hotfix dry-run matched integrated v2, parsed all Python, and resolved v3 config
```

## 4. Install

```bash
"$PY" "$HOTFIX/apply_option_critic_p0p1_hotfix.py" \
  --repo "$REPO"
```

Resolve the backup:

```bash
export P0P1_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_p0p1_hotfix_backup_* |
  head -1
)"

echo "$P0P1_BACKUP"
cat "$P0P1_BACKUP/option_critic_p0p1_hotfix_backup_manifest.json"
```

## 5. Verify the Tactical Mixture v1.2 source

```bash
export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"
export SOURCE_CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"

"$PY" - "$SOURCE_CHECKPOINT" <<'PY'
import sys, torch
p=sys.argv[1]
c=torch.load(p,map_location='cpu',weights_only=False)
m=c.get('tactical_mixture_metadata') or {}
s=c.get('agent_state_dict') or {}
print('checkpoint:',p)
print('step:',c.get('step'))
print('macro win:',c.get('val_macro_win_rate'))
print('original return:',c.get('val_macro_original_return'))
print('metadata:',m)
assert m.get('architecture')=='tactical_mixture_v1_2'
assert int(m.get('num_tactics',-1))==2
assert float(c.get('val_macro_win_rate',-1))>=0.3749
assert not any(k.startswith('hierarchical_options.') for k in s)
print('[OK] correct Tactical Mixture v1.2 source selected')
PY
```

## 6. Run the complete static audit

```bash
cd "$REPO"

REPO="$REPO" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_option_critic_8_v3_p0p1.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
bash scripts/static_audit_option_critic_hierarchy.sh \
  2>&1 | tee "$ROOT/option_critic_p0p1_audit.txt"
```

Required ending:

```text
47 passed
[OK] Option-Critic P0/P1 source/config audit passed
[OK] Option-Critic P0/P1 hotfix static audit passed
```

Do not launch unless all three appear.

## 7. Start a fresh 2M run in tmux

```bash
tmux new -s r2_option_critic_p0p1
```

Inside tmux:

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"

unset SOURCE_CHECKPOINT SOURCE_RUN_META RUN_DIR CONFIG
source "$ROOT/.venv/bin/activate"
cd "$REPO"

ROOT="$ROOT" \
REPO="$REPO" \
PY="$PY" \
TACTICAL_V12_RUN="$TACTICAL_V12_RUN" \
FINAL_STEP=2000000 \
bash scripts/run_option_critic_2m.sh
```

The launcher ignores stale source-checkpoint variables and derives both source files directly from `TACTICAL_V12_RUN`.

Required startup sequence includes:

```text
[OK] selected Tactical Mixture v1.2 best checkpoint
47 passed
[OK] Option-Critic P0/P1 hotfix static audit passed
[START] corrected Option-Critic P0/P1 run: ...
```

Detach with `Ctrl-b`, then `d`.

## 8. Monitor

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export OPTION_RUN="$(cat "$ROOT/CURRENT_OPTION_CRITIC_V3_P0P1_RUN.txt")"

echo "$OPTION_RUN"
pgrep -af 'train_r2dreamer_smaclite_multimap.py'
tail -f "$OPTION_RUN/train.log"
```

## 9. Runtime invariant checks

Once hierarchy metrics appear:

```bash
"$PY" "$REPO/scripts/assert_option_critic_metrics.py" "$OPTION_RUN"
```

Run it before 100k, between 100k and 300k, after 300k, and after each validation.

Macro-win guard:

```bash
"$PY" "$REPO/scripts/check_option_critic_win_guard.py" \
  "$OPTION_RUN" \
  --source-checkpoint "$TACTICAL_V12_RUN/best_val_macro_winrate.pt" \
  --max-regression 0.03
```

A failure means the latest recorded macro validation win rate is more than three percentage points below the source checkpoint. Preserve the run and inspect before continuing.

## 10. Rollback

Stop the trainer first, then:

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export HOTFIX="$ROOT/option_critic_p0p1_hotfix_bundle"
export P0P1_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_p0p1_hotfix_backup_* |
  head -1
)"

PY="$PY" bash "$HOTFIX/restore_option_critic_p0p1_hotfix.sh" \
  "$P0P1_BACKUP" \
  "$REPO"
```
