# Exact installation and run commands

## 1. Stop the current Option-Critic run

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"

pgrep -af 'train_r2dreamer_smaclite_multimap.py' || true
tmux ls || true
```

Attach to the active session, press `Ctrl-C`, then confirm no trainer remains.
Do not resume the current v3 run after this patch; preserve its directory only
for comparison.

## 2. Verify and extract

```bash
cd "$ROOT"
sha256sum option_critic_p1_final_1m_pipeline_bundle.zip
rm -rf "$ROOT/option_critic_p1_final_1m_pipeline_bundle"
unzip -q "$ROOT/option_critic_p1_final_1m_pipeline_bundle.zip" -d "$ROOT"
export PATCH="$ROOT/option_critic_p1_final_1m_pipeline_bundle"
```

Compare the hash with `OPTION_CRITIC_P1_FINAL_SHA256.txt`.

## 3. Fail-closed dry-run

```bash
"$PY" "$PATCH/apply_option_critic_p1_final_hotfix.py" \
  --repo "$REPO" \
  --dry-run
```

Required ending:

```text
[OK] P1-final dry-run matched integrated v3, parsed payloads, and resolved the 1M config
```

## 4. Install

```bash
"$PY" "$PATCH/apply_option_critic_p1_final_hotfix.py" \
  --repo "$REPO"
```

Resolve the backup:

```bash
export P1_FINAL_BACKUP="$(
  ls -dt "$ROOT"/smac-dreamer_option_critic_p1_final_backup_* | head -1
)"
echo "$P1_FINAL_BACKUP"
cat "$P1_FINAL_BACKUP/option_critic_p1_final_backup_manifest.json"
```

## 5. Verify the source checkpoint

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
print('step:',c.get('step'))
print('macro win:',c.get('val_macro_win_rate'))
print('metadata:',m)
assert m.get('architecture')=='tactical_mixture_v1_2'
assert int(m.get('num_tactics',-1))==2
assert float(c.get('val_macro_win_rate',-1))>=0.3749
assert not any(k.startswith('hierarchical_options.') for k in s)
print('[OK] correct Tactical Mixture v1.2 source selected')
PY
```

## 6. Run the 51-test real-repository audit

```bash
cd "$REPO"
REPO="$REPO" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_option_critic_8_v4_p1_final_1m.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
bash scripts/static_audit_option_critic_p1_final.sh \
  2>&1 | tee "$ROOT/option_critic_p1_final_audit.txt"
```

Required ending:

```text
51 passed
[OK] Option-Critic P1-final source/config audit passed
[OK] Option-Critic P1-final static audit passed
```

## 7. Run the complete RL -> forecast pipeline with one command

```bash
cd "$REPO"
bash scripts/run_option_critic_1m_then_exp45_pipeline.sh
```

The script starts a detached tmux session and prints its name. Attach with the
printed command, or inspect the current pipeline pointer:

```bash
cat "$ROOT/CURRENT_OPTION_CRITIC_AND_EXP45_PIPELINE.txt"
tmux ls
```

The RL run pointer is:

```text
$ROOT/CURRENT_OPTION_CRITIC_V4_P1_1M_RUN.txt
```

The master pipeline defaults are:

```text
CONTINUE_ON_FAILURE=1
EVAL_PARTIAL_ON_TRAIN_FAILURE=1
STRICT_EXIT=0
AUTO_TMUX=1
```

Thus, RL failure does not block Exp45. Internal forecast failures are recorded,
and the master script still reaches a final summary. To make the master return
nonzero after completing all attempted stages, launch with `STRICT_EXIT=1`.

## 8. Monitor RL

```bash
export OPTION_RUN="$(cat "$ROOT/CURRENT_OPTION_CRITIC_V4_P1_1M_RUN.txt")"
tail -f "$OPTION_RUN/train.log"
```

Runtime checks:

```bash
"$PY" "$REPO/scripts/assert_option_critic_p1_final_metrics.py" "$OPTION_RUN"
```

Run around 200k, 400k, 600k, 800k and 1M.

## 9. Rollback

Stop trainers first, then:

```bash
PY="$PY" bash "$PATCH/restore_option_critic_p1_final_hotfix.sh" \
  "$P1_FINAL_BACKUP" \
  "$REPO"
```
