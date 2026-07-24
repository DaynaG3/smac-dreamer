# Install and run Option-Critic v5 stability

The full commands are also included in the separate handoff supplied with the
ZIP.

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export PATCH="$ROOT/option_critic_v5_stability_hotfix_bundle"
```

Stop every active multimap trainer before applying the patch.

```bash
"$PY" "$PATCH/apply_option_critic_v5_stability_hotfix.py" \
  --repo "$REPO" --dry-run

"$PY" "$PATCH/apply_option_critic_v5_stability_hotfix.py" \
  --repo "$REPO"
```

Audit against the original Tactical Mixture v1.2 best checkpoint:

```bash
export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"
export SOURCE_CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$TACTICAL_V12_RUN/run_meta.json"

cd "$REPO"
REPO="$REPO" PY="$PY" \
CONFIG=configs/r2_2100_jepa_option_critic_2_v5_stability_1m.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" SOURCE_RUN_META="$SOURCE_RUN_META" \
bash scripts/static_audit_option_critic_v5_stability.sh
```

Launch the failure-isolated RL→Exp45 pipeline:

```bash
cd "$REPO"
bash scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh
```
