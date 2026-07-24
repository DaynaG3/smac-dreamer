```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export PATCH="$ROOT/v9_three_stage_pipeline_patch"

"$PY" "$PATCH/apply_v9_three_stage_pipeline_patch.py" --repo "$REPO" --dry-run
"$PY" "$PATCH/apply_v9_three_stage_pipeline_patch.py" --repo "$REPO"

export TACTICAL_V12_RUN="$(cat "$ROOT/CURRENT_TACTICAL_V1_2_RUN.txt")"
export SOURCE_CHECKPOINT="$TACTICAL_V12_RUN/best_val_macro_winrate.pt"
cd "$REPO"
REPO="$REPO" PY="$PY" CHECKPOINT="$SOURCE_CHECKPOINT" \
EXPECTED_SOURCE_CHECKPOINT_SHA256=74875c693150d4cd21be27201e332cb0d8d4f6648c10701761154dcd6588d99e \
bash scripts/static_audit_actor_critic_h15_800k.sh

bash scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```
