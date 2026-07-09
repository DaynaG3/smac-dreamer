# Policy visualisation (structured checkpoints)

Replay real SMAClite rollouts with a trained R2-Dreamer checkpoint to inspect **what executed
actions** the agent takes in winning and failing episodes. The tooling is purely additive — it
does not touch training, the replay buffer, validation selection, reward functions, or existing
evaluation metrics.

> **Structured checkpoints only.** If the checkpoint's `run_meta.json` reports
> `obs_mode != "structured"`, both scripts fail fast with a clear message. Flat-observation
> support is intentionally not implemented yet.

The agent + model are rebuilt with the **same logic as `scripts/evaluate_multimap.py`**: read
`run_meta.json` beside the checkpoint for `obs_mode` / `units` / `deter` / `batch_size` /
`batch_length` / `imag_horizon` / `padding`, build a probe env for the obs/action spaces,
construct `Dreamer`, propagate the device, load `agent_state_dict`, and `eval()`.

## Components

- `src/smacdreamer/visualization/trace.py` — pure helpers: action labels, target-focus metric,
  episode summary + classification (torch/pygame-free; unit-tested).
- `src/smacdreamer/visualization/render.py` — frame capture via the existing SMAClite Pygame
  renderer, text overlay, MP4 writing (imageio), and an optional live window.
- `src/smacdreamer/visualization/rollout.py` — checkpoint/agent reconstruction, single-map env
  construction, and the deterministic episode driver.
- `scripts/visualize_episode.py` — one (map, seed): interactive playback and/or MP4 + JSONL.
- `scripts/visualize_batch.py` — many (map × seed): select interesting episodes, record videos.

## Single-episode usage

```bash
python scripts/visualize_episode.py \
  --config configs/r2_650.yaml \
  --checkpoint logs/r2dreamer/r2_650/best_val_macro_winrate.pt \
  --split blind_iid \
  --map-name <map_name> \
  --seed 0 \
  --mode both \
  --output-dir results/replays
```

- `--map-name` is required unless `--maps-dir`/`--split` resolves to exactly one map.
- `--maps-dir <folder>` visualises an arbitrary maps folder instead of a config split.
- `--mode interactive | record | both`.
- `--run-meta <path>` overrides the default `checkpoint.parent/run_meta.json`.
- `--device cpu` forces CPU (default comes from the config, e.g. `cuda:0`).
- `--no-overlay` disables the on-frame text overlay.
- `--no-save-jsonl` / `--no-save-summary` disable those outputs (both on by default).

### Making the video easier to follow

Native SMAClite rendering is one frame per decision step and physically small, so raw video
plays very fast. Pacing/readability knobs (both scripts):

- `--fps` — playback frame rate. Default `8` (legible); pass `--fps 22.4` for realtime.
- `--scale` — nearest-neighbour upscale factor. Default `2` (bigger units + overlay text).
- `--hold-last-seconds` — freeze the final frame this long (default `1.5`) so the WIN/LOSS
  outcome doesn't flash by.

A slow, readable single episode:

```bash
python scripts/visualize_episode.py --config ... --checkpoint ... \
  --map-name <map> --seed 0 --mode record --headless \
  --fps 6 --scale 3 --hold-last-seconds 2.0 --output-dir results/replays
```

## Batch usage

```bash
python scripts/visualize_batch.py \
  --config configs/r2_650.yaml \
  --checkpoint logs/r2dreamer/r2_650/best_val_macro_winrate.pt \
  --split blind_compositional \
  --seeds 0,1,2 \
  --select failures \
  --max-videos 10 \
  --headless \
  --output-dir results/replays/blind_comp_failures
```

`--select` options:

- `wins` — `battle_won == true`
- `failures` — `battle_won == false`
- `low_enemy_damage` — lost **and** `final_enemy_ehp_frac >= --low-enemy-ehp-threshold` (0.75)
- `poor_target_focus` — lost **and** `mean_target_focus_score < --poor-focus-threshold` (0.5)
  **and** `attack_steps >= --min-attack-steps-for-focus` (5)
- `all` — up to `--max-videos` from all evaluated episodes

Batch runs in two passes: pass 1 evaluates every (map × seed) without frames (cheap) and
collects summaries; pass 2 reruns the selected episodes deterministically and records them.

## Headless usage (Kubeflow / Kaggle)

Pass `--headless` to set the SDL dummy drivers **before** Pygame initialises:

```python
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
```

MP4 recording works without a display. If `--mode interactive`/`both` is combined with
`--headless`, the live window is disabled (with a warning) and recording still proceeds.

Saving MP4 needs `imageio` with an ffmpeg backend; if missing you get a clear install hint
(`pip install imageio imageio-ffmpeg`).

## Output files

For each `(map, seed)` under `--output-dir`:

- `<map>_seed<seed>.mp4` — rendered episode (one frame at reset + one per step, with overlay).
- `<map>_seed<seed>.jsonl` — one row per env step (executed actions + per-step signals).
- `<map>_seed<seed>_summary.json` — episode-level summary + classification flags.

Batch additionally writes `batch_summary.json` listing all evaluated episodes, the selected
episodes, the selection reason, and the output paths.

### Per-step JSONL row

```json
{
  "step": 1,
  "map": "map_name",
  "seed": 0,
  "executed_actions": [6, 6, 7],
  "action_labels": ["ATTACK_0", "ATTACK_0", "ATTACK_1"],
  "reward": 0.0,
  "original_reward": 0.0,
  "battle_won": false,
  "enemies_alive": 2,
  "allies_alive": 3,
  "enemy_hp_damage_this_step": 0.0,
  "final_enemy_ehp_frac_if_available": null,
  "target_focus_score": 0.67
}
```

`executed_actions` are read from the env debug context (`last_executed_action`) — the actions
SMAClite actually ran, after masking/sanitisation — not the raw policy logits.

### Action labels

`0=NOOP`, `1=STOP`, `2=MOVE_N`, `3=MOVE_E`, `4=MOVE_S`, `5=MOVE_W`, `>=6 -> ATTACK_<action-6>`
(the `>=6` index is the SMAClite target slot id).

## Metric explanations

**Low enemy damage.** A *lost* episode where the enemy team still has most of its
effective HP at the end (`final_enemy_ehp_frac >= 0.75`). The agent failed to meaningfully
damage the enemy — a sign of passivity, poor engagement, or dying before dealing damage.

**Poor target focus.** Per step, `target_focus_score = most_common_target_count /
num_attack_actions` (fraction of attacks aimed at the single most-popular enemy that step);
`None` on steps with no attack. The episode mean over attack steps measures fire concentration.
A *lost* episode with mean focus `< 0.5` and at least 5 attack steps is flagged: the agent
attacked enough to judge, but spread its fire incoherently instead of focusing one target.

## Smoke check

Pure-logic tests (no checkpoint, no simulator):

```bash
python -m pytest tests/test_visualization.py -q
```

When a structured checkpoint is available, a one-episode end-to-end smoke run (substitute your
own checkpoint/map — no path is hard-coded):

```bash
python scripts/visualize_episode.py \
  --config <your_config.yaml> \
  --checkpoint <your_checkpoint.pt> \
  --map-name <a_small_map> --seed 0 --mode record --headless \
  --device cpu --output-dir results/replays/smoke
```
