# Phase 3D — First-Win Experiment: Original Reward Only

## Objective

Determine whether DreamerV3 can achieve wins (`battle_won > 0`) on the easiest
Phase 3 map using only the original SMAClite team reward, with no reward shaping.

This is a clean baseline before any reward engineering.

## Map used

```
2s_vs_1sc
```

2 allied Stalkers vs 1 enemy Spine Crawler.
This is the simplest Phase 3 map by allied-unit count and enemy difficulty.

## Config used

Config block: `smaclite_phase3d_2s_vs_1sc_original`  
Config file: `configs/smaclite_phase3.yaml`  
Manifest: `configs/maps/phase3d_overfit_2s_vs_1sc_manifest.yaml`

Padding dimensions (identical to all Phase 3 runs):

| Parameter      | Value |
|----------------|-------|
| max_agents     | 8     |
| max_enemies    | 9     |
| max_actions    | 15    |
| max_obs_size   | 136   |

Key hyperparameters:

| Parameter    | Value   |
|--------------|---------|
| steps        | 1000000 |
| train_ratio  | 8       |
| log_every    | 60 s    |
| save_every   | 300 s   |
| envs         | 1       |
| map_mode     | fixed   |

**Reward**: Original SMAClite team reward only. No shaping applied.

## Manual commands

### Setup

```cmd
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

### Smoke-test the manifest

```cmd
python scripts\smoke_test_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml
```

### Train for 1M steps

```cmd
python scripts\train_dreamer_smaclite_phase3.py --configs smaclite_phase3d_2s_vs_1sc_original size1m --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_original_1m --run.steps 1000000
```

### Evaluate (50 episodes)

```cmd
python scripts\evaluate_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_original_1m --episodes 50 --seed 42 --output results\eval_phase3d_2s_vs_1sc_original_1m.json --jsonl_output results\eval_phase3d_2s_vs_1sc_original_1m.jsonl
```

### Check NaN/Inf and masking metrics

```cmd
python -c "import json,pathlib,math; p=pathlib.Path(r'logs\smaclite_phase3d\overfit_2s_vs_1sc_original_1m\metrics.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; bad=[]; [bad.append((k,v)) for r in rows for k,v in r.items() if isinstance(v,float) and (math.isnan(v) or math.isinf(v))]; mf=[r.get('epstats/log/masking_failure_count/sum') for r in rows if 'epstats/log/masking_failure_count/sum' in r]; print('rows:', len(rows)); print('nan_inf_count:', len(bad)); print('masking_failure_total:', sum(x for x in mf if x is not None)); print('bad examples:', bad[:10])"
```

### Inspect evaluation JSON

```cmd
python -c "import json,pathlib; p=pathlib.Path(r'results\eval_phase3d_2s_vs_1sc_original_1m.json'); d=json.loads(p.read_text()); print('checkpoint_loaded:', d.get('checkpoint_loaded')); print('maps:', list(d.get('maps', {}).keys())); print(json.dumps(d.get('aggregate', {}), indent=2))"
```

## Expected outputs

- `logs/smaclite_phase3d/overfit_2s_vs_1sc_original_1m/metrics.jsonl` — training metrics
- `logs/smaclite_phase3d/overfit_2s_vs_1sc_original_1m/checkpoint/` — saved checkpoint
- `results/eval_phase3d_2s_vs_1sc_original_1m.json` — evaluation summary
- `results/eval_phase3d_2s_vs_1sc_original_1m.jsonl` — per-episode evaluation rows

## Interpretation guide

| Result | Interpretation |
|--------|----------------|
| `win_rate > 0` | DreamerV3 can win on the easiest map with original reward — Phase 3D success |
| `win_rate == 0` but reward > random baseline | Agent is learning something but not closing out wins — consider reward shaping next |
| `win_rate == 0` and reward ≈ random baseline | Agent is not learning; investigate model capacity, learning rate, replay ratio |
| `masking_failure_rate > 0` | Regression in masking — investigate before continuing |
| NaN/Inf in metrics | Numerical instability — reduce learning rate or check reward scale |

## Reward shaping policy

This run uses the **original SMAClite reward only**.

Reward shaping (e.g. kill reward, distance reward, health-delta reward) will only
be considered **after** this baseline completes and its win rate is documented.

If `win_rate == 0` after this 1M-step run, the next experiment will add a
conservative kill-reward shaping term and re-run on the same map.
