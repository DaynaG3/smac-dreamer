# Phase 3B Learning Validation Report

Comparison of DreamerV3 checkpoints against the valid-action random baseline on the Phase 3 padded multi-map SMAClite set.

## Run Metadata

- **Random baseline**: `configs\maps\phase3_manifest.yaml`  episodes=30  seed=42
- **5k**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3\debug_5k`  checkpoint_loaded=True
- **50k**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3\debug_50k`  checkpoint_loaded=True
- **overfit_2s3z**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3\overfit_2s3z`  checkpoint_loaded=True

## NaN / Inf Scan

| Run | metrics.jsonl | Lines | NaN | Inf | Status |
|-----|--------------|-------|-----|-----|--------|
| 5k | `metrics.jsonl` | 105 | 0 | 0 | clean |
| 50k | `metrics.jsonl` | 1006 | 0 | 0 | clean |
| overfit_2s3z | `metrics.jsonl` | 2164 | 0 | 0 | clean |

## Per-Map Comparison

| Map | Rand win | Rand reward | 5k win | 5k reward | 5k Δreward | 5k maskfail | 50k win | 50k reward | 50k Δreward | 50k maskfail | overfit_2s3z win | overfit_2s3z reward | overfit_2s3z Δreward | overfit_2s3z maskfail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2s3z | 0.000 | 3.327 | 0.000 | 3.916 | +0.589 | 0.0000 | 0.000 | 3.474 | +0.148 | 0.0000 | 0.000 | 3.625 | +0.299 | 0.0000 |
| 3s5z | 0.000 | 3.926 | 0.000 | 3.600 | -0.326 | 0.0000 | 0.000 | 3.888 | -0.038 | 0.0000 | N/A | N/A | N/A | N/A |
| 3s5z_vs_3s6z | 0.000 | 3.327 | 0.000 | 3.025 | -0.302 | 0.0000 | 0.000 | 3.266 | -0.061 | 0.0000 | N/A | N/A | N/A | N/A |
| 2s_vs_1sc | 0.000 | 0.878 | 0.000 | 0.983 | +0.105 | 0.0000 | 0.000 | 0.941 | +0.063 | 0.0000 | N/A | N/A | N/A | N/A |
| 3s_vs_5z | 0.000 | 2.923 | 0.000 | 3.011 | +0.087 | 0.0000 | 0.000 | 3.030 | +0.107 | 0.0000 | N/A | N/A | N/A | N/A |
| **OVERALL** | 0.000 | 2.876 | 0.000 | 2.907 | +0.031 | 0.0000 | 0.000 | 2.920 | +0.044 | 0.0000 | 0.000 | 3.625 | +0.749 | 0.0000 |

## Acceptance Checks

Checks whether each valid Dreamer run beats random mean_reward per map.

### 5k

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | 3.327 | 3.916 | +0.589 | **PASS** |
| 3s5z | 3.926 | 3.600 | -0.326 | **FAIL** |
| 3s5z_vs_3s6z | 3.327 | 3.025 | -0.302 | **FAIL** |
| 2s_vs_1sc | 0.878 | 0.983 | +0.105 | **PASS** |
| 3s_vs_5z | 2.923 | 3.011 | +0.087 | **PASS** |

**Summary**: 3/5 maps beat random mean_reward

### 50k

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | 3.327 | 3.474 | +0.148 | **PASS** |
| 3s5z | 3.926 | 3.888 | -0.038 | **FAIL** |
| 3s5z_vs_3s6z | 3.327 | 3.266 | -0.061 | **FAIL** |
| 2s_vs_1sc | 0.878 | 0.941 | +0.063 | **PASS** |
| 3s_vs_5z | 2.923 | 3.030 | +0.107 | **PASS** |

**Summary**: 3/5 maps beat random mean_reward

### overfit_2s3z

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | 3.327 | 3.625 | +0.299 | **PASS** |
| 3s5z | N/A | N/A | N/A | N/A |
| 3s5z_vs_3s6z | N/A | N/A | N/A | N/A |
| 2s_vs_1sc | N/A | N/A | N/A | N/A |
| 3s_vs_5z | N/A | N/A | N/A | N/A |

**Summary**: 1/1 maps beat random mean_reward  (4 N/A)

## Caveats

- **Single seed**: Phase 3B uses a single random seed (seed=42). Results are indicative only unless repeated with multiple seeds.
- **Debug configuration**: Training used `--configs debug`, which reduces batch size and model capacity compared to a production configuration. These results are not the final performance configuration.
- **Zero win rate**: A win rate of 0 is not automatic failure. Reward improvement over the random baseline shows the agent is learning to reduce casualties and deal damage even if no episode is won outright.
- **INVALID runs**: Any run where `checkpoint_loaded=false` indicates the checkpoint could not be loaded. Metrics from such runs are not valid for learning comparison and are marked INVALID in all tables.
