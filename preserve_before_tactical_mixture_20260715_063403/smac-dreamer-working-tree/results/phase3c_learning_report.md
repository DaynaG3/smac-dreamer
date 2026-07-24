# Phase 3B Learning Validation Report

Comparison of DreamerV3 checkpoints against the valid-action random baseline on the Phase 3 padded multi-map SMAClite set.

## Run Metadata

- **Random baseline**: `configs\maps\phase3_manifest.yaml`  episodes=30  seed=42
- **size1m_200k**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3c\size1m_200k`  checkpoint_loaded=True
- **overfit_3s5z**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3c\overfit_3s5z`  checkpoint_loaded=True
- **overfit_3s5z_vs_3s6z**: logdir=`C:\Users\gsimru\Documents\smac-dreamer\logs\smaclite_phase3c\overfit_3s5z_vs_3s6z`  checkpoint_loaded=True

## NaN / Inf Scan

| Run | metrics.jsonl | Lines | NaN | Inf | Status |
|-----|--------------|-------|-----|-----|--------|
| size1m_200k | `metrics.jsonl` | 3934 | 0 | 0 | clean |
| overfit_3s5z | `metrics.jsonl` | 4005 | 0 | 0 | clean |
| overfit_3s5z_vs_3s6z | `metrics.jsonl` | 4317 | 0 | 0 | clean |

## Per-Map Comparison

| Map | Rand win | Rand reward | size1m_200k win | size1m_200k reward | size1m_200k Δreward | size1m_200k maskfail | overfit_3s5z win | overfit_3s5z reward | overfit_3s5z Δreward | overfit_3s5z maskfail | overfit_3s5z_vs_3s6z win | overfit_3s5z_vs_3s6z reward | overfit_3s5z_vs_3s6z Δreward | overfit_3s5z_vs_3s6z maskfail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2s3z | 0.000 | 3.327 | 0.000 | 3.623 | +0.296 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 3s5z | 0.000 | 3.926 | 0.000 | 3.791 | -0.135 | 0.0000 | 0.000 | 3.738 | -0.189 | 0.0000 | N/A | N/A | N/A | N/A |
| 3s5z_vs_3s6z | 0.000 | 3.327 | 0.000 | 3.391 | +0.065 | 0.0000 | N/A | N/A | N/A | N/A | 0.000 | 3.381 | +0.054 | 0.0000 |
| 2s_vs_1sc | 0.000 | 0.878 | 0.000 | 0.916 | +0.038 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 3s_vs_5z | 0.000 | 2.923 | 0.000 | 2.929 | +0.006 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **OVERALL** | 0.000 | 2.876 | 0.000 | 2.930 | +0.054 | 0.0000 | 0.000 | 3.738 | +0.861 | 0.0000 | 0.000 | 3.381 | +0.504 | 0.0000 |

## Acceptance Checks

Checks whether each valid Dreamer run beats random mean_reward per map.

### size1m_200k

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | 3.327 | 3.623 | +0.296 | **PASS** |
| 3s5z | 3.926 | 3.791 | -0.135 | **FAIL** |
| 3s5z_vs_3s6z | 3.327 | 3.391 | +0.065 | **PASS** |
| 2s_vs_1sc | 0.878 | 0.916 | +0.038 | **PASS** |
| 3s_vs_5z | 2.923 | 2.929 | +0.006 | **PASS** |

**Summary**: 4/5 maps beat random mean_reward

### overfit_3s5z

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | N/A | N/A | N/A | N/A |
| 3s5z | 3.926 | 3.738 | -0.189 | **FAIL** |
| 3s5z_vs_3s6z | N/A | N/A | N/A | N/A |
| 2s_vs_1sc | N/A | N/A | N/A | N/A |
| 3s_vs_5z | N/A | N/A | N/A | N/A |

**Summary**: 0/1 maps beat random mean_reward  (4 N/A)

### overfit_3s5z_vs_3s6z

| Map | Random reward | Dreamer reward | Δreward | Result |
|-----|--------------|----------------|---------|--------|
| 2s3z | N/A | N/A | N/A | N/A |
| 3s5z | N/A | N/A | N/A | N/A |
| 3s5z_vs_3s6z | 3.327 | 3.381 | +0.054 | **PASS** |
| 2s_vs_1sc | N/A | N/A | N/A | N/A |
| 3s_vs_5z | N/A | N/A | N/A | N/A |

**Summary**: 1/1 maps beat random mean_reward  (4 N/A)

## Caveats

- **Single seed**: Phase 3B uses a single random seed (seed=42). Results are indicative only unless repeated with multiple seeds.
- **Debug configuration**: Training used `--configs debug`, which reduces batch size and model capacity compared to a production configuration. These results are not the final performance configuration.
- **Zero win rate**: A win rate of 0 is not automatic failure. Reward improvement over the random baseline shows the agent is learning to reduce casualties and deal damage even if no episode is won outright.
- **INVALID runs**: Any run where `checkpoint_loaded=false` indicates the checkpoint could not be loaded. Metrics from such runs are not valid for learning comparison and are marked INVALID in all tables.
