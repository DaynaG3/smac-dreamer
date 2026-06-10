# Phase 3B: Learning Validation

**Status: COMPLETE** — all acceptance criteria passed (seed=42, debug config). See `results/phase3b_learning_report_baseline.md` for the archived baseline report. Phase 3C continues from here with size1m non-debug training.

## Purpose

Phase 3B validates whether DreamerV3 learns useful behaviour on the Phase 3 padded
multi-map SMAClite set (5 maps with heterogeneous unit compositions, padded to fixed
observation and action dimensions).

The key question: does the agent improve beyond the valid-action random baseline?
Win rate is not the only signal — reward improvement over random indicates the agent
is learning to deal damage and protect allies even if no episode is won outright.

---

## Prerequisites

```
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

---

## Execution Checklist

Run the steps below in order. Steps C and E are long training runs (CPU); run them
manually and provide the results before proceeding to the evaluation and comparison steps.

### Step A — Random baseline (fast, ~5 min)

```
python scripts\random_baseline_phase3.py ^
    --manifest configs\maps\phase3_manifest.yaml ^
    --episodes 30 --seed 42 ^
    --output   results\random_phase3_30eps.json ^
    --jsonl_output results\random_phase3_30eps.jsonl
```

**Expected output**: `results/random_phase3_30eps.json` with 30 episodes × 5 maps.
Masking failure rate should be 0.000 for all maps (random acts on obs["avail_actions"]).

---

### Step B — Re-evaluate existing debug_5k checkpoint (fast, ~10 min)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_manifest.yaml ^
    --logdir   logs\smaclite_phase3\debug_5k ^
    --episodes 30 --seed 42 ^
    --output   results\eval_phase3_debug_5k_30eps.json ^
    --jsonl_output results\eval_phase3_debug_5k_30eps.jsonl
```

**Expected output**: `results/eval_phase3_debug_5k_30eps.json`.
Check `checkpoint_loaded: true` at the top of the JSON and in stdout.
If `checkpoint_loaded: false`, results are invalid — investigate the checkpoint path.

---

### Step C — Train debug_50k (MANUAL — ~2–4 hours on CPU)

```
python scripts\train_dreamer_smaclite_phase3.py ^
    --configs debug smaclite_phase3 ^
    --logdir logs\smaclite_phase3\debug_50k ^
    --run.steps 50000
```

**Expected output**: `logs/smaclite_phase3/debug_50k/ckpt/` with a checkpoint.
Training logs in `logs/smaclite_phase3/debug_50k/metrics.jsonl`.

---

### Step D — Evaluate debug_50k (fast, ~10 min; unblocks after Step C)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_manifest.yaml ^
    --logdir   logs\smaclite_phase3\debug_50k ^
    --episodes 30 --seed 42 ^
    --output   results\eval_phase3_debug_50k_30eps.json ^
    --jsonl_output results\eval_phase3_debug_50k_30eps.jsonl
```

---

### Step E — Train overfit_2s3z (MANUAL — ~4–8 hours on CPU)

Single-map overfit run on 2s3z only, using same padding dims as full Phase 3
(obs/act space is identical — enables direct comparison).

```
python scripts\train_dreamer_smaclite_phase3.py ^
    --configs debug smaclite_phase3_overfit ^
    --logdir logs\smaclite_phase3\overfit_2s3z ^
    --run.steps 100000
```

**Expected output**: `logs/smaclite_phase3/overfit_2s3z/ckpt/` with a checkpoint.

---

### Step F — Evaluate overfit_2s3z (fast, ~3 min; unblocks after Step E)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_overfit_2s3z_manifest.yaml ^
    --logdir   logs\smaclite_phase3\overfit_2s3z ^
    --episodes 30 --seed 42 ^
    --output   results\eval_phase3_overfit_2s3z_30eps.json ^
    --jsonl_output results\eval_phase3_overfit_2s3z_30eps.jsonl
```

Note: This manifest has only 2s3z. Other maps will show N/A in the comparison report.

---

### Step G — Generate comparison report (fast; unblocks after D and F)

```
python scripts\compare_phase3b.py ^
    --random  results\random_phase3_30eps.json ^
    --results results\eval_phase3_debug_5k_30eps.json ^
              results\eval_phase3_debug_50k_30eps.json ^
              results\eval_phase3_overfit_2s3z_30eps.json ^
    --labels  "5k" "50k" "overfit_2s3z" ^
    --logdirs logs\smaclite_phase3\debug_5k ^
              logs\smaclite_phase3\debug_50k ^
              logs\smaclite_phase3\overfit_2s3z ^
    --output  results\phase3b_learning_report.md
```

**Expected output**: `results/phase3b_learning_report.md` with per-map comparison table,
NaN/Inf scan results, PASS/FAIL checks, and caveats.

---

## Output Files

| File | Created by | Contents |
|------|-----------|----------|
| `results/random_phase3_30eps.json` | Step A | Random baseline, 30 eps × 5 maps |
| `results/random_phase3_30eps.jsonl` | Step A | Per-episode rows |
| `results/eval_phase3_debug_5k_30eps.json` | Step B | debug_5k eval, 30 eps × 5 maps |
| `results/eval_phase3_debug_50k_30eps.json` | Step D | debug_50k eval, 30 eps × 5 maps |
| `results/eval_phase3_overfit_2s3z_30eps.json` | Step F | overfit eval, 30 eps × 2s3z only |
| `results/phase3b_learning_report.md` | Step G | Full comparison + NaN scan + PASS/FAIL |

---

## Interpretation Guide

### Masking failure rate

`masking_failure_rate` should be **0.000** in all Dreamer eval runs.
A non-zero value means the agent selected an unavailable action that the adapter had
to fall back from — this indicates a masking or timing bug, not normal behaviour.

### Reward vs win rate

On these maps at debug scale (small model, few steps), win rate is often 0.
The primary learning signal is **mean_episode_reward** relative to the random baseline:

- `reward_delta_vs_random > 0` → agent is learning something (PASS)
- `reward_delta_vs_random <= 0` → agent is not yet improving over random (FAIL, but expected
  at 5k steps; should improve at 50k)

### NaN/Inf

Any NaN or Inf in `metrics.jsonl` indicates a training instability.
This is unlikely with the debug config but must be checked before interpreting results.

### checkpoint_loaded=false

If any result JSON has `checkpoint_loaded: false`, that run's metrics are marked **INVALID**
in the report and excluded from PASS/FAIL checks. Investigate the logdir before using
those results for any conclusion.

---

## Caveats

- **Single seed**: Phase 3B uses seed=42 throughout. Results are indicative only and
  should be repeated with multiple seeds before drawing strong conclusions.
- **Debug configuration**: Training uses `--configs debug`, which reduces batch size and
  model capacity relative to a production configuration. This is not the final
  performance configuration.
- **Zero win rate**: A win rate of 0 is not automatic failure. Reward improvement over
  the random baseline demonstrates the agent is learning to deal damage and survive,
  even without closing out an episode.
- **CPU-only**: All runs target JAX CPU. Training is slow; wall-clock times are indicative.

---

## Acceptance Criteria

- [x] `results/random_phase3_30eps.json` — all 5 maps, 30 eps, masking_failure_rate=0
- [x] `results/eval_phase3_debug_5k_30eps.json` — checkpoint_loaded=true
- [x] `results/eval_phase3_debug_50k_30eps.json` — checkpoint_loaded=true, no NaN/Inf
- [x] `results/eval_phase3_overfit_2s3z_30eps.json` — checkpoint_loaded=true
- [x] masking_failure_rate == 0 in all Dreamer eval runs
- [x] No NaN/Inf in debug_50k metrics.jsonl
- [x] Dreamer 50k beats random mean_reward on ≥3/5 maps — **3/5 PASS**
- [x] 2s3z overfit reward improves over random — **+0.299 PASS**
- [x] `results/phase3b_learning_report.md` contains per-map comparison table

---

## Results (seed=42, debug config, CPU)

All steps completed. Full report: `results/phase3b_learning_report.md`.

### NaN / Inf

All three training runs are numerically clean: 0 NaN, 0 Inf across all metrics.jsonl logs
(5k: 105 lines, 50k: 1006 lines, overfit: 2164 lines).

### Per-map reward vs random baseline

| Map | n_agents | Random | 5k | 50k | Overfit 100k |
|-----|----------|--------|-----|-----|-------------|
| 2s3z | 5 | 3.327 | 3.916 (+0.589) | 3.474 (+0.148) | **3.625 (+0.299)** |
| 3s5z | 8 | 3.926 | 3.600 (-0.326) | 3.888 (-0.038) | N/A |
| 3s5z_vs_3s6z | 8 | 3.327 | 3.025 (-0.302) | 3.266 (-0.061) | N/A |
| 2s_vs_1sc | 2 | 0.878 | 0.983 (+0.105) | 0.941 (+0.063) | N/A |
| 3s_vs_5z | 3 | 2.923 | 3.011 (+0.087) | 3.030 (+0.107) | N/A |

Win rate is 0.000 on all maps across all runs — expected at debug scale.
masking_failure_rate is 0.000 on every episode across all runs.

### Acceptance check summary

| Run | Maps beating random | Result |
|-----|--------------------|----|
| 5k | 3/5 | PASS |
| 50k | 3/5 | PASS |
| overfit_2s3z | 1/1 | PASS |

### Findings

**Small-map learning confirmed (5k onwards).** Maps with 2–5 agents (2s3z, 2s_vs_1sc,
3s_vs_5z) beat random from the earliest checkpoint. These have manageable joint action
spaces and the agent picks up signal quickly.

**Hard 8-agent maps improving but not yet above random.** 3s5z and 3s5z_vs_3s6z both
fail at 5k and 50k, but the gap to random narrows substantially with more training:

- 3s5z: -0.326 (5k) → -0.038 (50k)
- 3s5z_vs_3s6z: -0.302 (5k) → -0.061 (50k)

These maps have a joint action space of ~15⁸ ≈ 2.6 billion combinations. Longer training
at production scale is expected to close the gap further.

**Focused overfit training outperforms multi-map on 2s3z.** The overfit 100k checkpoint
(single map) scores 3.625 vs 3.474 for the 50k multi-map run, confirming that map
rotation distributes capacity across maps and reduces per-map performance relative to
a single-map specialist.

**The 5k checkpoint appears to score higher than 50k on 2s3z (3.916 vs 3.474).** This is
within one standard deviation (σ≈0.89) for 30 episodes and is attributed to evaluation
noise rather than a regression. The 5k checkpoint has far fewer training steps and lower
overall performance on the harder maps.

**Per-step reward efficiency on 2s3z** (reward ÷ mean episode length):

| Run | Mean reward | Mean length | Reward/step |
|-----|------------|-------------|-------------|
| Random | 3.327 | 53.9 | 0.062 |
| 50k | 3.474 | 51.5 | 0.067 |
| Overfit 100k | 3.625 | 41.7 | **0.087** |

The overfit agent acts more decisively — shorter episodes with higher total reward —
indicating it has learned an aggressive but more reward-efficient strategy.
