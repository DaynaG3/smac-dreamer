# Phase 3C: Non-Debug Training Validation

**Status: COMPLETE** — see `results/phase3c_learning_report.md` for full output.

## Purpose

Phase 3C tests the same Phase 3 padded multi-map set as Phase 3B but with a real model
size (`size1m`: deter=512, units=64, depth=4) instead of the debug toy model
(deter=8, units=8, depth=2). The goal is to establish a non-debug performance baseline
and determine whether the agent can beat random on the hard 8-agent maps (3s5z and
3s5z_vs_3s6z) that Phase 3B could not crack.

Hard-map overfit runs are added specifically to isolate per-map capacity for 3s5z and
3s5z_vs_3s6z. Evaluation uses 50 episodes per map (up from 30 in Phase 3B) for
tighter reward estimates.

**Phase 3B baseline**: `results/phase3b_learning_report_baseline.md`
Phase 3B result: 3/5 maps beat random (debug 50k). Hard maps missed by 0.04–0.06 reward.

---

## Prerequisites

```
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

---

## Execution Checklist

Steps A, B, C are independent — run in parallel in separate terminals.
D unblocks after A. E unblocks after B. F unblocks after C. G unblocks after D+E+F.

### Step A — Train size1m 5-map (MANUAL — long on CPU)

```
python scripts\train_dreamer_smaclite_phase3.py ^
    --configs size1m smaclite_phase3c ^
    --logdir logs\smaclite_phase3c\size1m_200k ^
    --run.steps 200000
```

**Expected output**: `logs/smaclite_phase3c/size1m_200k/ckpt/` with checkpoint.
Training log: `logs/smaclite_phase3c/size1m_200k/metrics.jsonl`.

---

### Step B — Train size1m overfit 3s5z (MANUAL — long on CPU)

```
python scripts\train_dreamer_smaclite_phase3.py ^
    --configs size1m smaclite_phase3c_overfit_3s5z ^
    --logdir logs\smaclite_phase3c\overfit_3s5z ^
    --run.steps 200000
```

---

### Step C — Train size1m overfit 3s5z_vs_3s6z (MANUAL — long on CPU)

```
python scripts\train_dreamer_smaclite_phase3.py ^
    --configs size1m smaclite_phase3c_overfit_3s5z_vs_3s6z ^
    --logdir logs\smaclite_phase3c\overfit_3s5z_vs_3s6z ^
    --run.steps 200000
```

---

### Step D — Evaluate size1m 5-map (50 eps; after Step A)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_manifest.yaml ^
    --logdir   logs\smaclite_phase3c\size1m_200k ^
    --episodes 50 --seed 42 ^
    --output   results\eval_phase3c_size1m_200k_50eps.json ^
    --jsonl_output results\eval_phase3c_size1m_200k_50eps.jsonl
```

Check `checkpoint_loaded: true`. If false, results are invalid.

---

### Step E — Evaluate overfit 3s5z (50 eps; after Step B)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_overfit_3s5z_manifest.yaml ^
    --logdir   logs\smaclite_phase3c\overfit_3s5z ^
    --episodes 50 --seed 42 ^
    --output   results\eval_phase3c_overfit_3s5z_50eps.json ^
    --jsonl_output results\eval_phase3c_overfit_3s5z_50eps.jsonl
```

---

### Step F — Evaluate overfit 3s5z_vs_3s6z (50 eps; after Step C)

```
python scripts\evaluate_phase3.py ^
    --manifest configs\maps\phase3_overfit_3s5z_vs_3s6z_manifest.yaml ^
    --logdir   logs\smaclite_phase3c\overfit_3s5z_vs_3s6z ^
    --episodes 50 --seed 42 ^
    --output   results\eval_phase3c_overfit_3s5z_vs_3s6z_50eps.json ^
    --jsonl_output results\eval_phase3c_overfit_3s5z_vs_3s6z_50eps.jsonl
```

---

### Step G — Generate comparison report (after D+E+F)

Uses the existing Phase 3B random baseline (30 eps/map) for delta computation.
Episode counts do not need to match — deltas are computed from aggregate means.

```
python scripts\compare_phase3b.py ^
    --random  results\random_phase3_30eps.json ^
    --results results\eval_phase3c_size1m_200k_50eps.json ^
              results\eval_phase3c_overfit_3s5z_50eps.json ^
              results\eval_phase3c_overfit_3s5z_vs_3s6z_50eps.json ^
    --labels  "size1m_200k" "overfit_3s5z" "overfit_3s5z_vs_3s6z" ^
    --logdirs logs\smaclite_phase3c\size1m_200k ^
              logs\smaclite_phase3c\overfit_3s5z ^
              logs\smaclite_phase3c\overfit_3s5z_vs_3s6z ^
    --output  results\phase3c_learning_report.md
```

---

## Output Files

| File | Created by | Contents |
|------|-----------|----------|
| `results/eval_phase3c_size1m_200k_50eps.json` | Step D | size1m 5-map, 50 eps × 5 maps |
| `results/eval_phase3c_overfit_3s5z_50eps.json` | Step E | overfit 3s5z, 50 eps |
| `results/eval_phase3c_overfit_3s5z_vs_3s6z_50eps.json` | Step F | overfit 3s5z_vs_3s6z, 50 eps |
| `results/phase3c_learning_report.md` | Step G | Full comparison + NaN scan + PASS/FAIL |

---

## Interpretation Guide

### Masking failure rate

`masking_failure_rate` must be **0.000** in all runs. Non-zero indicates a masking bug.

### Reward vs win rate

Win rate at size1m 200k on CPU may still be 0 on hard maps. Use reward delta vs random
as the primary signal. Size1m has 64× larger RSSM (deter=512 vs 8) — expect meaningfully
better sample efficiency than debug.

Phase 3B deltas for reference (debug 50k):
- 3s5z: −0.038 vs random (just below)
- 3s5z_vs_3s6z: −0.061 vs random (just below)

Phase 3C target: both hard maps positive with size1m at 200k steps.

### NaN/Inf

Check before interpreting results. Larger models are slightly more susceptible to
instability at initialisation, though rare with DreamerV3's symlog transforms.

### checkpoint_loaded=false

Results from a run with checkpoint_loaded=false are INVALID and excluded from PASS/FAIL
checks. Investigate the logdir before using those numbers.

---

## Caveats

- **Single seed**: Phase 3C uses seed=42 throughout. Results are indicative only.
- **size1m is still a small model**: Production DreamerV3 runs typically use size12m or
  larger. size1m is ~1M parameters; size12m is ~12M. This is a stepping-stone, not a
  final configuration.
- **CPU-only**: Training is slow. Wall-clock times for 200k steps at size1m will be
  significantly longer than the debug runs (likely 8–24 hours per run).
- **Zero win rate**: Not automatic failure — reward improvement over random demonstrates
  the agent is learning.

---

## Acceptance Criteria

- [ ] `results/eval_phase3c_size1m_200k_50eps.json` — checkpoint_loaded=true
- [ ] `results/eval_phase3c_overfit_3s5z_50eps.json` — checkpoint_loaded=true
- [ ] `results/eval_phase3c_overfit_3s5z_vs_3s6z_50eps.json` — checkpoint_loaded=true
- [ ] masking_failure_rate == 0 in all Phase 3C eval runs
- [ ] No NaN/Inf in any size1m metrics.jsonl
- [ ] size1m 200k beats random on ≥4/5 maps
- [ ] overfit_3s5z beats random mean_reward on 3s5z
- [ ] overfit_3s5z_vs_3s6z beats random mean_reward on 3s5z_vs_3s6z
- [ ] `results/phase3c_learning_report.md` generated with per-map table
