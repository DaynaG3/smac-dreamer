# Running DreamerV3 on SMAClite

## Current Phase 1 Status

Phase 1 has passed on the fixed `2s3z` scenario.

Validated configuration:

```text
scenario        : 2s3z
n_agents        : 5
n_enemies       : 5
n_actions       : 11
obs_size        : 80
state shape     : (400,)
avail_actions   : (55,)
action heads    : action_0, action_1, action_2, action_3, action_4
```

Validated runs:

```text
smoke test      : passed
500 steps       : passed
5,000 steps     : passed
10,000 steps    : passed
```

The implementation uses a centralised-controller formulation:

```text
one DreamerV3 agent controls all allied SMAClite units
```

Each allied unit receives one discrete action head.

---

## Command Prompt Setup

Run all commands from the project root:

```cmd
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

Confirm:

```cmd
echo %PYTHONPATH%
```

---

## Smoke Test

Run:

```cmd
python scripts\smoke_test_smaclite_env.py --scenario 2s3z
```

Expected result:

```text
All smoke tests PASSED.
```

The smoke test should confirm:

- reset works
- invalid actions do not crash
- full episode completes
- sequential reset works
- observation shapes remain stable
- `log/` metrics are scalar `float32`

---

## DreamerV3 Debug Training

### 500-Step Debug Run

```cmd
python scripts\train_dreamer_smaclite_phase1.py --configs debug --logdir logs\smaclite_phase1\debug --run.steps 500
```

### 5k-Step Stability Run

```cmd
python scripts\train_dreamer_smaclite_phase1.py --configs debug --logdir logs\smaclite_phase1\debug_5k --run.steps 5000
```

### 10k-Step Stability Run

```cmd
python scripts\train_dreamer_smaclite_phase1.py --configs debug --logdir logs\smaclite_phase1\debug_10k --run.steps 10000
```

The goal of these debug runs is not high win rate.

The goal is to verify:

- DreamerV3 creates the adapter
- the agent builds successfully
- multi-head discrete actions work
- replay accepts data
- training updates occur
- metrics are logged
- checkpoints are created
- no NaN/Inf values occur

---

## Verifying Training Output

### Check Log Directory

For the 10k run:

```cmd
dir logs\smaclite_phase1\debug_10k
```

Expected files/directories:

```text
config.yaml
metrics.jsonl
scores.jsonl
replay
ckpt
```

### Check Checkpoint Directory

```cmd
dir logs\smaclite_phase1\debug_10k\ckpt
```

Expected:

```text
latest
```

### Check Metrics File Exists

```cmd
python -c "import pathlib; p=pathlib.Path(r'logs\smaclite_phase1\debug_10k\metrics.jsonl'); print('exists:', p.exists()); print('size_bytes:', p.stat().st_size if p.exists() else 0); print('rows:', len(p.read_text().splitlines()) if p.exists() else 0)"
```

Expected:

```text
exists: True
size_bytes: > 0
rows: > 0
```

### Check for NaN/Inf

```cmd
python -c "import json,math,pathlib; p=pathlib.Path(r'logs\smaclite_phase1\debug_10k\metrics.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; bad=[]; [bad.append((i,k,v)) for i,row in enumerate(rows) for k,v in row.items() if isinstance(v,(int,float)) and (math.isnan(v) or math.isinf(v))]; print('rows:', len(rows)); print('nan_or_inf_count:', len(bad)); print('first_bad:', bad[:5])"
```

Expected:

```text
nan_or_inf_count: 0
first_bad: []
```

### Verify Required Metrics

```cmd
python -c "import json,pathlib; p=pathlib.Path(r'logs\smaclite_phase1\debug_10k\metrics.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; keys=set().union(*[r.keys() for r in rows]); wanted=['episode/score','episode/length','epstats/log/battle_won/avg','epstats/log/episode_invalid_action_count/sum','epstats/log/episode_invalid_action_rate/avg','epstats/log/episode_total_action_count/sum','replay/items','train/loss/policy','train/loss/value','train/ent/action_0','train/ent/action_4']; [print(k, 'FOUND' if k in keys else 'MISSING') for k in wanted]"
```

Expected:

```text
episode/score FOUND
episode/length FOUND
epstats/log/battle_won/avg FOUND
epstats/log/episode_invalid_action_count/sum FOUND
epstats/log/episode_invalid_action_rate/avg FOUND
epstats/log/episode_total_action_count/sum FOUND
replay/items FOUND
train/loss/policy FOUND
train/loss/value FOUND
train/ent/action_0 FOUND
train/ent/action_4 FOUND
```

Note: invalid-action metrics are named:

```text
episode_invalid_action_count
episode_invalid_action_rate
episode_total_action_count
```

not:

```text
invalid_action_count
invalid_action_rate
total_action_count
```

---

## Checkpoint Resume Test

Before starting Phase 2, verify that training can resume from an existing log directory.

Run:

```cmd
python scripts\train_dreamer_smaclite_phase1.py --configs debug --logdir logs\smaclite_phase1\debug_10k --run.steps 12000
```

Success criteria:

- the command does not crash
- the existing log directory is reused
- training continues beyond the previous run
- metrics continue to be appended
- no NaN/Inf values appear

After running, verify again:

```cmd
python -c "import json,pathlib; p=pathlib.Path(r'logs\smaclite_phase1\debug_10k\metrics.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; print('rows:', len(rows)); print('last row:', rows[-1])"
```

---

## Evaluation Script

This section covers Phase 1 fixed-scenario evaluation only.
Multi-map evaluation (Phase 2) is not yet implemented.

### Script

```text
scripts/evaluate.py
```

### Setup

```cmd
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

### Quick 2-episode test

Run this first to confirm checkpoint loading and env/agent wiring:

```cmd
python scripts\evaluate.py --logdir logs\smaclite_phase1\debug_10k --scenario 2s3z --episodes 2
```

### Full 10-episode evaluation

```cmd
python scripts\evaluate.py --logdir logs\smaclite_phase1\debug_10k --scenario 2s3z --episodes 10
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--logdir` | required | Path to a completed DreamerV3 training log directory |
| `--scenario` | `2s3z` | SMAClite scenario name |
| `--episodes` | `10` | Number of evaluation episodes |
| `--seed` | `42` | Environment random seed |
| `--max_episode_steps` | `200` | Episode step limit |
| `--deterministic` | `true` | Use `mode='eval'` (deterministic policy) |
| `--output` | `results/eval_smaclite_phase1.json` | JSON output path |

### Expected terminal output

```text
Logdir    : ...
Checkpoint: ...
Scenario  : 2s3z
Episodes  : 10
Seed      : 42
Mode      : eval
obs state shape : (400,)
act_space keys  : ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
Checkpoint loaded from: ...

Running 10 evaluation episode(s)...

  Episode   1/10: reward=...  length=...  WIN/loss
  ...

============================================================
Evaluation Summary
============================================================
  scenario                 : 2s3z
  episodes                 : 10
  mean_episode_reward      : ...
  std_episode_reward       : ...
  min_episode_reward       : ...
  max_episode_reward       : ...
  mean_episode_length      : ...
  win_rate                 : ...
  mean_invalid_action_count: ...
  mean_invalid_action_rate : ...
  mean_total_action_count  : ...
```

### Results JSON

Saved by default to:

```text
results/eval_smaclite_phase1.json
```

The `results/` directory is created automatically if it does not exist.

Check the file exists after evaluation:

```cmd
dir results
```

Inspect the JSON contents:

```cmd
python -c "import json,pathlib; d=json.loads(pathlib.Path('results/eval_smaclite_phase1.json').read_text()); print(json.dumps(d['aggregate'], indent=2))"
```

### JSON structure

```json
{
  "scenario": "2s3z",
  "logdir": "...",
  "checkpoint_path": "...",
  "episodes": 10,
  "seed": 42,
  "aggregate": {
    "mean_episode_reward": ...,
    "std_episode_reward": ...,
    "min_episode_reward": ...,
    "max_episode_reward": ...,
    "mean_episode_length": ...,
    "win_rate": ...,
    "mean_invalid_action_count": ...,
    "mean_invalid_action_rate": ...,
    "mean_total_action_count": ...
  },
  "episodes_data": [
    {
      "episode": 1,
      "reward": ...,
      "length": ...,
      "battle_won": false,
      "invalid_action_count": ...,
      "invalid_action_rate": ...,
      "total_action_count": ...
    }
  ]
}
```

### Notes

- Evaluation uses `mode='eval'` (deterministic policy) by default.
  Pass `--deterministic false` to use stochastic sampling instead.
- The evaluation script does not write to the replay buffer or update model parameters.
- The agent is reconstructed using the `config.yaml` saved inside `--logdir`,
  so the architecture matches the training run exactly.
- Phase 1 evaluation is fixed to a single scenario.
  Phase 2 multi-map evaluation is not yet implemented.

---

## Troubleshooting

### Pillow / PIL `_imaging` ImportError

Error:

```text
ImportError: cannot import name '_imaging' from 'PIL'
```

Cause:

The logger imports `scope`, which imports `PIL.Image`. This error means the local Pillow/PIL installation is broken.

Fix:

```cmd
conda activate smaclite-env
python -m pip uninstall -y pillow PIL
python -m pip install --no-cache-dir --force-reinstall pillow
```

Verify:

```cmd
python -c "import PIL; print('Pillow version:', PIL.__version__); from PIL import Image; print('PIL Image import OK')"
```

Alternative Conda fix:

```cmd
conda remove pillow -y
conda install -c conda-forge pillow -y
```

This is an environment/dependency issue, not an SMAClite adapter issue.

---

## Phase 2 Warning

Do not start Phase 2 until all of the following are true:

- smoke test passes
- 10k debug training passes
- metric verification passes
- checkpoint resume test passes
- evaluation script works

Phase 2 should only introduce same-shape multi-map support.

Do not add:

- padding
- variable unit counts
- large generated map datasets
- custom units
- Phase 3 curriculum logic