# Phase 3D Reward Shaping v2

## Why the original reward failed

After 1M training steps on `2s_vs_1sc`, DreamerV3 converged to a deterministic losing strategy:

| Metric | Observed |
|--------|----------|
| `win_rate` | **0.0** |
| `mean_episode_reward` | **8.696** |
| `mean_episode_length` | **32.0** |

Both Stalkers charged the Spine Crawler, dealt approximately 43% HP damage, and died at step 32 on every episode. The agent never won.

**Root cause:** The original SMAClite reward is `damage_dealt + 10*kills + 200*win`, normalised to `[0, 20]`. This gives a large one-time signal for winning, but dying quickly with some damage dealt produces a stable 8.696 per episode that is easy for the world model to predict — it became a local optimum that the agent never escaped.

## Reward shaping v2

The v2 system adds auxiliary signals on top of the original env reward. The policy sees `shaped_reward`. The original reward is preserved in `log/original_env_reward` for monitoring and evaluation.

### Formula (per step)

```
shaped_reward = original_env_reward
              + win_bonus          (terminal win step only)
              + loss_penalty       (terminal loss step only; negative)
              + kill_delta * enemy_kill_bonus
              + ally_deaths * ally_death_penalty   (negative)
              + n_alive_allies * ally_survival_bonus
              - step_penalty
```

**Terminal definitions:**
- `terminal_win` = `terminated AND battle_won`
- `terminal_loss` = `terminated AND NOT battle_won AND NOT truncated`
- Truncation (time limit) is neither win nor loss — no terminal bonus applied.

## v2 config values for `2s_vs_1sc`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `win_bonus` | 10.0 | Strong terminal signal to overcome the 8.7 loss-with-damage equilibrium |
| `loss_penalty` | -10.0 | Penalises dying; magnitude matches win_bonus |
| `enemy_kill_bonus` | 5.0 | Rewards eliminating the Spine Crawler (worth ~half of win_bonus) |
| `ally_death_penalty` | -2.0 | Penalises each Stalker death; discourages suicidal rushes |
| `ally_survival_bonus` | 0.02 | Small per-step per-alive-ally bonus; rewards staying alive |
| `step_penalty` | 0.003 | Mild time pressure; 2 alive allies → net +0.037/step before combat |
| `damage_delta_scale` | 0.0 | Reserved; not used |

**Net survival value**: with 2 allies alive, the per-step shaping bonus is `2*0.02 - 0.003 = +0.037`. Over 200 steps this is 7.4 — comparable to the original damage reward, so the agent still has incentive to fight.

## Metric naming

| Key | When v2 disabled | When v2 enabled |
|-----|-----------------|-----------------|
| `log/step_kill_bonus` | `kill_delta × kill_reward_bonus` | **0.0** |
| `log/step_step_penalty` | `step_penalty` | **0.0** |
| `log/step_v2_win_bonus` | 0.0 | `win_bonus` on terminal win |
| `log/step_v2_loss_penalty` | 0.0 | `loss_penalty` on terminal loss |
| `log/step_v2_enemy_kill_bonus` | 0.0 | `kill_delta × enemy_kill_bonus` |
| `log/step_v2_ally_death_penalty` | 0.0 | `ally_deaths × ally_death_penalty` |
| `log/step_v2_ally_survival_bonus` | 0.0 | `n_alive × ally_survival_bonus` |
| `log/step_v2_step_penalty` | 0.0 | `step_penalty` |
| `log/original_env_reward` | = `reward` | raw SMAClite reward (preserved) |
| `log/shaped_reward` | = `reward` | = `obs["reward"]` |
| `log/reward_shaping_bonus` | 0.0 | `shaped - original` |
| `log/reward_shaping_enabled` | 0.0 | 1.0 |
| `log/allies_alive` | count | count |
| `log/enemies_alive` | count | count |

## Manual commands

```cmd
cd C:\Users\gsimru\Documents\smac-dreamer
conda activate smaclite-env
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
```

### Smoke test (run first)

```cmd
python scripts\smoke_test_reward_shaping.py --scenario 2s_vs_1sc --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml
```

### Stage 1 — 100k steps

```cmd
python scripts\train_dreamer_smaclite_phase3.py --configs smaclite_phase3d_2s_vs_1sc_v2 size1m --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_100k --run.steps 100000
```

Evaluate (original reward mode — primary signal):

```cmd
python scripts\evaluate_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_100k --episodes 50 --seed 42 --reward_mode original --output results\eval_phase3d_2s_vs_1sc_v2_100k_original.json --jsonl_output results\eval_phase3d_2s_vs_1sc_v2_100k_original.jsonl
```

Evaluate (shaped reward mode — diagnostic):

```cmd
python scripts\evaluate_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_100k --episodes 50 --seed 42 --reward_mode shaped --output results\eval_phase3d_2s_vs_1sc_v2_100k_shaped.json --jsonl_output results\eval_phase3d_2s_vs_1sc_v2_100k_shaped.jsonl
```

Check training log for required metric keys:

```cmd
python -c "import json,pathlib,math; p=pathlib.Path(r'logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_100k\metrics.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; bad=[]; [bad.append((k,v)) for r in rows for k,v in r.items() if isinstance(v,float) and (math.isnan(v) or math.isinf(v))]; keys=set().union(*[r.keys() for r in rows]); required=['epstats/log/original_env_reward/avg','epstats/log/shaped_reward/avg','epstats/log/reward_shaping_bonus/avg','epstats/log/episode_original_env_return/avg','epstats/log/episode_shaped_return/avg','epstats/log/episode_reward_shaping_bonus/avg','epstats/log/reward_shaping_enabled/avg','epstats/log/masking_failure_rate/avg']; [print(k,'FOUND' if k in keys else 'MISSING') for k in required]; print('nan_inf_count:',len(bad)); print('bad:',bad[:5])"
```

### Stage 2 — 300k (only if 100k is sane)

```cmd
python scripts\train_dreamer_smaclite_phase3.py --configs smaclite_phase3d_2s_vs_1sc_v2 size1m --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_300k --run.steps 300000
```

```cmd
python scripts\evaluate_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_300k --episodes 50 --seed 42 --reward_mode original --output results\eval_phase3d_2s_vs_1sc_v2_300k_original.json --jsonl_output results\eval_phase3d_2s_vs_1sc_v2_300k_original.jsonl
```

### Stage 3 — 1M (only if 300k is sane)

```cmd
python scripts\train_dreamer_smaclite_phase3.py --configs smaclite_phase3d_2s_vs_1sc_v2 size1m --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_1m --run.steps 1000000
```

```cmd
python scripts\evaluate_phase3.py --manifest configs\maps\phase3d_overfit_2s_vs_1sc_manifest.yaml --logdir logs\smaclite_phase3d\overfit_2s_vs_1sc_v2_1m --episodes 50 --seed 42 --reward_mode original --output results\eval_phase3d_2s_vs_1sc_v2_1m_original.json --jsonl_output results\eval_phase3d_2s_vs_1sc_v2_1m_original.jsonl
```

## Baseline to beat

| Metric | Original 1M run |
|--------|----------------|
| `win_rate` | 0.0 |
| `mean_episode_reward` (original) | 8.696 |
| `mean_episode_length` | 32.0 |
| `masking_failure_rate` | 0.0 |

**`masking_failure_rate` must remain 0.0** throughout. Any regression is a blocker.

## Anti-stalling interpretation guide

Success is judged by `original_env_reward` and `win_rate`, not `shaped_reward`.

| What you see | Interpretation | Action |
|---|---|---|
| `mean_episode_length` > 32 AND `original_env_reward` ≥ 8.7 | Surviving longer while maintaining damage — good | Continue |
| `mean_episode_length` > 32 BUT `original_env_reward` < 8.0 | Stalling for survival bonus without fighting | Reduce `ally_survival_bonus` (e.g. 0.01) |
| `mean_episode_length` ≈ 32 AND `win_rate` > 0 | Finding fast wins — excellent | Continue |
| `win_rate` still 0.0 after 300k, episode_length increasing | Learning, needs more steps | Run 1M |
| `masking_failure_rate` > 0 | Regression in action masking | Stop, investigate |

## Acceptance criteria

**Implementation:**
1. `reward_shaping_config=None` → behaviour identical to pre-v2 pipeline
2. `reward_shaping.enabled: false` → same as above
3. `reward_shaping.enabled: true` → agent trains on shaped reward; original in `log/original_env_reward`
4. Old `log/step_kill_bonus` / `log/step_step_penalty` kept; == 0.0 when v2 active
5. New `log/step_v2_*` keys == 0.0 when v2 disabled
6. `--reward_mode` changes only output aggregation; policy input unchanged
7. Smoke tests pass all four cases with no assertion failures
8. No NaN/Inf in training log; `masking_failure_rate` == 0.0

**Experiment targets:**
- 100k: `mean_episode_length > 32`, no reward collapse, no NaN
- 300k: `original_env_reward ≥ 8.7` OR `win_rate > 0`
- 1M: `win_rate > 0` on `2s_vs_1sc` (main goal)
