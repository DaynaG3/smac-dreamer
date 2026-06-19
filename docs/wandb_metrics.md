# W&B Metrics Reference — R2-Dreamer × SMAClite

This document catalogues every metric logged to Weights & Biases by the R2-Dreamer
training pipeline on the SMAClite simulator, what it measures, and why it matters for
diagnosing and steering training.

## How logging works

All metrics flow through [`WandbLogger`](../src/smacdreamer/wandb_logger.py), a subclass of
R2-Dreamer's `tools.Logger`. Code calls `logger.scalar(name, value)` to buffer a scalar and
`logger.write(step)` to flush the buffer. `WandbLogger.write()` snapshots the buffered scalars
and forwards them to `wandb.log()`, while the parent still writes TensorBoard / console /
`metrics.jsonl`.

- **X-axis** — Every chart uses `global_step` as its x-axis (declared via
  `wandb.define_metric("*", step_metric="global_step")`). `global_step = local_env_step + step_offset`,
  so a 2M→4M continuation run plots on a continuous absolute axis.
- **`log_`-prefixed env keys** — The SMAClite adapter emits diagnostic values into the
  observation/info dict with a `log_` prefix. The model encoder deliberately excludes any
  `log_` key (so they never become model inputs), and the trainer aggregates them per
  episode. See [`smaclite_dreamer_env.py`](../src/smacdreamer/envs/smaclite_dreamer_env.py)
  and [`r2dreamer_adapter.py`](../src/smacdreamer/envs/r2dreamer_adapter.py).
- **Two training entry points** — [`train_r2dreamer_smaclite_debug.py`](../scripts/train_r2dreamer_smaclite_debug.py)
  uses the base `OnlineTrainer` (evaluation disabled). [`train_r2dreamer_smaclite_multimap.py`](../scripts/train_r2dreamer_smaclite_multimap.py)
  uses [`ValidationTrainer`](../src/smacdreamer/validation_trainer.py), which overrides eval to
  emit the `val/*` family below.

---

## 1. `episode/*` — Per-completed-episode training metrics

Logged in [`trainer.py`](../external/r2dreamer/trainer.py) each time a training episode finishes,
from the final transition's `log_*` info (mapped via `EPISODE_REWARD_LOG_MAP`). These are the
primary "is the agent actually getting better at the game" signals during training rollouts.

| Metric | Source | Significance |
|---|---|---|
| `episode/score` | `returns[i]` | Total reward the agent collected in the finished episode, using the **training reward** (shaped if shaping is on). Headline learning-progress signal, but reflects the shaped objective, not raw game outcome. |
| `episode/length` | `lengths[i]` | Episode length in steps. Rising length early can mean the agent is surviving longer; near the cap it can mean stalling/timeouts. |
| `episode/reward_original_return` | `log_episode_original_env_return` | Episode return under the **unshaped** SMAClite reward. The honest game-outcome signal — compare against `episode/score` to see how much reward shaping is adding. |
| `episode/reward_shaped_return` | `log_episode_shaped_return` | Episode return under the shaped reward actually optimized. Equals `episode/score` when shaping is active. |
| `episode/reward_shaping_total` | `log_episode_reward_shaping_bonus` | Total shaping bonus added over the episode (`shaped − original`). Tracks how much the shaping term is contributing; large drift here is a flag for objective distortion. |
| `episode/reward_ally_ehp_dense` | `log_reward_term_ally_ehp_dense_ep_sum` | Episode sum of the dense ally effective-HP shaping term. Shows whether the "keep allies alive / healthy" incentive is firing. |
| `episode/reward_win_ehp_quality` | `log_reward_term_win_ehp_quality_ep_sum` | Episode sum of the win-quality bonus tied to remaining ally EHP on a win. Rewards winning *decisively* (units healthy), not just barely. |
| `episode/reward_win_alive_quality` | `log_reward_term_win_alive_quality_ep_sum` | Episode sum of the win-quality bonus tied to number of allies still alive on a win. Encourages winning with fewer losses. |
| `episode/reward_timeout` | `log_reward_term_timeout_ep_sum` | Episode sum of the timeout penalty term. Non-zero indicates episodes ending by truncation rather than a decisive result. |
| `episode/final_ally_ehp_frac` | `log_final_ally_ehp_frac` | Fraction of allied effective HP (health+shield) remaining at episode end. Combat-survival quality; near 0 means the team was wiped. |
| `episode/final_ally_alive_frac` | `log_final_ally_alive_frac` | Fraction of allied units still alive at episode end. Coarser survival measure than EHP. |
| `episode/final_enemy_ehp_frac` | `log_final_enemy_ehp_frac` | Fraction of enemy effective HP remaining at episode end. Near 0 means the agent destroyed the enemy team (progress toward a win). |

---

## 2. `train/*` — World-model and actor-critic optimization metrics

Produced by `agent.update()` in [`dreamer.py`](../external/r2dreamer/dreamer.py), then prefixed
with `train/` and logged every `update_log_every` steps in `trainer.py`. These are the internal
health metrics of DreamerV3 itself — they tell you whether the world model and policy are
learning stably, independent of game score.

### 2a. Losses — `train/loss/*`

Each entry of the model's `losses` dict, logged as `train/loss/<name>`.

| Metric | Significance |
|---|---|
| `train/loss/dyn` | **Dynamics KL** — pulls the prior (imagined next-state) toward the posterior. Core world-model consistency loss; instability here means the latent dynamics are not learning. |
| `train/loss/rep` | **Representation KL** — pulls the posterior toward the prior (regularizes the encoder). Balanced against `dyn` via KL free-bits. |
| `train/loss/rew` | **Reward-prediction loss** — negative log-prob of observed reward under the reward head. High/rising = the model can't predict reward, which corrupts imagined returns. |
| `train/loss/con` | **Continuation (done) loss** — negative log-prob of the episode-continue flag. Needed for correct discounting of imagined rollouts at episode boundaries. |
| `train/loss/<decoder key>` | **Reconstruction losses** (when `rep_loss == "dreamer"`) — one per decoded observation key. Measures how well the latent reconstructs observations. |
| `train/loss/barlow` | **Barlow-Twins representation loss** (when `rep_loss == "r2dreamer"`) — invariance + redundancy-reduction term; the R2-Dreamer self-supervised objective. |
| `train/loss/infonce` | **InfoNCE contrastive loss** (when `rep_loss == "infonce"`). |
| `train/loss/swav`, `train/loss/temp`, `train/loss/norm` | **DreamerPro prototype losses** (when `rep_loss == "dreamerpro"`). |
| `train/loss/avail` | **Available-action head loss** (action masking on) — predicts each unit's valid-action mask so imagination can respect it. |
| `train/loss/alive` | **Alive/active head loss** (action masking on) — predicts which unit slots are real/alive (vs padded/dead). |
| `train/loss/policy` | **Actor loss** — the policy-gradient objective in imagination. Zeroed during actor warm-up (see `train/actor_updates_enabled`). |
| `train/loss/value` | **Critic loss** — value-function regression toward imagined λ-returns. |
| `train/loss/repval` | **Replay-value loss** — critic regression on real replayed states (anchors the critic to real data, not only imagination). |
| `train/opt/loss` | **Total weighted loss** actually backpropagated (sum of the above × loss scales). The single number to watch for divergence. |

### 2b. Optimizer health — `train/opt/*`

| Metric | Significance |
|---|---|
| `train/opt/grad_norm` | Global gradient L2 norm before clipping. Spikes precede/accompany instability; flatlining near 0 means learning has stalled. |
| `train/opt/grad_rms` | RMS of gradients. Scale-robust companion to `grad_norm`. |
| `train/opt/lr` | Current learning rate from the scheduler. Confirms warm-up/decay is behaving. |
| `train/opt/grad_scale` | AMP gradient-scaler scale factor. Repeated halving signals fp16/bf16 overflow (NaNs/Infs). |
| `train/opt/param_rms` | RMS of parameter magnitudes (when grad logging on). Drift indicates weight blow-up/collapse. |
| `train/opt/update_rms` | RMS of per-step parameter updates (when grad logging on). Effective step size. |
| `train/opt/updates` | Cumulative count of gradient updates applied. Tracks train-ratio / update throughput vs env steps. |

### 2c. Imagination, value, and policy diagnostics

| Metric | Significance |
|---|---|
| `train/dyn_entropy` | Entropy of the prior (imagined) latent distribution. Collapse → the world model became overconfident/degenerate. |
| `train/rep_entropy` | Entropy of the posterior latent distribution. Healthy spread of encoded states. |
| `train/ret` | Mean normalized imagined return — the target the actor optimizes after return normalization. |
| `train/ret_005`, `train/ret_095` | The 5th/95th-percentile EMAs used by the return normalizer (DreamerV3 percentile scaling). Their gap is the normalization range. |
| `train/adv` / `train/adv_std` | Mean / std of the advantage estimate driving the policy gradient. |
| `train/con` | Mean predicted continuation probability over imagined states. |
| `train/rew` | Mean predicted reward over imagined rollouts. |
| `train/val` | Mean critic value over imagined states. |
| `train/tar` | Mean λ-return target (the value-regression target). |
| `train/slowval` | Mean value from the slow/target critic (EMA copy) used to stabilize the target. |
| `train/weight` | Mean discount weight applied across the imagination horizon (down-weights post-termination steps). |
| `train/action_entropy` | Mean policy entropy in imagination. Drops as the policy commits; collapsing too fast = premature exploitation. |
| `train/action_mean/std/min/max` | Distribution stats of imagined actions (`tensorstats`). Sanity-checks the action distribution. |
| `train/ret_replay_*`, `train/value_replay_*`, `train/slow_value_replay_*` | Mean/std/min/max of returns and (slow) values on **real replayed** states. Compare against the imagined counterparts to detect imagination–reality mismatch. |

### 2d. Action-masking diagnostics (only when `action_masking: true`)

These quantify how well the learned avail/alive heads reproduce the true SMAClite action mask —
critical because the agent acts in imagination through the *predicted* mask.

| Metric | Significance |
|---|---|
| `train/mask_precision` | Precision of the predicted available-action mask vs ground truth. Low precision = the model thinks invalid actions are valid. |
| `train/mask_recall` | Recall of the predicted mask. Low recall = the model hides genuinely valid actions, shrinking the action space. |
| `train/mask_fpr` | False-positive rate of the predicted mask (invalid actions marked valid). |
| `train/imag_pre_mask_invalid_mass` | Probability mass the imagined policy places on invalid actions **before** masking. |
| `train/imag_pre_mask_invalid_sample_rate` (= `train/imag_invalid_rate`) | Rate at which a greedy imagined action would be invalid before masking. Back-compat alias kept. |
| `train/imag_post_mask_invalid_sample_rate` | Invalid-action rate **after** applying the predicted mask — residual leakage the mask failed to catch. |
| `train/imag_empty_mask_rate` | Rate at which the predicted mask leaves a unit with no valid action (degenerate mask). |
| `train/real_pre_mask_invalid_mass` | Same pre-mask invalid mass but measured on real replay states (not imagination). |
| `train/real_post_mask_invalid_sample_rate` | Post-mask invalid-action rate on real states. |
| `train/maskh0_posterior_precision/recall/fpr` | Mask quality at the **posterior** (horizon 0). Should stay high; the cleanest mask regime. |
| `train/maskh{1..}_*` | Mask quality at successive open-loop imagination horizons. Degradation with horizon isolates world-model drift from head accuracy. |

### 2e. Continuation / warm-up bookkeeping

| Metric | Significance |
|---|---|
| `train/actor_updates_enabled` | 1.0 once the actor warm-up threshold is passed, else 0.0. Confirms when the policy started updating (world model recalibrates first on a transfer/resume). |
| `train/local_step` | The trainer's local env step (resets to 0 on a fresh run). |
| `train/global_step` | `local_step + step_offset` — the absolute step shared as the chart x-axis. |

---

## 3. `val/*` — Held-out validation metrics (multimap only)

Emitted by [`ValidationTrainer.eval()`](../src/smacdreamer/validation_trainer.py) every
`validation.every` steps, by running every held-out validation map under every fixed seed with
the **original (unshaped) reward** — see [`evaluation.py`](../src/smacdreamer/evaluation.py).
These are the generalization metrics and the basis for checkpoint selection.

Aggregates are reported two ways:
- **macro** — each *map* contributes one sample (mean of per-map means). **Headline** /
  selection metric; robust to maps with many vs few episodes.
- **micro** — each *episode* contributes one sample (pooled). Reflects raw per-episode average.

| Metric (both `val/macro_*` and `val/micro_*`) | Significance |
|---|---|
| `..._win_rate` | Fraction of validation episodes won. **`val/macro_win_rate` is the primary checkpoint-selection metric** (best model saved as `best_val_macro_winrate.pt`); tie-broken by macro original return. |
| `..._original_return` | Mean **unshaped** SMAClite return on held-out maps. The honest generalization-return signal; never used as the shaped objective. |
| `..._length` | Mean validation episode length. |
| `..._timeout_rate` | Fraction of validation episodes ending by truncation (no decisive result) — high values flag stalling policies. |
| `..._final_ally_ehp_frac` | Mean remaining allied effective-HP fraction at episode end (combat efficiency). |
| `..._final_enemy_ehp_frac` | Mean remaining enemy effective-HP fraction (lower = more enemy destroyed). |

Win-quality (computed over winning episodes only; `0.0` sentinel when a map/pool has no wins):

| Metric | Significance |
|---|---|
| `val/macro_win_final_ally_ehp_frac`, `val/micro_win_final_ally_ehp_frac` | Average remaining ally EHP **on wins** — measures how *cleanly* the agent wins, not just whether it wins. |
| `val/macro_win_alive_fraction`, `val/micro_win_alive_fraction` | Average fraction of allies still alive **on wins**. |

Counts and validation-pass telemetry:

| Metric | Significance |
|---|---|
| `val/n_maps` | Number of held-out maps evaluated this pass. |
| `val/n_episodes` | Total validation episodes this pass (maps × seeds). |
| `val/rss_before_gb`, `val/rss_after_gb`, `val/rss_delta_gb` | Main-process resident memory before/after the validation pass and the delta. Catches memory leaks from the per-map isolated eval subprocesses. |

---

## 4. `system/*`, `replay/*`, `env/*` — Telemetry

Collected by [`system_metrics.py`](../src/smacdreamer/system_metrics.py) every
`system_log_every` steps (and at each validation tick). Telemetry failures are silently ignored,
so a metric may be absent on platforms where the source is unavailable (e.g. cgroup files on
non-Linux). These keep long cloud/GPU runs debuggable.

| Metric | Significance |
|---|---|
| `system/main_pid` | PID of the main training process. |
| `system/main_rss_available` | 1.0 if RSS could be read, else 0.0 (so a missing `main_rss_gb` is explained). |
| `system/main_rss_gb` | Main-process resident memory (GB). Primary leak/OOM watch on the trainer. |
| `system/worker_{slot}_rss_gb` | Resident memory of each env worker subprocess. |
| `system/worker_{slot}_pid` | PID of each env worker (for cross-referencing crashes/restarts). |
| `system/worker_{slot}_generation` | Worker generation counter — increments when a worker is restarted, exposing churn. |
| `system/worker_rss_gb` | Summed RSS across all env workers. Aggregate env-side memory footprint. |
| `system/cgroup_current_gb` | Current cgroup-v2 memory usage (container/pod accounting). |
| `system/cgroup_max_gb` | cgroup memory limit (0.0 when unlimited). Compare against `current` to predict OOM-kills. |
| `system/cgroup_events_*` | cgroup memory event counters (e.g. `oom`, `max`). Non-zero `oom` explains killed workers. |
| `system/cuda_allocated_gb` | CUDA memory actively allocated by tensors (GPU runs). |
| `system/cuda_reserved_gb` | CUDA memory reserved by the caching allocator. Gap vs allocated indicates fragmentation. |
| `replay/count` | Number of transitions/episodes currently in the replay buffer. Confirms the buffer is filling and (later) saturating at capacity. |
| `replay/storage_backend_code` | 1.0 = `memmap`, 0.0 = in-memory `tensor` replay backend. Records which storage path is in use. |
| `env/completed_episodes` | Total episodes completed across env workers. Cross-checks throughput against env steps. |
| `env/worker_restarts` | Cumulative env-worker restarts. Any growth signals env crashes worth investigating. |

---

## 5. `episode/eval_*` — Base-trainer evaluation (inherited; inactive in current scripts)

The base `OnlineTrainer.eval()` ([`trainer.py`](../external/r2dreamer/trainer.py)) logs
`episode/eval_score`, `episode/eval_length`, per-`log_*`-key aggregates `episode/eval_<key>`,
and (if `video_pred_log`) `eval_video` / `eval_open_loop` videos. **In this project these are
not produced**: the debug script disables eval (`eval_episode_num = 0`) and the multimap script
overrides `eval()` with the `val/*` family above. They are documented here only so the names are
not mistaken for missing data — held-out evaluation lives under `val/*`.

---

## Quick reference — what to watch for what

- **"Is the agent winning?"** → `val/macro_win_rate` (generalization), `episode/reward_original_return` (training rollouts).
- **"Is winning getting cleaner?"** → `val/macro_win_final_ally_ehp_frac`, `val/macro_win_alive_fraction`, `episode/final_ally_ehp_frac`.
- **"Is the world model healthy?"** → `train/loss/dyn`, `train/loss/rep`, `train/loss/rew`, `train/opt/loss`, `train/opt/grad_norm`, `train/opt/grad_scale`.
- **"Is the policy collapsing?"** → `train/action_entropy`, `train/adv_std`, `train/loss/policy`.
- **"Is action masking trustworthy?"** → `train/mask_precision/recall/fpr`, `train/imag_post_mask_invalid_sample_rate`, `train/maskh{h}_*`.
- **"Is shaping distorting the objective?"** → compare `episode/reward_shaped_return` vs `episode/reward_original_return`; watch `episode/reward_shaping_total`.
- **"Will the run OOM / is it leaking?"** → `system/main_rss_gb`, `system/cgroup_current_gb` vs `system/cgroup_max_gb`, `val/rss_delta_gb`, `env/worker_restarts`.
