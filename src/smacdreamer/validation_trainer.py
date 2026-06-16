"""ValidationTrainer (P0.4): replaces the old worker-based periodic evaluator.

On each validation tick it runs an explicit ``map × fixed_seed`` validation pass over the
held-out VALIDATION maps (ORIGINAL reward), logs per-map + macro/micro metrics, and saves
``best_val_macro_winrate.pt`` selected by MACRO validation win rate, tie-broken by MACRO
original return — never by shaped return.

NOTE: imports ``OnlineTrainer`` from R2-Dreamer, so it requires ``external/r2dreamer`` on
``sys.path`` (the training script sets this up before importing this module). The pure
selection rule lives in ``smacdreamer.evaluation.is_validation_improvement`` so it can be
unit-tested without R2-Dreamer.
"""

import math
import pathlib

import torch

from trainer import OnlineTrainer  # requires external/r2dreamer on sys.path

from smacdreamer.evaluation import evaluate_heldout, is_validation_improvement

# Non-None sentinel: the base training loop only calls self.eval() when eval_envs is not None
# (and eval_episode_num > 0). We pass this up and override eval(); eval_envs is never used.
_VALIDATION_SENTINEL = object()


class ValidationTrainer(OnlineTrainer):
    """OnlineTrainer whose periodic eval is an explicit map×seed validation + best-ckpt save."""

    def __init__(self, config, replay_buffer, logger, logdir, train_envs, *,
                 validation_entries, pad_dims, seeds, device, gamma, max_episode_steps, obs_mode):
        super().__init__(config, replay_buffer, logger, logdir, train_envs, _VALIDATION_SENTINEL)
        self._val_entries = list(validation_entries)
        self._val_pad = pad_dims
        self._val_seeds = [int(s) for s in seeds]
        self._val_device = str(device)
        self._val_gamma = float(gamma)
        self._val_max_steps = int(max_episode_steps)
        self._val_obs_mode = str(obs_mode)
        self._logdir = pathlib.Path(logdir)
        self._best_macro_wr = -1.0
        self._best_macro_ret = -math.inf

    def eval(self, agent, train_step):
        # Lazy import to avoid a circular import at module load (factory imports this module).
        from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env

        agent.eval()
        report = evaluate_heldout(
            agent, self._val_entries, self._val_pad,
            seeds=self._val_seeds, device=self._val_device, gamma=self._val_gamma,
            max_episode_steps=self._val_max_steps, obs_mode=self._val_obs_mode,
            env_factory=make_smaclite_multimap_env, progress=False,
        )
        macro, micro = report["macro"], report["micro"]
        for k in ("win_rate", "original_return", "length", "timeout_rate",
                  "final_ally_ehp_frac", "final_enemy_ehp_frac"):
            self.logger.scalar(f"val/macro_{k}", float(macro[k]))
            self.logger.scalar(f"val/micro_{k}", float(micro[k]))
        self.logger.scalar("val/n_maps", float(report["n_maps"]))
        self.logger.scalar("val/n_episodes", float(report["n_episodes_total"]))

        wr, ret = float(macro["win_rate"]), float(macro["original_return"])
        if is_validation_improvement(wr, ret, self._best_macro_wr, self._best_macro_ret):
            self._best_macro_wr, self._best_macro_ret = wr, ret
            torch.save(
                {"agent_state_dict": agent.state_dict(),
                 "val_macro_win_rate": wr, "val_macro_original_return": ret,
                 "step": int(train_step), "obs_mode": self._val_obs_mode},
                self._logdir / "best_val_macro_winrate.pt",
            )
            print(f"  [val step {train_step}] NEW BEST macro win_rate={wr:.3f} "
                  f"(orig_return={ret:.3f}) -> best_val_macro_winrate.pt")
        else:
            print(f"  [val step {train_step}] macro win_rate={wr:.3f} "
                  f"orig_return={ret:.3f} (best {self._best_macro_wr:.3f})")
        self.logger.write(train_step)
        agent.train()


__all__ = ["ValidationTrainer"]
