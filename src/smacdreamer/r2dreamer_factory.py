"""Factory that creates R2-Dreamer-compatible parallel env pools for SMAClite.

Returns (train_envs, eval_envs, obs_space, act_space) — the same 4-tuple that
R2-Dreamer's train.py expects from make_envs() — so the rest of the training
pipeline is unchanged.

Does NOT modify any file inside external/r2dreamer.
"""

import pathlib
import sys


def _ensure_paths():
    """Add r2dreamer and smaclite to sys.path in the calling process.

    Called before any r2dreamer import so that r2dreamer's own unqualified
    imports (tools, rssm, networks, envs.parallel, …) resolve correctly.
    Also called inside worker lambdas so spawned subprocesses get the paths too.
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    for sub in ("external/r2dreamer", "external/smaclite"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def make_smaclite_env(scenario, max_episode_steps=200, seed=0, worker_idx=0):
    """Construct a single R2-Dreamer-compatible SMAClite env instance.

    Called inside ParallelEnv worker processes via a cloudpickle-serialised
    lambda; _ensure_paths() sets up sys.path before any import.
    """
    _ensure_paths()
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.r2dreamer_adapter import SMACliteR2DreamerAdapter

    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        seed=seed + worker_idx,
    )
    return SMACliteR2DreamerAdapter(env)


def make_smaclite_envs(
    scenario,
    env_num,
    eval_episode_num,
    device,
    max_episode_steps=200,
    seed=0,
):
    """Create train and (optionally) eval ParallelEnv pools.

    Parameters
    ----------
    scenario        : SMAClite scenario ID, e.g. "2s3z"
    env_num         : number of parallel training environments
    eval_episode_num: number of parallel evaluation environments (0 = disabled)
    device          : torch device string, e.g. "cpu" or "cuda:0"
    max_episode_steps: per-episode step limit passed to SMACliteDreamerEnv
    seed            : base RNG seed; each worker gets seed + worker_idx

    Returns
    -------
    (train_envs, eval_envs, obs_space, act_space)
        train_envs : ParallelEnv
        eval_envs  : ParallelEnv | None
        obs_space  : gymnasium.spaces.Dict
        act_space  : gymnasium.spaces.Box  (multi_discrete=True)
    """
    _ensure_paths()
    from envs.parallel import ParallelEnv

    def constructor(idx):
        # Returns a zero-argument callable for ParallelEnv; captured vars are
        # cloudpickle-serialised so the worker subprocess can reconstruct the env.
        return lambda: make_smaclite_env(scenario, max_episode_steps, seed, idx)

    train_envs = ParallelEnv(constructor, env_num, device)
    eval_envs = (
        ParallelEnv(constructor, eval_episode_num, device)
        if eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space
