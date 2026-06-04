"""
Custom DreamerV3 training launcher for SMAClite Phase 1.

Bypasses DreamerV3's suite registry; injects SMACliteDreamerEnv directly.
No modifications to external/dreamerv3 are required.

Usage (PowerShell):
    $env:PYTHONPATH = "$PWD\src;$PWD\external\dreamerv3;$PWD\external\smaclite"
    python scripts\train_dreamer_smaclite_phase1.py --configs debug --logdir logs\p1\debug

To also load the phase1 config block (recommended):
    python scripts\train_dreamer_smaclite_phase1.py --configs smaclite_phase1 debug --logdir logs\p1\debug --run.steps 500
"""

import os
import sys
import pathlib
from functools import partial as bind

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ruamel.yaml as yaml
import numpy as np
import elements
import embodied
import portal
import dreamerv3

from dreamerv3.main import make_replay, make_stream, make_logger, wrap_env
from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv

# --- Windows path-separator fix for elements.LocalPath ---
# elements.Path.name splits only on '/'. On Windows, glob.glob returns
# backslash-separated paths regardless of the input pattern, so
# LocalPath.name returns 'ckpt\timestamp' instead of 'timestamp'.
# This breaks Checkpoint._cleanup(): the 'latest' file is not filtered
# out, the checkpoint folder sorts before 'latest' (digits < letters),
# and _cleanup deletes the checkpoint folder itself.
# Fix: normalize all glob output to forward slashes before wrapping.
import glob as _globlib
def _glob_normalized(self, pattern):
    for p in _globlib.glob(f'{str(self)}/{pattern}', recursive=True):
        yield type(self)(p.replace('\\', '/'))
elements.path.LocalPath.glob = _glob_normalized

_DREAMER_CONFIGS = ROOT / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml"
_SMAC_CONFIGS = ROOT / "configs" / "smaclite_phase1.yaml"


def load_configs() -> dict:
    """Merge DreamerV3's configs.yaml with our smaclite_phase1.yaml.

    elements.Config.update() only permits updating keys that already exist in
    the base (defaults) config.  Because DreamerV3's defaults have no
    env.smaclite section, we inject it into the defaults dict before building
    the Config so that subsequent named-block updates can override it cleanly.
    """
    dreamer_text = _DREAMER_CONFIGS.read_text(encoding="utf-8")
    configs = yaml.YAML(typ="safe").load(dreamer_text)

    # Seed env.smaclite into defaults so named-block updates can override it.
    configs["defaults"].setdefault("env", {})["smaclite"] = {
        "scenario": "2s3z",
        "max_episode_steps": 200,
        "seed": 0,
    }

    configs["defaults"]["wandb"] = {
        "project": "smac-dreamer",
        "entity": "",
        "group": "",
        "tags": [],
        "notes": "",
        "mode": "online",
    }

    smac_text = _SMAC_CONFIGS.read_text(encoding="utf-8")
    smac_configs = yaml.YAML(typ="safe").load(smac_text)
    configs.update(smac_configs)
    return configs


def _init_wandb(config):
    if 'wandb' not in config.logger.outputs:
        return
    import wandb
    wb = config.get('wandb', {})
    parts = str(config.logdir).replace('\\', '/').split('/')
    run_name = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    wandb.init(
        project=wb.get('project', 'smac-dreamer') or 'smac-dreamer',
        entity=wb.get('entity') or None,
        group=wb.get('group') or None,
        name=run_name,
        tags=list(wb.get('tags', [])),
        notes=wb.get('notes') or None,
        mode=wb.get('mode', 'online'),
        config=dict(config.flat),
        resume='allow',
    )


def make_env(config, index: int):
    smaclite_cfg = config.env.get("smaclite", {})
    scenario = smaclite_cfg.get("scenario", "2s3z")
    max_ep = smaclite_cfg.get("max_episode_steps", 200)
    seed = smaclite_cfg.get("seed", 0)
    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_ep,
        seed=seed + index,
    )
    return wrap_env(env, config)


def make_agent(config, env_factory):
    from smacdreamer.agent import SMACliteAgent

    env = env_factory(0)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith("log/")}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()

    cpdir = elements.Path(config.logdir)
    cpdir = cpdir.parent if config.replicas > 1 else cpdir
    return SMACliteAgent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=str(cpdir),
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )


def main(argv=None):
    configs = load_configs()

    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
    config = config.update(
        logdir=os.path.abspath(
            config.logdir.format(timestamp=elements.timestamp())
        ).replace('\\', '/')
    )

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    print("Script:", config.script)
    logdir.mkdir()
    config.save(logdir / "config.yaml")
    _init_wandb(config)

    def init():
        elements.timer.global_timer.enabled = config.logger.timer

    portal.setup(
        errfile=config.errfile and logdir / "error",
        clientkw=dict(logging_color="cyan"),
        serverkw=dict(logging_color="cyan"),
        initfns=[init],
        ipv6=config.ipv6,
    )

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    env_factory = bind(make_env, config)

    if config.script == "train":
        embodied.run.train(
            bind(make_agent, config, env_factory),
            bind(make_replay, config, "replay"),
            env_factory,
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )
    elif config.script == "train_eval":
        embodied.run.train_eval(
            bind(make_agent, config, env_factory),
            bind(make_replay, config, "replay"),
            bind(make_replay, config, "eval_replay", "eval"),
            env_factory,
            env_factory,
            bind(make_stream, config),
            bind(make_logger, config),
            args,
        )
    else:
        raise NotImplementedError(
            f"Script '{config.script}' not supported by this launcher. "
            "Use 'train' or 'train_eval'."
        )


if __name__ == "__main__":
    main()
