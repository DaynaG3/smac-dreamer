"""
DreamerV3 training launcher for SMAClite Phase 2 (same-shape multi-map).

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\train_dreamer_smaclite_phase2.py --configs debug smaclite_phase2 --logdir logs\\smaclite_phase2\\debug --run.steps 500
    python scripts\\train_dreamer_smaclite_phase2.py --configs debug smaclite_phase2 --logdir logs\\smaclite_phase2\\debug_5k --run.steps 5000
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
from smacdreamer.envs.map_sampler import MapSampler

# --- Windows path-separator fix for elements.LocalPath ---
# See train_dreamer_smaclite_phase1.py for explanation.
import glob as _globlib
def _glob_normalized(self, pattern):
    for p in _globlib.glob(f'{str(self)}/{pattern}', recursive=True):
        yield type(self)(p.replace('\\', '/'))
elements.path.LocalPath.glob = _glob_normalized

_DREAMER_CONFIGS = ROOT / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml"
_SMAC_CONFIGS_P1 = ROOT / "configs" / "smaclite_phase1.yaml"
_SMAC_CONFIGS_P2 = ROOT / "configs" / "smaclite_phase2.yaml"


def load_configs() -> dict:
    dreamer_text = _DREAMER_CONFIGS.read_text(encoding="utf-8")
    configs = yaml.YAML(typ="safe").load(dreamer_text)

    # Seed env.smaclite into defaults so named-block updates can override it.
    configs["defaults"].setdefault("env", {})["smaclite"] = {
        "scenario": "2s3z",
        "max_episode_steps": 200,
        "seed": 0,
        "map_manifest": "",
        "map_mode": "round_robin",
        "map_seed": 42,
    }

    for cfg_path in [_SMAC_CONFIGS_P1, _SMAC_CONFIGS_P2]:
        text = cfg_path.read_text(encoding="utf-8")
        configs.update(yaml.YAML(typ="safe").load(text))
    return configs


def make_env(config, index: int):
    smaclite_cfg = config.env.get("smaclite", {})
    scenario = smaclite_cfg.get("scenario", "2s3z")
    max_ep = smaclite_cfg.get("max_episode_steps", 200)
    seed = smaclite_cfg.get("seed", 0)
    map_manifest = smaclite_cfg.get("map_manifest", None)
    map_mode = smaclite_cfg.get("map_mode", "round_robin")
    map_seed = smaclite_cfg.get("map_seed", 42)

    map_sampler = None
    if map_manifest:
        manifest_path = str(ROOT / map_manifest)
        map_sampler = MapSampler.from_manifest(manifest_path, mode=map_mode, seed=map_seed + index)

    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_ep,
        seed=seed + index,
        map_sampler=map_sampler,
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
            f"Script '{config.script}' not supported. Use 'train' or 'train_eval'."
        )


if __name__ == "__main__":
    main()
