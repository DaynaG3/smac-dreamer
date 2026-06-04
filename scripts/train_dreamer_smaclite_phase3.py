"""
DreamerV3 training launcher for SMAClite Phase 3 (padded multi-map).

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\train_dreamer_smaclite_phase3.py --configs debug smaclite_phase3 --logdir logs\\smaclite_phase3\\debug --run.steps 500
    python scripts\\train_dreamer_smaclite_phase3.py --configs debug smaclite_phase3 --logdir logs\\smaclite_phase3\\debug_5k --run.steps 5000
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
from smacdreamer.envs.map_sampler import MapSampler, MapEntry

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
_SMAC_CONFIGS_P3 = ROOT / "configs" / "smaclite_phase3.yaml"


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
        "use_padding": False,
        "kill_reward_bonus": 0.0,
        "step_penalty": 0.0,
        "reward_shaping": {
            "enabled": False,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "enemy_kill_bonus": 0.0,
            "ally_death_penalty": 0.0,
            "ally_survival_bonus": 0.0,
            "step_penalty": 0.0,
            "damage_delta_scale": 0.0,
        },
    }

    configs["defaults"]["wandb"] = {
        "project": "smac-dreamer",
        "entity": "",
        "group": "",
        "notes": "",
        "mode": "online",
    }

    for cfg_path in [_SMAC_CONFIGS_P1, _SMAC_CONFIGS_P2, _SMAC_CONFIGS_P3]:
        text = cfg_path.read_text(encoding="utf-8")
        configs.update(yaml.YAML(typ="safe").load(text))
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


def _load_pad_dims(map_manifest: str):
    """Parse the padding block from a manifest and return a PaddingDims, or None."""
    from smacdreamer.envs.padding import PaddingDims
    raw = yaml.YAML(typ='safe').load(
        pathlib.Path(ROOT / map_manifest).read_text(encoding='utf-8')
    )
    if 'padding' not in raw:
        return None
    p = raw['padding']
    return PaddingDims(
        max_agents=p['max_agents'],
        max_enemies=p['max_enemies'],
        max_actions=p['max_actions'],
        max_obs_size=p['max_obs_size'],
    )


def make_env(config, index: int):
    smaclite_cfg = config.env.get("smaclite", {})
    scenario     = smaclite_cfg.get("scenario", "2s3z")
    max_ep       = smaclite_cfg.get("max_episode_steps", 200)
    seed         = smaclite_cfg.get("seed", 0)
    map_manifest = smaclite_cfg.get("map_manifest", None)
    map_mode     = smaclite_cfg.get("map_mode", "round_robin")
    map_seed     = smaclite_cfg.get("map_seed", 42)
    use_padding  = bool(smaclite_cfg.get("use_padding", False))

    kill_reward_bonus = float(smaclite_cfg.get("kill_reward_bonus", 0.0))
    step_penalty      = float(smaclite_cfg.get("step_penalty",      0.0))

    from smacdreamer.envs.reward_shaping import from_dict as _rs_from_dict
    rs_raw = smaclite_cfg.get("reward_shaping", {})
    reward_shaping_config = _rs_from_dict(rs_raw)

    map_sampler = None
    pad_dims = None

    if map_manifest:
        manifest_path = str(ROOT / map_manifest)
        map_sampler = MapSampler.from_manifest(
            manifest_path, mode=map_mode, seed=map_seed + index)
        if use_padding:
            pad_dims = _load_pad_dims(map_manifest)

    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_ep,
        seed=seed + index,
        map_sampler=map_sampler,
        pad_dims=pad_dims,
        kill_reward_bonus=kill_reward_bonus,
        step_penalty=step_penalty,
        reward_shaping_config=reward_shaping_config,
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


def _validate_padding_at_startup(config):
    """Run validate_padding_dims once before training begins."""
    smaclite_cfg = config.env.get("smaclite", {})
    use_padding  = bool(smaclite_cfg.get("use_padding", False))
    map_manifest = smaclite_cfg.get("map_manifest", None)
    if not (use_padding and map_manifest):
        return

    from smacdreamer.envs.padding import PaddingDims, validate_padding_dims
    from smacdreamer.envs.map_sampler import validate_manifest, MapEntry

    manifest_path = str(ROOT / map_manifest)
    raw = validate_manifest(manifest_path)
    pad_dims = _load_pad_dims(map_manifest)
    if pad_dims is None:
        return

    entries = [
        MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
        for e in raw['maps']
    ]
    print("Phase 3: validating padding dims against all manifest maps...")
    validate_padding_dims(entries, pad_dims)
    print(f"  max_agents={pad_dims.max_agents}  max_enemies={pad_dims.max_enemies}"
          f"  max_actions={pad_dims.max_actions}  max_obs_size={pad_dims.max_obs_size}")
    print(f"  All {len(entries)} maps fit. Proceeding to training.\n")


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

    # Validate padding dims before allocating any JAX memory.
    _validate_padding_at_startup(config)

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
