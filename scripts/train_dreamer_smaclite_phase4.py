"""
DreamerV3 training launcher for SMAClite Phase 4 (folder-driven large-scale multi-map).

Loads a Phase 4 manifest (built by build_phase4_manifest.py), selects the train split,
computes padding dims, builds the map sampler, and starts DreamerV3 training.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    # 500-step W&B smoke run
    python scripts\\train_dreamer_smaclite_phase4.py ^
      --configs smaclite_phase4 debug ^
      --logdir logs\\smaclite_phase4\\wandb_smoke ^
      --run.steps 500 ^
      --wandb.project smac-dreamer ^
      --wandb.group phase4-folder-smoke

    # 5k integration run
    python scripts\\train_dreamer_smaclite_phase4.py ^
      --configs smaclite_phase4 debug ^
      --logdir logs\\smaclite_phase4\\debug_5k ^
      --run.steps 5000 ^
      --wandb.project smac-dreamer ^
      --wandb.group phase4-integration

    # Full 1M training
    python scripts\\train_dreamer_smaclite_phase4.py ^
      --configs smaclite_phase4 size1m ^
      --logdir logs\\smaclite_phase4\\size1m_1m ^
      --run.steps 1000000 ^
      --wandb.project smac-dreamer ^
      --wandb.group phase4-generalisation
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
from smacdreamer.envs.padding import PaddingDims, validate_padding_dims

# --- Windows path-separator fix for elements.LocalPath ---
import glob as _globlib
def _glob_normalized(self, pattern):
    for p in _globlib.glob(f'{str(self)}/{pattern}', recursive=True):
        yield type(self)(p.replace('\\', '/'))
elements.path.LocalPath.glob = _glob_normalized

_DREAMER_CONFIGS = ROOT / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml"
_SMAC_CONFIGS_P1 = ROOT / "configs" / "smaclite_phase1.yaml"
_SMAC_CONFIGS_P2 = ROOT / "configs" / "smaclite_phase2.yaml"
_SMAC_CONFIGS_P3 = ROOT / "configs" / "smaclite_phase3.yaml"
_SMAC_CONFIGS_P4 = ROOT / "configs" / "smaclite_phase4.yaml"


def load_configs() -> dict:
    dreamer_text = _DREAMER_CONFIGS.read_text(encoding="utf-8")
    configs = yaml.YAML(typ="safe").load(dreamer_text)

    configs["defaults"].setdefault("env", {})["smaclite"] = {
        "scenario": "2s3z",
        "max_episode_steps": 200,
        "seed": 0,
        "map_manifest": "",
        "dataset_manifest": "",
        "manifest_split": "train",
        "map_mode": "shuffled_round_robin",
        "map_seed": 42,
        "use_padding": True,
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

    for cfg_path in [_SMAC_CONFIGS_P1, _SMAC_CONFIGS_P2, _SMAC_CONFIGS_P3, _SMAC_CONFIGS_P4]:
        text = cfg_path.read_text(encoding="utf-8")
        configs.update(yaml.YAML(typ="safe").load(text))
    return configs


def _load_phase4_raw(manifest_path: str) -> dict:
    raw = yaml.YAML(typ='safe').load(
        pathlib.Path(manifest_path).read_text(encoding='utf-8')
    )
    if raw.get('version') != 1:
        raise ValueError(
            f"Expected a Phase 4 manifest (version: 1) at '{manifest_path}'. "
            f"Got version={raw.get('version')!r}. "
            "Build the manifest first with scripts/build_phase4_manifest.py."
        )
    return raw


def _load_pad_dims(raw: dict) -> PaddingDims:
    p = raw['padding']
    return PaddingDims(
        max_agents=p['max_agents'],
        max_enemies=p['max_enemies'],
        max_actions=p['max_actions'],
        max_obs_size=p['max_obs_size'],
    )


def _init_wandb(config, extra_config: dict = None):
    if 'wandb' not in config.logger.outputs:
        return
    import wandb
    wb = config.get('wandb', {})
    parts = str(config.logdir).replace('\\', '/').split('/')
    run_name = '/'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    run_config = dict(config.flat)
    if extra_config:
        run_config.update(extra_config)
    wandb.init(
        project=wb.get('project', 'smac-dreamer') or 'smac-dreamer',
        entity=wb.get('entity') or None,
        group=wb.get('group') or None,
        name=run_name,
        tags=list(wb.get('tags', [])),
        notes=wb.get('notes') or None,
        mode=wb.get('mode', 'online'),
        config=run_config,
        resume='allow',
    )


def _print_dataset_summary(raw: dict, split: str, pad_dims: PaddingDims,
                           map_mode: str, map_seed: int):
    print(f"\n{'='*60}")
    print("Phase 4 Dataset Summary")
    print(f"{'='*60}")
    print(f"  Dataset name  : {raw.get('dataset_name', 'unknown')}")
    print(f"  Dataset hash  : {raw.get('dataset_hash', 'N/A')[:24]}...")
    print(f"  Generated     : {raw.get('generated_at', 'N/A')}")
    split_counts = raw.get('split_counts', {})
    for s in ('train', 'validation', 'test'):
        n = split_counts.get(s, len(raw.get('splits', {}).get(s, [])))
        print(f"  {s:<12} : {n} maps")
    print(f"  Active split  : {split}")
    print(f"  Sampler mode  : {map_mode}")
    print(f"  Sampler seed  : {map_seed}")

    fam_counts = raw.get('family_split_counts', {})
    if fam_counts:
        print(f"  Families:")
        for fam, counts in sorted(fam_counts.items()):
            print(f"    {fam}: train={counts.get('train',0)} "
                  f"val={counts.get('validation',0)} test={counts.get('test',0)}")

    print(f"  Padding:")
    print(f"    max_agents   = {pad_dims.max_agents}")
    print(f"    max_enemies  = {pad_dims.max_enemies}")
    print(f"    max_actions  = {pad_dims.max_actions}")
    print(f"    max_obs_size = {pad_dims.max_obs_size}")
    print(f"{'='*60}\n")


def make_env(config, index: int):
    smaclite_cfg = config.env.get("smaclite", {})
    scenario        = smaclite_cfg.get("scenario", "2s3z")
    max_ep          = smaclite_cfg.get("max_episode_steps", 200)
    seed            = smaclite_cfg.get("seed", 0)
    dataset_manifest= smaclite_cfg.get("dataset_manifest", "")
    manifest_split  = smaclite_cfg.get("manifest_split", "train")
    map_mode        = smaclite_cfg.get("map_mode", "shuffled_round_robin")
    map_seed        = smaclite_cfg.get("map_seed", 42)
    use_padding     = bool(smaclite_cfg.get("use_padding", True))

    kill_reward_bonus = float(smaclite_cfg.get("kill_reward_bonus", 0.0))
    step_penalty      = float(smaclite_cfg.get("step_penalty", 0.0))

    from smacdreamer.envs.reward_shaping import from_dict as _rs_from_dict
    rs_raw = smaclite_cfg.get("reward_shaping", {})
    reward_shaping_config = _rs_from_dict(rs_raw)

    if not dataset_manifest:
        raise ValueError(
            "smaclite.dataset_manifest is not set. "
            "Build the manifest with scripts/build_phase4_manifest.py and set "
            "--env.smaclite.dataset_manifest or update smaclite_phase4.yaml."
        )

    manifest_path = str(ROOT / dataset_manifest)
    sampler = MapSampler.from_phase4_manifest(
        manifest_path, split=manifest_split, mode=map_mode, seed=map_seed + index)

    pad_dims = None
    if use_padding:
        raw = _load_phase4_raw(manifest_path)
        pad_dims = _load_pad_dims(raw)

    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_ep,
        seed=seed + index,
        map_sampler=sampler,
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


def _validate_padding_at_startup(config, raw: dict):
    """Validate that all train-split maps fit within the manifest's padding dims."""
    smaclite_cfg = config.env.get("smaclite", {})
    use_padding  = bool(smaclite_cfg.get("use_padding", True))
    if not use_padding:
        return

    pad_dims = _load_pad_dims(raw)
    split = smaclite_cfg.get("manifest_split", "train")
    entries = [
        MapEntry(name=e['name'], type=e.get('type', 'custom'), path=e.get('path'),
                 family=e.get('family', 'uncategorised'), map_id=e.get('map_id', 0))
        for e in raw.get('splits', {}).get(split, [])
    ]
    print(f"Phase 4: validating padding dims for {len(entries)} {split}-split maps...")
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

    smaclite_cfg  = config.env.get("smaclite", {})
    manifest_rel  = smaclite_cfg.get("dataset_manifest", "")
    manifest_path = str(ROOT / manifest_rel) if manifest_rel else ""
    manifest_split= smaclite_cfg.get("manifest_split", "train")
    map_mode      = smaclite_cfg.get("map_mode", "shuffled_round_robin")
    map_seed      = int(smaclite_cfg.get("map_seed", 42))

    raw = _load_phase4_raw(manifest_path) if manifest_path else {}

    # Build extra W&B config metadata
    pad_info = raw.get('padding', {})
    wb_extra = {
        "phase":            4,
        "dataset_name":     raw.get("dataset_name", ""),
        "dataset_hash":     raw.get("dataset_hash", ""),
        "manifest_path":    manifest_rel,
        "manifest_split":   manifest_split,
        "num_train_maps":   raw.get("split_counts", {}).get("train", 0),
        "num_val_maps":     raw.get("split_counts", {}).get("validation", 0),
        "num_test_maps":    raw.get("split_counts", {}).get("test", 0),
        "num_families":     len(raw.get("family_split_counts", {})),
        "sampler_mode":     map_mode,
        "map_seed":         map_seed,
        "padding_max_agents":   pad_info.get("max_agents"),
        "padding_max_enemies":  pad_info.get("max_enemies"),
        "padding_max_actions":  pad_info.get("max_actions"),
        "padding_max_obs_size": pad_info.get("max_obs_size"),
    }

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    if raw:
        pad_dims = _load_pad_dims(raw)
        _print_dataset_summary(raw, manifest_split, pad_dims, map_mode, map_seed)

    _init_wandb(config, extra_config=wb_extra)

    if raw:
        _validate_padding_at_startup(config, raw)

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
