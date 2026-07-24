"""R2-Dreamer × SMAClite MULTIMAP training (Phase 2/3 — generalisation study).

Trains one centralised Dreamer agent across many maps discovered from a folder, with a
config-driven train / held-out-test split and a swappable (denser) reward. Periodically
evaluates on the HELD-OUT test maps using the ORIGINAL (unshaped) reward + win rate.

Reuses the model/buffer/trainer config builder from train_r2dreamer_smaclite_debug.py so
the Dreamer hyperparameters stay identical; only the env construction (multimap factory),
eval cadence, and reward/padding logging differ.

Usage (smac-r2 conda env, from project root):
    python scripts\\train_r2dreamer_smaclite_multimap.py --config configs\\multimap.yaml
    python scripts\\train_r2dreamer_smaclite_multimap.py --config configs\\multimap.yaml --steps 500

Acceptance (this script):
  * discovery prints train/test split + resolved padding (TRAIN-max or override).
  * WM losses + log_* (incl. invalid-action + per-term log_reward_*) appear in logs.
  * periodic held-out eval logs episode/eval_battle_won + episode/eval_reward_original.
  * run config records resolved reward name+params + padding; run name carries a
    resolved-params hash so configs are distinguishable.
  * latest.pt written; no crash for the full run.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_envs
from smacdreamer.envs.map_discovery import discover, discover_folders, SplitSpec
from smacdreamer.validation_trainer import ValidationTrainer
from smacdreamer.wandb_logger import WandbLogger
from smacdreamer.checkpointing import PeriodicCheckpointer, attach_checkpointing
from smacdreamer.envs.reward_registry import resolved_params
from smacdreamer.cuda_preflight import resolve_amp_dtype, run_cuda_preflight
from smacdreamer.jepa.online_tokens import JEPAVisibilityConfig

# Reuse the exact Dreamer/buffer/trainer config from the debug script.
from train_r2dreamer_smaclite_debug import make_config as _make_debug_config  # noqa: E402

torch.set_float32_matmul_precision("high")


def _read_jepa_checkpoint_runtime_config(path: pathlib.Path) -> tuple[dict, JEPAVisibilityConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"JEPA checkpoint must be a dict: {path}")
    metadata = dict(checkpoint.get("metadata", {}))
    cfg = dict(checkpoint.get("resolved_config", checkpoint.get("config", {})))
    visibility = JEPAVisibilityConfig(
        enemy_visibility_mask=bool(cfg.get("enemy_visibility_mask", metadata.get("enemy_visibility_mask", False))),
        enemy_sight_range=float(cfg.get("enemy_sight_range", metadata.get("enemy_sight_range", 9.0))),
        xy_indices=tuple(cfg.get("xy_indices", metadata.get("visibility_xy_indices", (2, 3)))),
    )
    live_metadata = dict(metadata)
    live_metadata.setdefault("latent_dim", cfg.get("latent_dim"))
    live_metadata.setdefault("memory_dim", cfg.get("rollout_memory_dim", cfg.get("memory_dim")))
    live_metadata.setdefault("action_conditioned_memory", bool(cfg.get("action_conditioned_memory", False)))
    live_metadata.update(visibility.metadata())
    live_metadata.setdefault("latent_normalization", cfg.get("latent_normalization", cfg.get("latent_normalize", "none")))
    return live_metadata, visibility


def _propagate_device(node, device: str) -> None:
    """Recursively set every `device`/`storage_device` field in an OmegaConf tree.

    The reused debug config builder writes device="cpu" into nested buffer/encoder/
    head blocks that the multimap script must override for a GPU run. Walks dicts and
    lists in place so the whole config agrees on one device.
    """
    if OmegaConf.is_dict(node):
        for key in list(node.keys()):
            if key in ("device", "storage_device"):
                node[key] = device
            else:
                _propagate_device(node[key], device)
    elif OmegaConf.is_list(node):
        for item in node:
            _propagate_device(item, device)


def _reward_hash(name: str, resolved: dict) -> str:
    """Stable 8-char hash over the FULLY-resolved reward (name + resolved params).

    Computed on resolved params (defaults filled) so identical effective configs hash
    identically regardless of which fields the user left implicit.
    """
    blob = json.dumps({"name": name, "params": resolved}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser(description="R2-Dreamer × SMAClite multimap training")
    ap.add_argument("--config", default="configs/multimap.yaml", help="multimap YAML config")
    ap.add_argument("--steps", type=int, default=None, help="override total env steps")
    ap.add_argument("--logdir", default=None, help="override logdir")
    ap.add_argument("--wandb-project", default=None, help="override W&B project")
    ap.add_argument("--wandb-entity", default=None, help="override W&B entity/user/team")
    ap.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"),
                    help="override W&B mode")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume model/training state")
    ap.add_argument("--jepa-checkpoint", default=None, help="override world_model.jepa.checkpoint")
    args = ap.parse_args()

    cfg = OmegaConf.load(str((ROOT / args.config) if not pathlib.Path(args.config).is_absolute() else args.config))
    steps = int(args.steps if args.steps is not None else cfg.steps)
    logdir = pathlib.Path(args.logdir or cfg.get("logdir", "logs/r2dreamer/multimap"))
    logdir.mkdir(parents=True, exist_ok=True)
    train_envs = None
    logger = None
    replay_buffer = None
    checkpointer = None
    wandb_project = None

    # --- Build the Dreamer/buffer/trainer config (reuse debug builder) ---------
    debug_args = argparse.Namespace(
        steps=steps, batch_size=int(cfg.batch_size), batch_length=int(cfg.batch_length),
        units=int(cfg.units), deter=int(cfg.deter), imag_horizon=int(cfg.imag_horizon),
    )
    config = _make_debug_config(debug_args)
    # The debug builder hard-codes device="cpu" in MANY nested places (buffer,
    # storage, encoder, decoder, and every head), not just the three top-level
    # fields. On CPU that is invisible; on GPU the model is .to(device)'d but
    # some submodules read their own `device` field at forward time, so every
    # field must agree or you get a CUDA/CPU mismatch. Propagate to all of them.
    _propagate_device(config, str(cfg.device))

    # --- Replay buffer: large capacity + CPU storage (model still computes on cfg.device).
    # storage_device is overridden AFTER _propagate_device (which set every device field to
    # cfg.device); the buffer pins+moves sampled batches to config.buffer.device on sample().
    _buf_cfg = cfg.get("buffer") or {}
    config.buffer.max_size = int(_buf_cfg.get("max_size", config.buffer.max_size))
    config.buffer.storage_device = str(_buf_cfg.get("storage_device", config.buffer.storage_device))
    config.buffer.storage_backend = str(_buf_cfg.get("storage_backend", "tensor"))
    scratch_cfg = _buf_cfg.get("scratch_dir", "replay")
    scratch_path = pathlib.Path(str(scratch_cfg))
    if not scratch_path.is_absolute():
        scratch_path = logdir / scratch_path
    config.buffer.scratch_dir = str(scratch_path)

    # --- Observation mode (flat | structured) ----------------------------------
    obs_mode = str(cfg.observation.mode) if cfg.get("observation") else "flat"
    if obs_mode not in ("flat", "structured"):
        raise ValueError(f"observation.mode must be 'flat' or 'structured', got {obs_mode!r}")

    # --- Action masking (P0.1/P0.2): requires structured obs (per-agent avail + masks) -----
    action_masking = bool(cfg.get("action_masking", False))
    if action_masking and obs_mode != "structured":
        raise ValueError("action_masking requires observation.mode: structured")
    config.model.action_masking = action_masking
    config.model.mask_threshold = float(cfg.get("mask_threshold", 0.7))
    wm_cfg = cfg.get("world_model") or {}
    wm_backend = str(wm_cfg.get("backend", "rssm")).lower()
    if wm_backend not in ("rssm", "jepa"):
        raise ValueError(f"world_model.backend must be 'rssm' or 'jepa', got {wm_backend!r}")
    config.model.world_model = OmegaConf.create(OmegaConf.to_container(wm_cfg, resolve=True) if wm_cfg else {})
    config.model.world_model.backend = wm_backend
    jepa_visibility_config = None
    jepa_live_metadata = None
    if wm_backend == "jepa":
        if obs_mode != "structured":
            raise ValueError("world_model.backend='jepa' requires observation.mode: structured")
        if bool(config.model.world_model.get("jepa", {}).get("freeze_core", True)) is False:
            raise NotImplementedError("JEPA online fine-tuning is not implemented; set freeze_core: true")
        if bool(getattr(config.model, "compile", False)):
            raise NotImplementedError("torch.compile is not enabled for the JEPA backend in this branch")
        jepa_cfg = config.model.world_model.get("jepa") or OmegaConf.create({})
        if args.jepa_checkpoint:
            jepa_cfg.checkpoint = args.jepa_checkpoint
        if not jepa_cfg.get("checkpoint"):
            raise ValueError("world_model.backend='jepa' requires world_model.jepa.checkpoint or --jepa-checkpoint")
        ckpt_path = pathlib.Path(str(jepa_cfg.checkpoint))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"JEPA checkpoint not found: {ckpt_path}")
        jepa_live_metadata, jepa_visibility_config = _read_jepa_checkpoint_runtime_config(ckpt_path)
        jepa_cfg.live_metadata = OmegaConf.create(jepa_live_metadata)
        config.model.world_model.jepa = jepa_cfg
    config.model.amp_dtype = resolve_amp_dtype(str(cfg.get("amp_dtype", "bfloat16")), str(cfg.device))
    run_cuda_preflight(str(cfg.device), str(config.model.amp_dtype))

    # --- Validation cadence + fixed seeds (explicit seed list, NOT a worker count) -----
    val_cfg = cfg.get("validation") or {}
    _eval_cfg = cfg.get("eval") or {}
    val_every = int(val_cfg.get("every", _eval_cfg.get("every", 0)))
    val_run_at_start = bool(val_cfg.get("run_at_start", False))
    if val_cfg.get("seeds") is not None:
        val_seeds = [int(s) for s in OmegaConf.to_container(val_cfg.seeds, resolve=True)]
    elif _eval_cfg.get("fixed_seeds") is not None:
        val_seeds = [int(s) for s in OmegaConf.to_container(_eval_cfg.fixed_seeds, resolve=True)]
    else:
        val_seeds = [0, 1, 2]
    config.trainer.eval_every = val_every if val_every > 0 else steps + 1
    config.trainer.eval_episode_num = 1   # sentinel >0 so ValidationTrainer.eval() fires
    config.trainer.system_log_every = int(cfg.get("system_log_every", 10_000))

    # --- Dataset: explicit train/validation folders (no ratio split) OR legacy split ----
    padding_override = OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None
    maps_cfg = cfg.get("maps") or {}
    explicit = bool(maps_cfg.get("train"))
    print("Discovering maps (train + validation only; blind splits untouched) ...")
    if explicit:
        train_entries, val_entries, pad_dims = discover_folders(
            str(maps_cfg.train), str(maps_cfg.validation),
            padding_override=padding_override, obs_mode=obs_mode, isolate_probe=True, verbose=True,
        )
        dataset_tag = pathlib.Path(str(maps_cfg.train)).parent.name or "dataset"
    else:
        train_entries, val_entries, pad_dims = discover(
            str(cfg.maps_folder),
            SplitSpec(**OmegaConf.to_container(cfg.split, resolve=True)),
            padding_override=padding_override, obs_mode=obs_mode, isolate_probe=True, verbose=True,
        )
        dataset_tag = pathlib.Path(str(cfg.maps_folder)).name

    # --- Resolve reward for logging + hash -------------------------------------
    reward_name = str(cfg.reward.name)
    reward_params = OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True) or {}
    resolved = resolved_params(reward_name, reward_params)
    rhash = _reward_hash(reward_name, resolved)
    run_name = cfg.wandb.get("run_name") or f"{dataset_tag}-{reward_name}-{rhash}"

    print(f"\n{'='*64}")
    print("R2-Dreamer × SMAClite  —  MULTIMAP training")
    print(f"{'='*64}")
    print(f"  dataset    : {dataset_tag}  (explicit folders: {explicit})")
    print(f"  obs_mode   : {obs_mode}")
    print(f"  reward     : {reward_name}  (hash {rhash})")
    print(f"  train maps : {len(train_entries)}   validation maps: {len(val_entries)}")
    print(f"  validation : every {val_every} steps, seeds {val_seeds}")
    print(f"  val start  : {val_run_at_start}")
    print(f"  world_model: {wm_backend}")
    if wm_backend == "jepa":
        print(f"  jepa ckpt  : {config.model.world_model.jepa.checkpoint}")
        print(f"  jepa vis   : {jepa_visibility_config}")
    print(f"  amp_dtype  : {config.model.amp_dtype}")
    print(f"  replay     : backend={config.buffer.storage_backend} capacity={config.buffer.max_size} "
          f"storage_device={config.buffer.storage_device} scratch={config.buffer.scratch_dir}")
    print(f"  steps      : {steps}   env_num: {cfg.env_num}   device: {cfg.device}")
    print(f"  run_name   : {run_name}")
    print(f"{'='*64}\n")

    tools.set_seed_everywhere(int(cfg.seed))

    # --- Train envs ONLY (validation handled by ValidationTrainer; no worker-eval pool) -
    env_lifecycle = OmegaConf.to_container(cfg.get("env_lifecycle", {}), resolve=True) or {}
    train_envs, eval_envs, obs_space, act_space, discovery = make_smaclite_multimap_envs(
        maps_folder=str(maps_cfg.get("train", cfg.get("maps_folder", ""))),
        split_spec={},
        env_num=int(cfg.env_num),
        eval_episode_num=0,
        device=str(cfg.device),
        sampling_mode=str(cfg.sampling_mode),
        reward_name=reward_name,
        reward_params=reward_params,
        gamma=float(cfg.gamma),
        max_episode_steps=int(cfg.max_episode_steps),
        seed=int(cfg.seed),
        padding_override=padding_override,
        obs_mode=obs_mode,
        train_entries=train_entries, test_entries=val_entries, pad_dims=pad_dims,
        env_lifecycle=env_lifecycle,
        include_jepa_obs=(wm_backend == "jepa"),
        jepa_visibility_config=jepa_visibility_config,
    )
    print(f"  obs keys : {sorted(obs_space.spaces)}")

    # --- Logger: record resolved reward + padding into the run config ----------
    run_config = OmegaConf.create({
        "reward_name": reward_name,
        "reward_params_resolved": resolved,
        "reward_hash": rhash,
        "obs_mode": obs_mode,
        "dataset_tag": dataset_tag,
        "explicit_folders": explicit,
        "padding": discovery["padding"],
        "split": (OmegaConf.to_container(cfg.split, resolve=True)
                  if cfg.get("split") else {"mode": "explicit_folders"}),
        "sampling_mode": str(cfg.sampling_mode),
        "n_train_maps": len(train_entries),
        "n_validation_maps": len(val_entries),
        "model": config.model,
        "world_model": config.model.world_model,
    })

    # Reconstruction metadata for standalone checkpoint eval: the EXACT obs mode + model dims
    # used in training, written beside the checkpoint so eval rebuilds an identical model
    # regardless of which --config is passed later.
    run_meta = {
        "obs_mode": obs_mode,
        "units": int(cfg.units), "deter": int(cfg.deter),
        "batch_size": int(cfg.batch_size), "batch_length": int(cfg.batch_length),
        "imag_horizon": int(cfg.imag_horizon),
        "max_episode_steps": int(cfg.max_episode_steps), "gamma": float(cfg.gamma),
        "reward_name": reward_name, "padding": discovery["padding"],
        "dataset_tag": dataset_tag, "explicit_folders": explicit,
        "validation_seeds": val_seeds,
        "maps_folder": str(maps_cfg.get("train", cfg.get("maps_folder", ""))),
        "world_model_backend": wm_backend,
    }
    if wm_backend == "jepa":
        from smacdreamer.jepa.checkpoint import sha256_file
        ckpt_path = pathlib.Path(str(config.model.world_model.jepa.checkpoint))
        run_meta["jepa_checkpoint"] = str(ckpt_path)
        run_meta["jepa_checkpoint_sha256"] = sha256_file(ckpt_path) if ckpt_path.exists() else None
        run_meta["jepa_visibility"] = jepa_visibility_config.metadata() if jepa_visibility_config is not None else None
    (logdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    wandb_cfg = cfg.get("wandb") or {}
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project")
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or wandb_cfg.get("mode")
    wandb_tags = wandb_cfg.get("tags")
    if wandb_tags is not None:
        wandb_tags = list(OmegaConf.to_container(wandb_tags, resolve=True))
    if wandb_project:
        wandb_kwargs = {}
        if wandb_entity:
            wandb_kwargs["entity"] = str(wandb_entity)
        if wandb_mode:
            wandb_kwargs["mode"] = str(wandb_mode)
        if wandb_tags:
            wandb_kwargs["tags"] = wandb_tags
        logger = WandbLogger(
            logdir,
            project=str(wandb_project),
            run_name=run_name,
            config=run_config,
            **wandb_kwargs,
        )
    else:
        logger = tools.Logger(logdir)
        # Persist the run config alongside TensorBoard/JSONL so configs are distinguishable.
        (logdir / "run_config.json").write_text(
            json.dumps(OmegaConf.to_container(run_config, resolve=True), indent=2, default=str),
            encoding="utf-8",
        )

    try:
        replay_buffer = Buffer(config.buffer)

        # --- Agent -------------------------------------------------------------
        print("\nBuilding Dreamer agent...")
        agent = Dreamer(config.model, obs_space, act_space).to(config.device)
        print(f"  Parameters : {sum(p.numel() for p in agent.parameters()):,}")
        if args.resume:
            ckpt = torch.load(args.resume, map_location=str(cfg.device), weights_only=False)
            agent.load_state_dict(ckpt["agent_state_dict"])
            if ckpt.get("agent_training_state") and hasattr(agent, "load_training_state_dict"):
                agent.load_training_state_dict(ckpt["agent_training_state"])
                print(f"  [resume] restored model + optimizer/scheduler/scaler state from {args.resume}")
            else:
                print(f"  [resume] restored model weights only from {args.resume}")
            print("  [resume] replay memmap restore is not implemented; replay will refill before updates.")

        # --- Checkpointing -----------------------------------------------------
        checkpointer = None
        if float(cfg.get("checkpoint_every_minutes", 0)) > 0:
            checkpointer = PeriodicCheckpointer(
                agent, logdir,
                interval_seconds=float(cfg.checkpoint_every_minutes) * 60.0,
                step_fn=lambda: replay_buffer.count() * config.trainer.action_repeat,
            )
            attach_checkpointing(agent, checkpointer)
            print(f"  Checkpoints : every {cfg.checkpoint_every_minutes:g} min -> {logdir/'latest.pt'}")

        # --- Train -------------------------------------------------------------
        print(f"\nStarting multimap training ({steps} env steps)...\n")
        # ValidationTrainer replaces the old worker-based periodic evaluator: every `val_every`
        # steps it runs map×seed validation, logs macro/micro metrics, and saves
        # best_val_macro_winrate.pt (macro win rate; macro original return as tie-breaker).
        trainer = ValidationTrainer(
            config.trainer, replay_buffer, logger, logdir, train_envs,
            validation_entries=val_entries, pad_dims=pad_dims, seeds=val_seeds,
            device=str(cfg.device), gamma=float(cfg.gamma),
            max_episode_steps=int(cfg.max_episode_steps), obs_mode=obs_mode,
            run_at_start=val_run_at_start,
            shutdown_timeout_seconds=float(env_lifecycle.get("shutdown_timeout_seconds", 5.0)),
        )
        trainer.begin(agent)

        if checkpointer is not None:
            checkpointer.save(final=True)
        else:
            torch.save({"agent_state_dict": agent.state_dict()}, logdir / "latest.pt")
            print(f"\nCheckpoint saved -> {logdir/'latest.pt'}")
        print("\nMultimap training complete.")
    finally:
        if train_envs is not None:
            train_envs.close()
        if replay_buffer is not None and hasattr(replay_buffer, "close"):
            replay_buffer.close()
        if wandb_project and logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
