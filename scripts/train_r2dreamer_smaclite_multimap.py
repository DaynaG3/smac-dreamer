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

# Reuse the exact Dreamer/buffer/trainer config from the debug script.
from train_r2dreamer_smaclite_debug import make_config as _make_debug_config  # noqa: E402

torch.set_float32_matmul_precision("high")


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


def _build_map_id_maps(entries):
    """Replicate SMACliteDreamerEnv's map_id assignment so the tracker's id->name matches the
    env's ``log_map_id``: stable map_id when all distinct (Phase 4), else sequential index."""
    ids = [e.map_id for e in entries]
    if len(set(ids)) == len(ids):
        id_to_name = {int(e.map_id): e.name for e in entries}
        id_to_family = {int(e.map_id): getattr(e, "family", "uncategorised") for e in entries}
    else:
        id_to_name = {i: e.name for i, e in enumerate(entries)}
        id_to_family = {i: getattr(e, "family", "uncategorised") for i, e in enumerate(entries)}
    return id_to_name, id_to_family


def main():
    ap = argparse.ArgumentParser(description="R2-Dreamer × SMAClite multimap training")
    ap.add_argument("--config", default="configs/multimap.yaml", help="multimap YAML config")
    ap.add_argument("--steps", type=int, default=None, help="override total env steps")
    ap.add_argument("--logdir", default=None, help="override logdir")
    ap.add_argument("--wandb-project", default=None, help="override W&B project")
    ap.add_argument("--wandb-entity", default=None, help="override W&B entity/user/team")
    ap.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"),
                    help="override W&B mode")
    ap.add_argument("--wandb-run-id", default=None,
                    help="resume logging to an EXISTING W&B run id (continues that run's history). "
                         "Pair with --step-offset so the global_step x-axis continues, not restarts.")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume model/training state")
    ap.add_argument("--resume-mode", default=None,
                    choices=("full", "weights_only", "transfer_reward"),
                    help="full (default) | weights_only | transfer_reward (overrides cfg.resume.mode)")
    ap.add_argument("--step-offset", type=int, default=None,
                    help="global-step offset (overrides cfg.resume.step_offset)")
    ap.add_argument("--actor-warmup-steps", type=int, default=None,
                    help="local steps with no actor updates (overrides cfg.resume.actor_warmup_steps)")
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

    # --- Full-observability ablation: optional sight-range override -------------
    # Export SMACLITE_SIGHT_RANGE BEFORE any env/discovery is built; spawn children (train
    # workers, validation env children, discovery probes) inherit it and apply the override via
    # smacdreamer.sight_range.maybe_override_sight_range(). Absent -> partial obs (sight 9).
    sight_range = None
    if cfg.get("observation") and cfg.observation.get("sight_range") is not None:
        sight_range = int(cfg.observation.sight_range)
        os.environ["SMACLITE_SIGHT_RANGE"] = str(sight_range)
        print(f"  [observation] full-visibility override: AGENT_SIGHT_RANGE -> {sight_range}")

    # --- Action masking (P0.1/P0.2): requires structured obs (per-agent avail + masks) -----
    action_masking = bool(cfg.get("action_masking", False))
    if action_masking and obs_mode != "structured":
        raise ValueError("action_masking requires observation.mode: structured")
    config.model.action_masking = action_masking
    config.model.mask_threshold = float(cfg.get("mask_threshold", 0.7))
    config.model.amp_dtype = resolve_amp_dtype(str(cfg.get("amp_dtype", "bfloat16")), str(cfg.device))
    run_cuda_preflight(str(cfg.device), str(config.model.amp_dtype))

    # --- Continuation / resume settings (mode, global-step offset, actor warm-up) -----
    resume_cfg = cfg.get("resume") or {}
    resume_mode = str(args.resume_mode or resume_cfg.get("mode", "full"))
    step_offset = int(args.step_offset if args.step_offset is not None else resume_cfg.get("step_offset", 0))
    actor_warmup_steps = int(args.actor_warmup_steps if args.actor_warmup_steps is not None
                             else resume_cfg.get("actor_warmup_steps", 0))
    config.trainer.step_offset = step_offset
    config.trainer.actor_warmup_steps = actor_warmup_steps

    # Guard: a continuation resume MUST carry a checkpoint. transfer_reward/weights_only load
    # weights from --resume, and a non-zero step_offset only makes sense when continuing an
    # existing run — refuse to silently start a fresh model with continuation settings.
    from smacdreamer.checkpoint_transfer import validate_resume_args
    validate_resume_args(resume_mode, step_offset, args.resume)
    print(f"  [resume] mode={resume_mode}  step_offset={step_offset}  "
          f"actor_warmup_steps={actor_warmup_steps}  resume_path={args.resume!r}")

    # --- Adaptive hard-map curriculum (prioritized_hard_maps) --------------------------
    mp_cfg = cfg.get("map_priority") or {}
    mp_enabled = bool(mp_cfg.get("enabled", False))
    hard_map_probability = float(mp_cfg.get("hard_map_probability", 0.25))
    if mp_enabled and str(cfg.sampling_mode) != "prioritized_hard_maps":
        raise ValueError(
            "map_priority.enabled=true requires sampling_mode: prioritized_hard_maps "
            f"(got {str(cfg.sampling_mode)!r})")
    if mp_enabled:
        print(f"  [map_priority] enabled  hard_map_probability={hard_map_probability}  "
              f"every={int(mp_cfg.get('every', 100000))}  warmup={int(mp_cfg.get('warmup', 100000))}  "
              f"ema_decay={float(mp_cfg.get('ema_decay', 0.98))}  "
              f"min_episodes={int(mp_cfg.get('min_episodes', 5))}")

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
        hard_map_probability=hard_map_probability,
    )
    print(f"  obs keys : {sorted(obs_space.spaces)}")

    # --- Logger: record resolved reward + padding into the run config ----------
    run_config = OmegaConf.create({
        "reward_name": reward_name,
        "reward_params_resolved": resolved,
        "reward_hash": rhash,
        "obs_mode": obs_mode,
        "sight_range": sight_range,
        "dataset_tag": dataset_tag,
        "explicit_folders": explicit,
        "padding": discovery["padding"],
        "split": (OmegaConf.to_container(cfg.split, resolve=True)
                  if cfg.get("split") else {"mode": "explicit_folders"}),
        "sampling_mode": str(cfg.sampling_mode),
        "n_train_maps": len(train_entries),
        "n_validation_maps": len(val_entries),
        "model": config.model,
    })

    # Reconstruction metadata for standalone checkpoint eval: the EXACT obs mode + model dims
    # used in training, written beside the checkpoint so eval rebuilds an identical model
    # regardless of which --config is passed later.
    run_meta = {
        "obs_mode": obs_mode,
        "sight_range": sight_range,
        "units": int(cfg.units), "deter": int(cfg.deter),
        "batch_size": int(cfg.batch_size), "batch_length": int(cfg.batch_length),
        "imag_horizon": int(cfg.imag_horizon),
        "max_episode_steps": int(cfg.max_episode_steps), "gamma": float(cfg.gamma),
        "reward_name": reward_name, "padding": discovery["padding"],
        "dataset_tag": dataset_tag, "explicit_folders": explicit,
        "validation_seeds": val_seeds,
        "maps_folder": str(maps_cfg.get("train", cfg.get("maps_folder", ""))),
    }
    (logdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    wandb_cfg = cfg.get("wandb") or {}
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project")
    wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or wandb_cfg.get("mode")
    wandb_run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID") or wandb_cfg.get("id")
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
        if wandb_run_id:
            # Resume logging to an existing run; its history continues. Pair with --step-offset so
            # the global_step x-axis continues forward instead of restarting at 0.
            wandb_kwargs["id"] = str(wandb_run_id)
            wandb_kwargs["resume"] = "allow"
            print(f"  [wandb] resuming existing run id={wandb_run_id} (global_step offset={step_offset})")
        logger = WandbLogger(
            logdir,
            project=str(wandb_project),
            run_name=run_name,
            config=run_config,
            step_offset=step_offset,
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
            from smacdreamer.checkpoint_transfer import transfer_reward_load, load_weights_only
            if resume_mode == "transfer_reward":
                transfer_reward_load(agent, args.resume)
            elif resume_mode == "weights_only":
                load_weights_only(agent, args.resume)
            else:  # full — existing behaviour preserved
                # weights_only=False: our own checkpoints carry RNG state (numpy/torch generators),
                # which PyTorch>=2.6's safe loader (weights_only=True default) refuses to unpickle.
                ckpt = torch.load(args.resume, map_location=str(cfg.device), weights_only=False)
                agent.load_state_dict(ckpt["agent_state_dict"])
                if ckpt.get("agent_training_state") and hasattr(agent, "load_training_state_dict"):
                    agent.load_training_state_dict(ckpt["agent_training_state"])
                    print(f"  [resume:full] restored model + optimizer/scheduler/scaler from {args.resume}")
                else:
                    print(f"  [resume:full] restored model weights only from {args.resume}")
            print(f"  [resume] mode={resume_mode} step_offset={step_offset} "
                  f"actor_warmup_steps={actor_warmup_steps}")
            print("  [resume] replay restore not implemented; replay refills before updates.")

        # --- Checkpointing -----------------------------------------------------
        checkpointer = None
        if float(cfg.get("checkpoint_every_minutes", 0)) > 0:
            checkpointer = PeriodicCheckpointer(
                agent, logdir,
                interval_seconds=float(cfg.checkpoint_every_minutes) * 60.0,
                # GLOBAL step for checkpoint metadata/snapshot names = trainer's actual local
                # step + offset (NOT replay count, which saturates at capacity). The trainer
                # publishes this on the agent each loop; fall back to step_offset pre-loop.
                step_fn=lambda: int(getattr(agent, "_smacdreamer_global_step", step_offset)),
            )
            attach_checkpointing(agent, checkpointer)
            print(f"  Checkpoints : every {cfg.checkpoint_every_minutes:g} min -> {logdir/'latest.pt'}")

        # --- Adaptive hard-map curriculum tracker (attached to the agent) ------
        if mp_enabled:
            from smacdreamer.map_priority import MapPriorityTracker
            _id_to_name, _id_to_family = _build_map_id_maps(train_entries)
            agent._map_priority_tracker = MapPriorityTracker(
                id_to_name=_id_to_name,
                id_to_family=_id_to_family,
                every=int(mp_cfg.get("every", 100000)),
                warmup=int(mp_cfg.get("warmup", 100000)),
                ema_decay=float(mp_cfg.get("ema_decay", 0.98)),
                min_episodes=int(mp_cfg.get("min_episodes", 5)),
                hard_map_probability=hard_map_probability,
            )
            print(f"  [map_priority] tracker attached over {len(_id_to_name)} train maps")

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
            step_offset=step_offset,
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
