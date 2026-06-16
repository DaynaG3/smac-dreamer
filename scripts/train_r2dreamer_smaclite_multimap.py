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
from trainer import OnlineTrainer
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_envs
from smacdreamer.wandb_logger import WandbLogger
from smacdreamer.checkpointing import PeriodicCheckpointer, attach_checkpointing
from smacdreamer.envs.reward_registry import resolved_params

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


def main():
    ap = argparse.ArgumentParser(description="R2-Dreamer × SMAClite multimap training")
    ap.add_argument("--config", default="configs/multimap.yaml", help="multimap YAML config")
    ap.add_argument("--steps", type=int, default=None, help="override total env steps")
    ap.add_argument("--logdir", default=None, help="override logdir")
    args = ap.parse_args()

    cfg = OmegaConf.load(str((ROOT / args.config) if not pathlib.Path(args.config).is_absolute() else args.config))
    steps = int(args.steps if args.steps is not None else cfg.steps)
    logdir = pathlib.Path(args.logdir or cfg.get("logdir", "logs/r2dreamer/multimap"))
    logdir.mkdir(parents=True, exist_ok=True)

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
    # Wire periodic held-out eval.
    eval_every = int(cfg.eval.get("every", 0))
    episodes_per_map = int(cfg.eval.get("episodes_per_map", 0))
    config.trainer.eval_every = eval_every if eval_every > 0 else steps + 1
    config.trainer.eval_episode_num = episodes_per_map

    # --- Observation mode (flat | structured) ----------------------------------
    obs_mode = str(cfg.observation.mode) if cfg.get("observation") else "flat"
    if obs_mode not in ("flat", "structured"):
        raise ValueError(f"observation.mode must be 'flat' or 'structured', got {obs_mode!r}")

    # --- Resolve reward for logging + hash -------------------------------------
    reward_name = str(cfg.reward.name)
    reward_params = OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True) or {}
    resolved = resolved_params(reward_name, reward_params)
    rhash = _reward_hash(reward_name, resolved)

    folder_tag = pathlib.Path(str(cfg.maps_folder)).name
    run_name = cfg.wandb.get("run_name") or f"{folder_tag}-{reward_name}-{rhash}"

    print(f"\n{'='*64}")
    print("R2-Dreamer × SMAClite  —  MULTIMAP training")
    print(f"{'='*64}")
    print(f"  maps_folder : {cfg.maps_folder}")
    print(f"  reward      : {reward_name}  (hash {rhash})")
    print(f"  resolved    : {resolved}")
    print(f"  sampling    : {cfg.sampling_mode}")
    print(f"  steps       : {steps}   env_num: {cfg.env_num}   device: {cfg.device}")
    print(f"  eval        : every {eval_every} steps, {episodes_per_map} episodes")
    print(f"  run_name    : {run_name}")
    print(f"{'='*64}\n")

    tools.set_seed_everywhere(int(cfg.seed))

    # --- Environments (discovery happens inside the factory) -------------------
    print("Discovering maps and creating environments...")
    train_envs, eval_envs, obs_space, act_space, discovery = make_smaclite_multimap_envs(
        maps_folder=str(cfg.maps_folder),
        split_spec=OmegaConf.to_container(cfg.split, resolve=True),
        env_num=int(cfg.env_num),
        eval_episode_num=episodes_per_map if eval_every > 0 else 0,
        device=str(cfg.device),
        sampling_mode=str(cfg.sampling_mode),
        reward_name=reward_name,
        reward_params=reward_params,
        gamma=float(cfg.gamma),
        max_episode_steps=int(cfg.max_episode_steps),
        seed=int(cfg.seed),
        padding_override=OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None,
        obs_mode=obs_mode,
    )
    print(f"  obs_mode : {obs_mode}")
    print(f"  obs keys : {sorted(obs_space.spaces)}")
    print(f"  train maps: {discovery['n_train']}   held-out test maps: {discovery['n_test']}")

    # --- Logger: record resolved reward + padding into the run config ----------
    run_config = OmegaConf.create({
        "reward_name": reward_name,
        "reward_params_resolved": resolved,
        "reward_hash": rhash,
        "obs_mode": obs_mode,
        "padding": discovery["padding"],
        "split": OmegaConf.to_container(cfg.split, resolve=True),
        "sampling_mode": str(cfg.sampling_mode),
        "n_train_maps": discovery["n_train"],
        "n_test_maps": discovery["n_test"],
        "model": config.model,
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
        "maps_folder": str(cfg.maps_folder),
    }
    (logdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    wandb_project = cfg.wandb.get("project")
    if wandb_project:
        logger = WandbLogger(logdir, project=wandb_project, run_name=run_name, config=run_config)
    else:
        logger = tools.Logger(logdir)
        # Persist the run config alongside TensorBoard/JSONL so configs are distinguishable.
        (logdir / "run_config.json").write_text(
            json.dumps(OmegaConf.to_container(run_config, resolve=True), indent=2, default=str),
            encoding="utf-8",
        )

    replay_buffer = Buffer(config.buffer)

    # --- Agent -----------------------------------------------------------------
    print("\nBuilding Dreamer agent...")
    agent = Dreamer(config.model, obs_space, act_space).to(config.device)
    print(f"  Parameters : {sum(p.numel() for p in agent.parameters()):,}")

    # --- Checkpointing ---------------------------------------------------------
    checkpointer = None
    if float(cfg.get("checkpoint_every_minutes", 0)) > 0:
        checkpointer = PeriodicCheckpointer(
            agent, logdir,
            interval_seconds=float(cfg.checkpoint_every_minutes) * 60.0,
            step_fn=lambda: replay_buffer.count() * config.trainer.action_repeat,
        )
        attach_checkpointing(agent, checkpointer)
        print(f"  Checkpoints : every {cfg.checkpoint_every_minutes:g} min -> {logdir/'latest.pt'}")

    # --- Train -----------------------------------------------------------------
    print(f"\nStarting multimap training ({steps} env steps)...\n")
    trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    trainer.begin(agent)

    if checkpointer is not None:
        checkpointer.save(final=True)
    else:
        torch.save({"agent_state_dict": agent.state_dict()}, logdir / "latest.pt")
        print(f"\nCheckpoint saved -> {logdir/'latest.pt'}")

    if wandb_project:
        logger.finish()
    print("\nMultimap training complete.")


if __name__ == "__main__":
    main()
