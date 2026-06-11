"""Build ONE multimap SMAClite env in the MAIN process (no worker) to surface the
real construction error that ParallelEnv's worker swallows as "Lost connection".

The multimap factory builds envs inside spawn workers, so a crash there only shows
up in the parent as EOFError / "Lost connection to worker". Running the identical
constructor in-process means:
  * a Python exception prints its real traceback here, and
  * a native crash prints "Segmentation fault" / an OMP/SDL abort message,
pinpointing the failing library.

Usage (smac-r2 env, from project root):
    python scripts/debug_build_one_env.py --config configs/multimap_gpu.yaml
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "external" / "r2dreamer"), str(ROOT / "external" / "smaclite")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf

from smacdreamer.envs.map_discovery import discover, SplitSpec
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimap_gpu.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(str(ROOT / args.config if not pathlib.Path(args.config).is_absolute() else args.config))

    print(">> discovering maps (parent process)...")
    split = OmegaConf.to_container(cfg.split, resolve=True)
    pad_override = OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None
    train_entries, test_entries, pad_dims = discover(
        str(cfg.maps_folder), SplitSpec(**split), padding_override=pad_override, verbose=True,
        isolate_probe=True,   # subprocess-isolated probe, same as the training factory
    )
    print(f">> train maps: {len(train_entries)}  test maps: {len(test_entries)}")
    print(f">> pad_dims: {pad_dims}")
    if not train_entries:
        raise SystemExit("No TRAIN maps discovered — check maps_folder / split. "
                         f"maps_folder={cfg.maps_folder!r}")

    print(">> building ONE env IN-PROCESS (this is where the worker was dying)...")
    env = make_smaclite_multimap_env(
        train_entries, pad_dims, str(cfg.sampling_mode), int(cfg.seed), 0,
        str(cfg.reward.name), OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True),
        float(cfg.gamma), int(cfg.max_episode_steps),
    )
    print(">> env constructed OK")
    print(">> observation_space:", env.observation_space)
    print(">> action_space:", env.action_space)

    print(">> reset()...")
    obs = env.reset()
    keys = sorted(obs[0].keys()) if isinstance(obs, tuple) else sorted(obs.keys())
    print(">> reset OK; obs keys:", keys)
    print("\nSUCCESS — single-process env construction works. The crash is worker/spawn-specific "
          "(try env_num: 1, or the threading env-vars in the docs).")


if __name__ == "__main__":
    main()
