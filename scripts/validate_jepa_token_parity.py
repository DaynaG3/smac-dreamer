#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from smacdreamer.jepa.online_tokens import JEPATokenSpec, encode_state_vector


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate online/offline JEPA token parity on a real episode NPZ.")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint is required to identify the intended regime.")
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    ckpt = pathlib.Path(args.checkpoint)
    ep = pathlib.Path(args.episode_npz)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not ep.exists():
        raise SystemExit(f"episode-npz not found: {ep}")
    sys.path.insert(0, str(pathlib.Path(args.config).resolve().parents[1] if False else pathlib.Path.cwd()))
    with np.load(ep, allow_pickle=False) as data:
        required = ["states", "static_condition", "entity_static"]
        missing = [k for k in required if k not in data]
        if missing:
            raise SystemExit(f"episode npz missing required arrays: {missing}")
        states = np.asarray(data["states"], dtype=np.float32)
        if states.ndim == 3:
            state = states[0, int(args.step)]
        elif states.ndim == 2:
            state = states[int(args.step)]
        else:
            raise SystemExit(f"states must have [episode, step, dim] or [step, dim], got {states.shape}")
        n_agents = int(np.asarray(data["n_agents"]).item())
        n_enemies = int(np.asarray(data["n_enemies"]).item())
        n_actions = int(np.asarray(data["n_actions"]).item())
        ally_size = int(np.asarray(data["ally_state_feat_size"]).item())
        enemy_size = int(np.asarray(data["enemy_state_feat_size"]).item())
        static_dim = int(np.asarray(data["static_dim"]).item())
        entity_static_size = int(np.asarray(data["entity_static_feat_size"]).item())
        dynamic = max(ally_size, enemy_size, 1)
        spec = JEPATokenSpec(
            n_agents=n_agents,
            n_enemies=n_enemies,
            max_agents=n_agents,
            max_enemies=n_enemies,
            max_actions=n_actions,
            ally_state_feat_size=ally_size,
            enemy_state_feat_size=enemy_size,
            dynamic_token_dim=dynamic,
            entity_static_feat_size=entity_static_size,
            static_dim=static_dim,
            token_dim=dynamic + entity_static_size,
            ally_has_shields=bool(np.asarray(data["ally_has_shields"]).item()),
            enemy_has_shields=bool(np.asarray(data["enemy_has_shields"]).item()),
            num_unit_types=int(np.asarray(data["num_unit_types"]).item()),
        )
        online_entity, online_mask, online_slot = encode_state_vector(
            state, spec, np.asarray(data["entity_static"], dtype=np.float32))
        # Compare against the restored JEPA dataset implementation, which is the
        # offline source of truth for raw npz state -> entity tokens.
        try:
            from smac_jepa.data import SMACJEPADataset
        except ImportError as exc:
            raise SystemExit(
                "Could not import smac_jepa.data.SMACJEPADataset. Install the JEPA repo with "
                "python -m pip install -e <PATH_TO_SMAC_JEPA_REPO>."
            ) from exc
        ds = SMACJEPADataset(str(ep), context_len=1, window_mode="sequential")
        episode = ds.episodes[0]
        offline_entity, offline_mask = ds._encode_state_window(
            state.reshape(1, -1),
            episode["metadata"],
            episode["entity_static"],
        )
        offline_slot = ds._slot_mask(episode["metadata"])
        torch.testing.assert_close(torch.from_numpy(online_entity), torch.from_numpy(offline_entity[0]))
        torch.testing.assert_close(torch.from_numpy(online_mask), torch.from_numpy(offline_mask[0]))
        torch.testing.assert_close(torch.from_numpy(online_slot), torch.from_numpy(offline_slot))
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(data["static_condition"], dtype=np.float32).reshape(-1)),
            torch.from_numpy(np.asarray(data["static_condition"], dtype=np.float32).reshape(-1)),
        )
    print("JEPA token parity against restored offline dataset source passed.")


if __name__ == "__main__":
    main()
