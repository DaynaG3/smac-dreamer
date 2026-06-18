#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint
from smacdreamer.jepa.online_tokens import JEPATokenSpec, encode_state_vector
from smacdreamer.jepa.world_model import FrozenJEPAWorldModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare original JEPA runtime and R2 JEPA wrapper on a real batch.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not pathlib.Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not pathlib.Path(args.episode_npz).exists():
        raise SystemExit(f"episode-npz not found: {args.episode_npz}")
    with np.load(args.episode_npz, allow_pickle=False) as data:
        n_agents = int(np.asarray(data["n_agents"]).item())
        n_enemies = int(np.asarray(data["n_enemies"]).item())
        n_actions = int(np.asarray(data["n_actions"]).item())
        ally_size = int(np.asarray(data["ally_state_feat_size"]).item())
        enemy_size = int(np.asarray(data["enemy_state_feat_size"]).item())
        static_dim = int(np.asarray(data["static_dim"]).item())
        entity_static_size = int(np.asarray(data["entity_static_feat_size"]).item())
        dynamic = max(ally_size, enemy_size, 1)
        spec = JEPATokenSpec(
            n_agents=n_agents, n_enemies=n_enemies, max_agents=n_agents, max_enemies=n_enemies,
            max_actions=n_actions, ally_state_feat_size=ally_size, enemy_state_feat_size=enemy_size,
            dynamic_token_dim=dynamic, entity_static_feat_size=entity_static_size,
            static_dim=static_dim, token_dim=dynamic + entity_static_size,
            ally_has_shields=bool(np.asarray(data["ally_has_shields"]).item()),
            enemy_has_shields=bool(np.asarray(data["enemy_has_shields"]).item()),
            num_unit_types=int(np.asarray(data["num_unit_types"]).item()),
        )
        core, memory, info = load_frozen_jepa_checkpoint(
            args.checkpoint, map_location=args.device, live_metadata=spec.metadata())
        wm = FrozenJEPAWorldModel(core=core, memory_module=memory, info=info, feature_dim=32).to(args.device)
        states = np.asarray(data["states"], dtype=np.float32)
        state = states[0, 0] if states.ndim == 3 else states[0]
        entity, mask, slot = encode_state_vector(state, spec, np.asarray(data["entity_static"], dtype=np.float32))
        obs = {
            "jepa_entity": torch.from_numpy(entity).unsqueeze(0).to(args.device),
            "jepa_entity_mask": torch.from_numpy(mask).unsqueeze(0).to(args.device),
            "jepa_entity_slot_mask": torch.from_numpy(slot).unsqueeze(0).to(args.device),
            "jepa_static_condition": torch.from_numpy(np.asarray(data["static_condition"], dtype=np.float32)).reshape(1, -1).to(args.device),
        }
        encoded = wm.encode_obs(obs)
        direct = core.encoder(obs["jepa_entity"].unsqueeze(1), obs["jepa_entity_mask"].unsqueeze(1))[:, 0]
        torch.testing.assert_close(encoded["z"], direct)
        z0, d0 = wm.initial(1, device=torch.device(args.device))
        action = torch.zeros(1, n_agents * n_actions, device=args.device)
        action[:, 0] = 1.0
        z1, d1 = wm.obs_step(z0, d0, action, encoded, torch.ones(1, dtype=torch.bool, device=args.device))
        z2, d2 = wm.img_step(z1, d1, action)
        assert torch.isfinite(z2).all() and torch.isfinite(d2).all()
    print("JEPA R2 wrapper encoder/memory one-step checks passed.")


if __name__ == "__main__":
    main()
