from __future__ import annotations

from pathlib import Path
from omegaconf import OmegaConf

src = Path("configs/r2_2100_jepa_local.yaml")
out = Path("configs/tmp_r2_2100_jepa_local_modelpt_eval10k.yaml")

cfg = OmegaConf.load(src)

cfg.world_model.jepa.checkpoint = "checkpoints/jepa/model.pt"
cfg.validation.every = 10000
cfg.validation.run_at_start = True
cfg.steps = 50000
cfg.logdir = "logs/r2dreamer/debug_exp33_jepa_beliefmask_slotadapter_grad_eval10k"

# Keep the smoke isolated and offline.
if "wandb" in cfg:
    cfg.wandb.mode = "disabled"

OmegaConf.save(cfg, out)
print(f"wrote {out}")
print("world_model.jepa.checkpoint =", cfg.world_model.jepa.checkpoint)
print("validation.every =", cfg.validation.every)
print("validation.run_at_start =", cfg.validation.run_at_start)
print("steps =", cfg.steps)
print("logdir =", cfg.logdir)
