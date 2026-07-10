from __future__ import annotations

from pathlib import Path
from omegaconf import OmegaConf

base = Path("configs/r2_2100_jepa_local.yaml")
out = Path("configs/tmp_r2_2100_jepa_final_2m_wandb.yaml")

cfg = OmegaConf.load(base)
cfg.world_model.jepa.checkpoint = "checkpoints/jepa/model.pt"
cfg.validation.every = 50000
cfg.validation.run_at_start = False
cfg.steps = 2000000
cfg.wandb.mode = "online"
cfg.wandb.project = "smac-dreamer-jepa"
cfg.wandb.tags = ["r2_2100", "smaclite", "frozen_jepa", "exp33", "belief_mask", "slot_adapter", "adapter_grad", "2m"]
cfg.wandb.run_name = None
cfg.logdir = "logs/r2dreamer/exp33_jepa_final_beliefmask_slotadapter_grad_2m"
OmegaConf.save(cfg, out)
print(f"wrote {out}")
print("checkpoint:", cfg.world_model.jepa.checkpoint)
print("validation.every:", cfg.validation.every)
print("wandb.mode:", cfg.wandb.mode)
print("steps:", cfg.steps)
