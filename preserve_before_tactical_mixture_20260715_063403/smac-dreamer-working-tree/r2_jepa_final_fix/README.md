# Final R2-JEPA Integration Fix

Fixes:
1. Activates slot-preserving adapter by passing `max_agents=self.max_agents` from `world_model.py`.
2. Moves `feature_adapter(...)` outside `torch.no_grad()` in `get_feat()` so the adapter can actually train.
3. Keeps the belief-mask / hidden-seen entity exposure patch.
4. Provides a no-50k-smoke preflight and immediate 2M wandb launcher.

Use from repo root:

```bash
unzip r2_jepa_final_fix.zip
bash r2_jepa_final_fix/apply_final_fix.sh
python -m py_compile src/smacdreamer/jepa/world_model.py src/smacdreamer/jepa/feature_adapter.py preflight_final_r2_jepa.py make_final_2m_config.py
python make_final_2m_config.py
python preflight_final_r2_jepa.py --jepa-checkpoint checkpoints/jepa/model.pt
WANDB_PROJECT=smac-dreamer-jepa bash launch_final_2m_wandb.sh
```

Monitor:

```bash
bash monitor_final_run.sh
```
