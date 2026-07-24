from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "payload" / "scripts"

def test_stage_order_and_independent_sources():
    text = (P / "run_forecast_then_actor_critic_h15_then_option_critic_v9.sh").read_text()
    assert text.index("run_stage forecast_jepa") < text.index("run_stage actor_critic_h15_800k") < text.index("run_stage option_critic_v9_h15_800k")
    assert "run_actor_critic_h15_800k.sh" in text
    assert "run_option_critic_v9_anchor_safe_800k.sh" in text
    assert "not from the 800k baseline result" in text

def test_baseline_exact_horizon_and_steps():
    text = (P / "run_actor_critic_h15_800k.sh").read_text()
    assert "FINAL_STEP:-800000" in text
    assert "FINAL_STEP == 800000" in text
    assert "--resume-start-step 0" in text
    assert "--steps \"$FINAL_STEP\"" in text

def test_gpu_safety_guards():
    text = (P / "run_forecast_then_actor_critic_h15_then_option_critic_v9.sh").read_text()
    assert "active_forecast" in text and "active_rl" in text
    assert "SKIPPED_ACTIVE_TRAIN_PROCESS" in text
