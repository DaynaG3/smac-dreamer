import importlib.util
from pathlib import Path


INSTALLER = Path(__file__).parents[1] / "install_tactical_hardening.py"
spec = importlib.util.spec_from_file_location("hardener", INSTALLER)
hardener = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hardener)


def test_patch_dreamer_fixture():
    source = f'''import math
import torch
# TACTICAL_MIXTURE_V1
class Dreamer:
    def __init__(self):
        self._named_params = {{}}
        self.tactical_enabled = False
        self.tactical_policy = None
        self.actor = object()
        self.world_model_backend = "jepa"
        self.jepa_world_model = object()
        modules = {{"actor": self.actor, "jepa_feature_adapter": object()}}
        # count number of parameters in each module
        print(f"Optimizer has: {{sum(p.numel() for p in self._named_params.values())}} parameters.")

    def act(self, feat, eval=False):
        if self.action_masking:
            raw_logits = base_logits
            if self.tactical_enabled:
                tactic = self._frozen_tactical_policy.select_tactic(
                    feat, deterministic=eval
                )
                raw_logits = self._frozen_tactical_policy.combine_logits(
                    raw_logits, feat, tactic
                )
        return raw_logits

    def _cal_grad_jepa(self):
        primitive_policy_loss = 0
{hardener.OLD_OBJECTIVE}
        metrics["jepa/trainable_adapter_parameter_count"] = torch.tensor(
            float(sum(
                p.numel()
                for p in self.jepa_world_model.feature_adapter.parameters()
            )),
            device=feat.device,
        )
    def tactical_metadata(self):
        if not self.tactical_enabled:
            return {{
                "schema_version": 1,
                "architecture": "legacy",
                "enabled": False,
            }}
        metadata = self.tactical_policy.metadata()
        metadata["enabled"] = True
        return metadata

    def load_tactical_compatible_state_dict(
        self,
        state_dict,
        checkpoint_metadata=None,
    ):
        pass

    def _update_slow_target(self):
        pass
'''
    patched = hardener.patch_dreamer(source)
    compile(patched, "dreamer.py", "exec")
    assert hardener.HARDENING_MARKER in patched
    assert "tactic/mutual_information" in patched
    assert "base_policy_logits[:, :-1].detach()" in patched
    assert "optimizer parameter registry contains duplicates" in patched
    assert "metadata-less tactical checkpoint" in patched
    assert "inherited base actor frozen" in patched
    assert "eval_combined_logits" in patched
    assert 'metrics["jepa/adapter_total_parameter_count"]' in patched
    assert "if p.requires_grad" in patched
    assert "metadata_is_legacy" in patched
    assert "max_abs_residual_logit" in patched


def test_patch_runner_guards_buffer_and_adaptive_state():
    source = """# UNIFIED_PRIORITY_V1
# TACTICAL_MIXTURE_V1
def main():
    _adaptive_any = False
    replay_buffer = object()
    resume_step = 0
    replay_buffer.set_env_step(resume_step)
    ckpt = {}
    priority_controller = object()
    if ckpt.get('adaptive_priority_state') is not None:
        priority_controller.load_state_dict(
            ckpt['adaptive_priority_state'], strict=True
        )
        print(' [resume] restored adaptive map-priority state')
    else:
        print(' [resume] old checkpoint has no adaptive state; maps start uniform')
    agent = object()
    def _extra_checkpoint_state():
        return {
            'adaptive_priority_schema': 1,
            'adaptive_priority_state': priority_controller.state_dict(),
            'tactical_mixture_metadata': agent.tactical_metadata(),
        }
    checkpointer = None
"""
    patched = hardener.patch_runner(source)
    compile(patched, "runner.py", "exec")
    assert 'if hasattr(replay_buffer, "set_env_step"):' in patched
    assert "if _adaptive_any and ckpt.get('adaptive_priority_state')" in patched
    assert "if _adaptive_any:" in patched


def test_patch_validation_best_checkpoint_metadata():
    source = '''import math\nimport torch\nclass ValidationTrainer:\n    def eval(self, agent, train_step):\n        wr, ret = 0.1, 1.0\n        self._val_obs_mode = "entity"\n        self._logdir = None\n        torch.save(\n            {"agent_state_dict": agent.state_dict(),\n             "val_macro_win_rate": wr, "val_macro_original_return": ret,\n             "step": int(train_step), "obs_mode": self._val_obs_mode},\n            self._logdir / "best_val_macro_winrate.pt",\n        )\n'''
    patched = hardener.patch_validation_trainer(source)
    compile(patched, "validation_trainer.py", "exec")
    assert "tactical_mixture_metadata" in patched
    assert "best_payload" in patched



def test_build_config_is_prevalidated_and_forces_fresh_replay(tmp_path):
    repo = tmp_path / "smac-dreamer"
    source = repo / "configs" / "r2_2100_jepa_tactical_mixture.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """world_model:
  backend: jepa
reward:
  name: dense_v3
imag_horizon: 5
buffer:
  scratch_dir: /tmp/stale-replay
sampling_mode: adaptive_priority
adaptive_priority:
  enabled: true
  map: {enabled: true}
  sequence: {enabled: true}
validation: {run_at_start: true, every: 1}
tactical_mixture:
  enabled: true
  duration: 1
""",
        encoding="utf-8",
    )
    target, cfg = hardener.build_config(
        repo, "configs/r2_2100_jepa_tactical_mixture.yaml"
    )
    assert target == repo / "configs/r2_2100_jepa_tactical_mixture_hardened.yaml"
    assert not target.exists()
    assert cfg.buffer.scratch_dir == "replay"
    assert cfg.sampling_mode == "shuffled_round_robin"
    assert cfg.adaptive_priority.enabled is False
    assert cfg.adaptive_priority.map.enabled is False
    assert cfg.adaptive_priority.sequence.enabled is False
    assert cfg.validation.run_at_start is False
    assert cfg.validation.every == 200000


def test_build_config_rejects_wrong_backend_before_writes(tmp_path):
    repo = tmp_path / "smac-dreamer"
    source = repo / "configs" / "bad.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """world_model: {backend: rssm}
reward: {name: dense_v3}
imag_horizon: 5
tactical_mixture: {enabled: true, duration: 1}
""",
        encoding="utf-8",
    )
    try:
        hardener.build_config(repo, "configs/bad.yaml")
    except SystemExit as exc:
        assert "world_model.backend=jepa" in str(exc)
    else:
        raise AssertionError("wrong-backend source config was accepted")
