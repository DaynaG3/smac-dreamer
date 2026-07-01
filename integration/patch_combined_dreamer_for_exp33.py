from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".pre_exp33")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def patch_checkpoint(path: Path) -> bool:
    text = path.read_text()
    if "AnchoredActionConditionedEntityRolloutGRUMemory" in text:
        return False
    backup(path)

    start = text.index("def _import_jepa():")
    end = text.index("def _required", start)
    import_block = '''def _import_jepa():
    try:
        from smac_jepa.jepa import SMACJEPA
        try:
            from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory
        except ImportError:
            EntityRolloutGRUMemory = CompatEntityRolloutGRUMemory
        from smac_jepa.anchored_belief_memory import (
            AnchoredActionConditionedEntityRolloutGRUMemory,
        )
        return (
            SMACJEPA,
            EntityRolloutGRUMemory,
            AnchoredActionConditionedEntityRolloutGRUMemory,
        )
    except ImportError as exc:
        raise JEPADependencyError(
            "world_model.backend='jepa' requires the external smac-jepa-wm package. "
            "Install it with: python -m pip install -e \\\"<PATH_TO_SMAC_JEPA_REPO>\\\""
        ) from exc


'''
    text = text[:start] + import_block + text[end:]
    text = text.replace(
        "SMACJEPA, EntityRolloutGRUMemory = _import_jepa()",
        "(SMACJEPA, EntityRolloutGRUMemory, "
        "AnchoredActionConditionedEntityRolloutGRUMemory) = _import_jepa()",
        1,
    )

    memory_start_marker = '    action_conditioned = bool(cfg.get("action_conditioned_memory", False))'
    memory_end_marker = "    missing, unexpected = memory.load_state_dict(memory_state, strict=False)"
    memory_start = text.index(memory_start_marker)
    memory_end = text.index(memory_end_marker, memory_start)
    memory_block = '''    action_conditioned = bool(cfg.get("action_conditioned_memory", False))
    memory_dim = int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", 128)))
    hidden = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in memory_state
    )
    if anchored:
        if not action_conditioned:
            raise JEPACompatibilityError(
                "Anchored Exp33 checkpoint must set action_conditioned_memory=True"
            )
        memory = AnchoredActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            n_actions=int(metadata["n_actions"]),
            max_agents=int(metadata["max_agents"]),
            hidden_dim=hidden,
            residual=residual,
        )
    elif action_conditioned:
        memory = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            n_actions=int(metadata["n_actions"]),
            hidden_dim=hidden,
            residual=residual,
        )
    else:
        memory = EntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            hidden_dim=hidden,
            residual=residual,
        )
'''
    text = text[:memory_start] + memory_block + text[memory_end:]
    path.write_text(text)
    return True


def patch_world_model(path: Path) -> bool:
    text = path.read_text()
    changed = False
    if "self.presence_rollout_mode" not in text:
        backup(path)
        marker = "        self.presence_threshold = float(presence_threshold)\n"
        replacement = marker + '''        anchored = bool(info.resolved_config.get("anchored_belief_memory", False))
        default_presence_mode = "soft" if anchored else "hard"
        self.presence_rollout_mode = str(
            info.resolved_config.get("presence_rollout_mode", default_presence_mode)
        ).strip().lower()
        if self.presence_rollout_mode not in {"soft", "hard"}:
            raise ValueError(
                "Unsupported JEPA presence_rollout_mode: "
                f"{self.presence_rollout_mode!r}"
            )
'''
        if marker not in text:
            raise RuntimeError("Could not locate presence_threshold assignment")
        text = text.replace(marker, replacement, 1)
        changed = True

    lines = text.splitlines(keepends=True)
    logits_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "logits = self.core.predict_presence(pred)" in line
        ),
        None,
    )
    already_soft = "presence_probability = torch.sigmoid(logits)" in text
    if logits_index is not None and not already_soft:
        next_mask_index = next(
            (
                index
                for index in range(logits_index + 1, min(logits_index + 6, len(lines)))
                if "next_mask =" in lines[index]
            ),
            None,
        )
        pred_index = next(
            (
                index
                for index in range(logits_index + 1, min(logits_index + 8, len(lines)))
                if "pred = pred * next_mask.unsqueeze(-1)" in lines[index]
            ),
            None,
        )
        if next_mask_index is None or pred_index is None:
            raise RuntimeError("Could not locate the JEPA presence propagation block")
        indent = lines[logits_index][: len(lines[logits_index]) - len(lines[logits_index].lstrip())]
        block = [
            f"{indent}logits = self.core.predict_presence(pred)\n",
            f"{indent}presence_probability = torch.sigmoid(logits).to(dtype=z.dtype)\n",
            f"{indent}if self.presence_rollout_mode == \"soft\":\n",
            f"{indent}    next_mask = presence_probability * slot_mask\n",
            f"{indent}else:\n",
            f"{indent}    next_mask = (\n",
            f"{indent}        presence_probability >= self.presence_threshold\n",
            f"{indent}    ).to(dtype=z.dtype) * slot_mask\n",
            f"{indent}pred = pred * next_mask.unsqueeze(-1)\n",
        ]
        if not changed:
            backup(path)
        lines[logits_index : pred_index + 1] = block
        text = "".join(lines)
        changed = True
    elif not already_soft:
        raise RuntimeError("Could not locate the JEPA presence propagation block")

    if changed:
        path.write_text(text)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dreamer_root", type=Path)
    args = parser.parse_args()
    root = args.dreamer_root.resolve()
    jepa_dir = root / "src" / "smacdreamer" / "jepa"
    checkpoint = jepa_dir / "checkpoint.py"
    world_model = jepa_dir / "world_model.py"
    for path in (checkpoint, world_model):
        if not path.exists():
            raise SystemExit(f"Missing expected Dreamer file: {path}")

    c1 = patch_checkpoint(checkpoint)
    c2 = patch_world_model(world_model)
    print(f"checkpoint.py: {'patched' if c1 else 'already patched'}")
    print(f"world_model.py: {'patched' if c2 else 'already patched'}")
    print("Backups use the suffix .pre_exp33")


if __name__ == "__main__":
    main()
