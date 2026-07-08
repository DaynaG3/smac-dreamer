#!/usr/bin/env python3
"""
Robust generator for Exp34/Exp35 Dreamer-compatible JEPA trainers.
Run from smac-jepa-wm repo root:
  python scripts/make_exp34_exp35_trainers_v2.py

Creates, without modifying Exp33:
  smac_jepa/train_jepa_exp31_exp35.py
  smac_jepa/train_jepa_exp34_dreamer.py
  smac_jepa/train_jepa_exp35_dreamer.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
PKG = ROOT / "smac_jepa"
SRC_BASE = PKG / "train_jepa_exp31_exp33.py"
SRC_DREAMER = PKG / "train_jepa_exp33_dreamer.py"
OUT_BASE = PKG / "train_jepa_exp31_exp35.py"
OUT_EXP34 = PKG / "train_jepa_exp34_dreamer.py"
OUT_EXP35 = PKG / "train_jepa_exp35_dreamer.py"


def die(msg: str) -> None:
    raise SystemExit(f"[patch-error] {msg}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        die(f"Expected exactly one match for {label}, found {count}.")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, repl: str, label: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        die(f"Expected exactly one regex match for {label}, found {count}.")
    return new_text


def regex_all_exact_count(text: str, pattern: str, repl: str, label: str, expected: int, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, repl, text, count=expected, flags=flags)
    if count != expected:
        die(f"Expected {expected} regex matches for {label}, found {count}.")
    return new_text


def main() -> None:
    if not SRC_BASE.exists():
        die(f"Missing {SRC_BASE}; run from smac-jepa-wm repo root.")
    if not SRC_DREAMER.exists():
        die(f"Missing {SRC_DREAMER}; run from smac-jepa-wm repo root.")

    base = SRC_BASE.read_text()
    if "exp34_two_mask_loss" in base or "presence_neg_class_weight" in base:
        die("Original train_jepa_exp31_exp33.py already contains Exp34 markers. Refusing to patch original.")

    patched = base

    # 1) Add CLI args after presence-weight.
    patched = regex_once(
        patched,
        r'parser\.add_argument\("--presence-weight",\s*type=float,\s*default=1\.0\)',
        'parser.add_argument("--presence-weight", type=float, default=1.0)\n'
        '    parser.add_argument("--presence-neg-class-weight", type=float, default=1.0, '
        'help="Exp34/35: multiply BCE for target_absent/dead slots. Use 2-3 to reduce false-alive hallucination.")\n'
        '    parser.add_argument("--exp34-two-mask-loss", action="store_true", '
        'help="Exp34: use structural slot masks for latent/decode/delta/hidden supervision instead of target-presence masks.")\n'
        '    parser.add_argument("--exp35-simple-loss", action="store_true", '
        'help="Exp35: keep dyn + one-step + decoder + presence + SIGReg, zero broader auxiliary losses inside the trainer.")',
        "add exp34/35 parser args",
    )

    # 2) Add helper before markov_rollout_rnn_losses.
    helper = r'''

def weighted_bce_with_negative_class_weight(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    neg_class_weight: float = 1.0,
) -> torch.Tensor:
    """
    logits/target/mask: [B, P, H, E]
    weights: [H]

    Same temporal/mask semantics as weighted_bce(), but optionally punishes false-alive
    predictions on target_absent/dead slots more strongly.
    """
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = weights.view(1, 1, -1, 1)
    neg_w = torch.as_tensor(float(neg_class_weight), device=target.device, dtype=target.dtype)
    class_w = torch.where(target > 0.5, torch.ones_like(target), neg_w)
    weighted_mask = mask.to(dtype=target.dtype) * w * class_w
    return (raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
'''
    patched = regex_once(
        patched,
        r'\ndef\s+markov_rollout_rnn_losses\(',
        helper + '\n\ndef markov_rollout_rnn_losses(',
        "insert weighted BCE helper",
    )

    # 3) Extend markov_rollout_rnn_losses signature.
    patched = regex_once(
        patched,
        r'inverse_dynamics_weight:\s*float,\s*\)\s*->\s*dict\[str,\s*torch\.Tensor\]:',
        'inverse_dynamics_weight: float,\n'
        '    presence_neg_class_weight: float = 1.0,\n'
        '    exp34_two_mask_loss: bool = False,\n'
        '    exp35_simple_loss: bool = False,\n'
        ') -> dict[str, torch.Tensor]:',
        "extend loss function signature",
    )

    # 4) Exp35 ablation switch near function start.
    patched = regex_once(
        patched,
        r'del\s+detach_rollout_targets',
        'del detach_rollout_targets\n'
        '    if exp35_simple_loss:\n'
        '        # Exp35 ablation: after Exp34 mask cleanup, keep only essential world-model terms.\n'
        '        r2_rep_scale = 0.0\n'
        '        r2_barlow_scale = 0.0\n'
        '        memory_barlow_scale = 0.0\n'
        '        delta_loss_weight = 0.0\n'
        '        hidden_reconstruction_weight = 0.0\n'
        '        last_seen_anchor_weight = 0.0\n'
        '        hidden_presence_weight = 0.0\n'
        '        reappearance_consistency_weight = 0.0\n'
        '        inverse_dynamics_weight = 0.0',
        "insert exp35 simple ablation",
    )

    # 5) Define target_encoding_mask_seq after slot_mask_seq.
    patched = regex_once(
        patched,
        r'(slot_mask_seq\s*=\s*batch\["entity_slot_mask_seq"\])\s*(feature_valid\s*=\s*batch\["feature_valid_mask"\])',
        r'\1\n    target_encoding_mask_seq = (slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full)\n    \2',
        "define target_encoding_mask_seq",
    )

    # 6) Encode target with target_encoding_mask_seq.
    patched = regex_once(
        patched,
        r'online_target_latents_raw\s*=\s*model\.encoder\(\s*target_entity_seq_full,\s*target_entity_mask_seq_full,\s*\)',
        'online_target_latents_raw = model.encoder(\n        target_entity_seq_full, target_encoding_mask_seq,\n    )',
        "online target encoder mask",
    )
    patched = regex_once(
        patched,
        r'consistency_target_latents_raw\s*=\s*target_encoder\(\s*target_entity_seq_full,\s*target_entity_mask_seq_full,\s*\)',
        'consistency_target_latents_raw = target_encoder(\n                target_entity_seq_full, target_encoding_mask_seq,\n            )',
        "EMA target encoder mask",
    )
    patched = regex_once(
        patched,
        r'online_target_latents\s*=\s*r2_normalize_latent\(\s*online_target_latents_raw,\s*target_entity_mask_seq_full,\s*enabled=r2_latent_normalize,\s*\)',
        'online_target_latents = r2_normalize_latent(\n        online_target_latents_raw, target_encoding_mask_seq, enabled=r2_latent_normalize,\n    )',
        "online target norm mask",
    )
    patched = regex_once(
        patched,
        r'target_latents\s*=\s*r2_normalize_latent\(\s*consistency_target_latents_raw,\s*target_entity_mask_seq_full,\s*enabled=r2_latent_normalize,\s*\)',
        'target_latents = r2_normalize_latent(\n        consistency_target_latents_raw, target_encoding_mask_seq, enabled=r2_latent_normalize,\n    )',
        "target norm mask",
    )

    # 7) Replace masks used by latent/decode/delta/hidden losses.
    patched = regex_once(
        patched,
        r'latent_mask\s*=\s*\(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
        'loss_entity_mask = (entity_slot_mask if exp34_two_mask_loss else target_entity_mask)\n'
        '    latent_mask = (\n'
        '        loss_entity_mask.unsqueeze(-1)\n'
        '        * valid_mask.unsqueeze(-1).unsqueeze(-1)\n'
        '    )',
        "latent mask target->slot switch",
    )
    patched = regex_once(
        patched,
        r'dynamic_feature_valid\[:,\s*None,\s*None\]\s*\.expand_as\(target_entity\)\s*\*\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)',
        'dynamic_feature_valid[:, None, None]\n        .expand_as(target_entity)\n        * loss_entity_mask.unsqueeze(-1)\n        * valid_mask.unsqueeze(-1).unsqueeze(-1)',
        "dynamic_reference_mask target->loss_entity_mask",
    )
    patched = regex_once(
        patched,
        r'target_valid\s*=\s*\(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
        'target_valid = (\n'
        '        loss_entity_mask.unsqueeze(-1)\n'
        '        * valid_mask.unsqueeze(-1).unsqueeze(-1)\n'
        '    )',
        "decoded target_valid target->loss_entity_mask",
    )

    # 8) Presence BCE with optional negative class weight.
    patched = regex_once(
        patched,
        r'presence_loss\s*=\s*weighted_bce\(\s*presence_logits,\s*target_entity_mask,\s*presence_mask,\s*aux_weights,\s*\)',
        'presence_loss = weighted_bce_with_negative_class_weight(\n'
        '        presence_logits,\n'
        '        target_entity_mask,\n'
        '        presence_mask,\n'
        '        aux_weights,\n'
        '        neg_class_weight=presence_neg_class_weight,\n'
        '    )',
        "presence_loss weighted BCE",
    )
    patched = regex_once(
        patched,
        r'hidden_presence_loss\s*=\s*weighted_bce\(\s*presence_logits,\s*target_entity_mask,\s*hidden_presence_mask,\s*aux_weights,\s*\)',
        'hidden_presence_loss = weighted_bce_with_negative_class_weight(\n'
        '        presence_logits,\n'
        '        target_entity_mask,\n'
        '        hidden_presence_mask,\n'
        '        aux_weights,\n'
        '        neg_class_weight=presence_neg_class_weight,\n'
        '    )',
        "hidden_presence_loss weighted BCE",
    )

    # 9) SIGReg target mask follows target encoding mask under Exp34.
    patched = regex_once(
        patched,
        r'reg_masks\s*=\s*torch\.cat\(\s*\[observation_mask_seq,\s*target_entity_mask_seq_full\],\s*dim=1\s*\)',
        'reg_masks = torch.cat(\n'
        '            [observation_mask_seq, target_encoding_mask_seq], dim=1\n'
        '        )',
        "SIGReg target mask switch",
    )

    # 10) Diagnostics use loss_entity_mask.
    patched = regex_once(
        patched,
        r'latent_valid\s*=\s*\(\s*target_entity_mask\s*\*\s*entity_slot_mask\s*\*\s*valid_mask\.unsqueeze\(-1\)\s*\)',
        'latent_valid = (\n'
        '        loss_entity_mask * entity_slot_mask * valid_mask.unsqueeze(-1)\n'
        '    )',
        "latent_valid diagnostic switch",
    )

    # 11) Pass args into loss call.
    patched = regex_once(
        patched,
        r'inverse_dynamics_weight=args\.inverse_dynamics_weight,',
        'inverse_dynamics_weight=args.inverse_dynamics_weight,\n'
        '                    presence_neg_class_weight=args.presence_neg_class_weight,\n'
        '                    exp34_two_mask_loss=args.exp34_two_mask_loss,\n'
        '                    exp35_simple_loss=args.exp35_simple_loss,',
        "pass exp34/35 args to loss",
    )

    OUT_BASE.write_text(patched)
    print(f"[done] wrote {OUT_BASE}")

    # 12) Create Dreamer wrappers that point to patched base.
    dreamer = SRC_DREAMER.read_text()
    if "train_jepa_exp31_exp33" not in dreamer:
        die("Could not find train_jepa_exp31_exp33 import in dreamer wrapper.")
    for out, exp_name in [(OUT_EXP34, "Exp34 two-mask Dreamer"), (OUT_EXP35, "Exp35 simple-loss Dreamer")]:
        txt = dreamer.replace("train_jepa_exp31_exp33", "train_jepa_exp31_exp35")
        txt = txt.replace("Exp33", exp_name).replace("exp33", out.stem.replace("train_jepa_", ""))
        out.write_text(txt)
        print(f"[done] wrote {out}")

    print("[done] generated Exp34/Exp35 trainers. Now run py_compile.")


if __name__ == "__main__":
    main()
