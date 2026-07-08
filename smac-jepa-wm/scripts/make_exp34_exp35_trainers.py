#!/usr/bin/env python3
"""
Create Exp34/Exp35 Dreamer-compatible JEPA trainers from the DaynaG3 combined repo.

Run from:
  ~/workspace/dreamer/combined-upload/smac-jepa-wm

What it does:
  1. Copies smac_jepa/train_jepa_exp31_exp33.py -> smac_jepa/train_jepa_exp31_exp35.py
  2. Patches the copied trainer only, leaving original Exp33 intact.
  3. Copies train_jepa_exp33_dreamer.py -> train_jepa_exp34_dreamer.py and train_jepa_exp35_dreamer.py
     and rewires their base import to the patched trainer.

Exp34:
  Full Exp33 complex loss family, but with optional two-mask target/loss semantics.
  Use --exp34-two-mask-loss and --presence-neg-class-weight.

Exp35:
  Same code path, but enables --exp35-simple-loss to zero broad aux losses while keeping:
    L_dyn + L_step + L_dec + L_pres + L_SIGReg.

This patch is intentionally conservative: it does not modify the original Exp33 files.
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


def regex_replace_once(text: str, pattern: str, repl: str, label: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        die(f"Expected exactly one regex match for {label}, found {count}.")
    return new_text


def main() -> None:
    if not SRC_BASE.exists():
        die(f"Missing {SRC_BASE}; run from smac-jepa-wm repo root.")
    if not SRC_DREAMER.exists():
        die(f"Missing {SRC_DREAMER}; run from smac-jepa-wm repo root.")

    base = SRC_BASE.read_text()

    if "exp34_two_mask_loss" in base or "presence_neg_class_weight" in base:
        print("[info] source base already seems patched; refusing to patch original. Use generated file if present.")

    patched = base

    # 1) Add CLI args after presence-weight.
    patched = replace_once(
        patched,
        'parser.add_argument("--presence-weight", type=float, default=1.0)',
        'parser.add_argument("--presence-weight", type=float, default=1.0)\n'
        '    parser.add_argument("--presence-neg-class-weight", type=float, default=1.0, '\
        'help="Exp34/35: multiply BCE for target_absent/dead slots. Use 2-3 to reduce false-alive hallucination.")\n'
        '    parser.add_argument("--exp34-two-mask-loss", action="store_true", '\
        'help="Exp34: use structural slot masks for latent/decode/delta/hidden supervision instead of target-presence masks.")\n'
        '    parser.add_argument("--exp35-simple-loss", action="store_true", '\
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
    predictions on target_absent/dead slots more strongly. This directly targets the
    qualitative probe failure where dead/absent enemies remained decoded as alive.
    """
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = weights.view(1, 1, -1, 1)
    neg_w = torch.as_tensor(float(neg_class_weight), device=target.device, dtype=target.dtype)
    class_w = torch.where(target > 0.5, torch.ones_like(target), neg_w)
    weighted_mask = mask.to(dtype=target.dtype) * w * class_w
    return (raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
'''
    patched = replace_once(
        patched,
        "def markov_rollout_rnn_losses(",
        helper + "\ndef markov_rollout_rnn_losses(",
        "insert weighted BCE helper",
    )

    # 3) Add function signature args near inverse_dynamics_weight.
    patched = replace_once(
        patched,
        "inverse_dynamics_weight: float,\n) -> dict[str, torch.Tensor]:",
        "inverse_dynamics_weight: float,\n    presence_neg_class_weight: float = 1.0,\n    exp34_two_mask_loss: bool = False,\n    exp35_simple_loss: bool = False,\n) -> dict[str, torch.Tensor]:",
        "extend loss function signature",
    )

    # 4) Add simple-loss ablation at function start after del detach_rollout_targets.
    patched = replace_once(
        patched,
        "del detach_rollout_targets",
        "del detach_rollout_targets\n    if exp35_simple_loss:\n"
        "        # Exp35 ablation: after Exp34 mask cleanup, keep only the essential world-model terms.\n"
        "        r2_rep_scale = 0.0\n"
        "        r2_barlow_scale = 0.0\n"
        "        memory_barlow_scale = 0.0\n"
        "        delta_loss_weight = 0.0\n"
        "        hidden_reconstruction_weight = 0.0\n"
        "        last_seen_anchor_weight = 0.0\n"
        "        hidden_presence_weight = 0.0\n"
        "        reappearance_consistency_weight = 0.0\n"
        "        inverse_dynamics_weight = 0.0",
        "insert exp35 simple ablation",
    )

    # 5) Define target_encoding_mask_seq after slot_mask_seq is available.
    patched = replace_once(
        patched,
        'slot_mask_seq = batch["entity_slot_mask_seq"] feature_valid = batch["feature_valid_mask"]',
        'slot_mask_seq = batch["entity_slot_mask_seq"] target_encoding_mask_seq = (slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full) feature_valid = batch["feature_valid_mask"]',
        "define target_encoding_mask_seq",
    )

    # 6) Encode/norm target with target_encoding_mask_seq.
    patched = patched.replace(
        "target_entity_seq_full, target_entity_mask_seq_full,",
        "target_entity_seq_full, target_encoding_mask_seq,",
        2,  # online target encoder and EMA target encoder calls
    )
    patched = patched.replace(
        "online_target_latents_raw, target_entity_mask_seq_full, enabled=r2_latent_normalize,",
        "online_target_latents_raw, target_encoding_mask_seq, enabled=r2_latent_normalize,",
        1,
    )
    patched = patched.replace(
        "consistency_target_latents_raw, target_entity_mask_seq_full, enabled=r2_latent_normalize,",
        "consistency_target_latents_raw, target_encoding_mask_seq, enabled=r2_latent_normalize,",
        1,
    )

    # 7) Replace latent/decoded mask block with two-mask switch.
    patched = replace_once(
        patched,
        "latent_mask = ( target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1) )",
        "loss_entity_mask = (entity_slot_mask if exp34_two_mask_loss else target_entity_mask)\n"
        "    latent_mask = ( loss_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1) )",
        "latent mask target->slot switch",
    )
    patched = replace_once(
        patched,
        "dynamic_feature_valid[:, None, None] .expand_as(target_entity) * target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)",
        "dynamic_feature_valid[:, None, None] .expand_as(target_entity) * loss_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)",
        "dynamic_reference_mask target->loss_entity_mask",
    )
    patched = replace_once(
        patched,
        "target_valid = ( target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1) )",
        "target_valid = ( loss_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1) )",
        "decoded target_valid target->loss_entity_mask",
    )

    # 8) Presence BCE with optional negative class weight.
    patched = replace_once(
        patched,
        "presence_loss = weighted_bce( presence_logits, target_entity_mask, presence_mask, aux_weights, )",
        "presence_loss = weighted_bce_with_negative_class_weight( presence_logits, target_entity_mask, presence_mask, aux_weights, neg_class_weight=presence_neg_class_weight, )",
        "presence_loss weighted BCE",
    )
    patched = replace_once(
        patched,
        "hidden_presence_loss = weighted_bce( presence_logits, target_entity_mask, hidden_presence_mask, aux_weights, )",
        "hidden_presence_loss = weighted_bce_with_negative_class_weight( presence_logits, target_entity_mask, hidden_presence_mask, aux_weights, neg_class_weight=presence_neg_class_weight, )",
        "hidden_presence_loss weighted BCE",
    )

    # 9) SIGReg target mask should match target encoding mask in Exp34.
    patched = replace_once(
        patched,
        "reg_masks = torch.cat( [observation_mask_seq, target_entity_mask_seq_full], dim=1 )",
        "reg_masks = torch.cat( [observation_mask_seq, target_encoding_mask_seq], dim=1 )",
        "SIGReg target mask switch",
    )

    # 10) Diagnostics use loss_entity_mask.
    patched = replace_once(
        patched,
        "latent_valid = ( target_entity_mask * entity_slot_mask * valid_mask.unsqueeze(-1) )",
        "latent_valid = ( loss_entity_mask * entity_slot_mask * valid_mask.unsqueeze(-1) )",
        "latent_valid diagnostic switch",
    )

    # 11) Patch call sites to pass args.
    patched = replace_once(
        patched,
        "inverse_dynamics_weight=args.inverse_dynamics_weight,",
        "inverse_dynamics_weight=args.inverse_dynamics_weight,\n"
        "                    presence_neg_class_weight=args.presence_neg_class_weight,\n"
        "                    exp34_two_mask_loss=args.exp34_two_mask_loss,\n"
        "                    exp35_simple_loss=args.exp35_simple_loss,",
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

    print("\n[ok] Created Exp34/Exp35 trainer files. Original Exp33 files are untouched.")
    print("[next] Run: python -m py_compile smac_jepa/train_jepa_exp31_exp35.py smac_jepa/train_jepa_exp34_dreamer.py smac_jepa/train_jepa_exp35_dreamer.py")


if __name__ == "__main__":
    main()
