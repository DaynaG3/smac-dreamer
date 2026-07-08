#!/usr/bin/env python3
from pathlib import Path
import re
import sys

BASE = Path("smac_jepa/train_jepa_exp31_exp33.py")
WRAP33 = Path("smac_jepa/train_jepa_exp33_dreamer.py")

OUT = Path("smac_jepa/train_jepa_exp31_exp35.py")
WRAP34 = Path("smac_jepa/train_jepa_exp34_dreamer.py")
WRAP35 = Path("smac_jepa/train_jepa_exp35_dreamer.py")

def die(msg: str) -> None:
    print(f"[patch-error] {msg}", file=sys.stderr)
    sys.exit(1)

if not BASE.exists():
    die(f"missing {BASE}")
if not WRAP33.exists():
    die(f"missing {WRAP33}")

src = BASE.read_text()
out = src

def sub_once(pattern: str, repl: str, name: str) -> None:
    global out
    new, n = re.subn(pattern, repl, out, count=1, flags=re.S)
    if n != 1:
        print(f"\n[patch-error] failed at: {name}", file=sys.stderr)
        print("Useful context search:", file=sys.stderr)
        for needle in [
            "del detach_rollout_targets",
            "target_entity_mask_seq_full",
            "online_target_latents_raw",
            "latent_mask =",
            "target_valid =",
            "presence_loss =",
            "reg_masks =",
            "latent_valid =",
        ]:
            i = out.find(needle)
            print(f"\n--- {needle} index={i} ---", file=sys.stderr)
            if i >= 0:
                print(out[max(0, i-300):i+900], file=sys.stderr)
        sys.exit(1)
    out = new

# 1) Inject env flags and define slot_mask_seq early enough for target encoder mask.
sub_once(
    r'(\s*)del detach_rollout_targets\s*\n\s*entity_seq = batch\["entity_seq"\]',
    r'''\1del detach_rollout_targets
\1_exp_os = __import__("os")
\1exp34_two_mask_loss = _exp_os.environ.get("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "0") == "1"
\1exp35_simple_loss = _exp_os.environ.get("SMAC_JEPA_EXP35_SIMPLE_LOSS", "0") == "1"
\1presence_neg_class_weight = float(_exp_os.environ.get("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "1.0"))

\1# Exp35 simpler-loss ablation: keep core dynamics/step/decoder/presence/SIGReg,
\1# remove broader representation/belief auxiliary pressure.
\1if exp35_simple_loss:
\1    r2_rep_scale = 0.0
\1    r2_barlow_scale = 0.0
\1    memory_barlow_scale = 0.0
\1    delta_loss_weight = 0.0
\1    hidden_reconstruction_weight = 0.0
\1    last_seen_anchor_weight = 0.0
\1    hidden_presence_weight = 0.0
\1    reappearance_consistency_weight = 0.0
\1    inverse_dynamics_weight = 0.0
\1entity_seq = batch["entity_seq"]
\1slot_mask_seq = batch["entity_slot_mask_seq"]''',
    "inject env flags after del detach_rollout_targets",
)

# 2) Define target_encoding_mask_seq after target_entity_mask_seq_full is created.
sub_once(
    r'(\s*)observation_mask_seq = original_observation_mask_seq',
    r'''\1# Exp34 two-mask cleanup:
\1# input encoder uses observation_mask_seq;
\1# target encoder can use structural slots instead of target presence.
\1target_encoding_mask_seq = (
\1    slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full
\1)
\1observation_mask_seq = original_observation_mask_seq''',
    "define target_encoding_mask_seq",
)

# 3) Replace target encoder masks.
for name, pattern in [
    (
        "online target encoder mask",
        r'(online_target_latents_raw = model\.encoder\(\s*target_entity_seq_full,\s*)target_entity_mask_seq_full(\s*,\s*\))',
    ),
    (
        "ema target encoder mask",
        r'(consistency_target_latents_raw = target_encoder\(\s*target_entity_seq_full,\s*)target_entity_mask_seq_full(\s*,\s*\))',
    ),
    (
        "online target normalize mask",
        r'(online_target_latents = r2_normalize_latent\(\s*online_target_latents_raw,\s*)target_entity_mask_seq_full(\s*,)',
    ),
    (
        "target latent normalize mask",
        r'(target_latents = r2_normalize_latent\(\s*consistency_target_latents_raw,\s*)target_entity_mask_seq_full(\s*,)',
    ),
]:
    new, n = re.subn(pattern, r'\1target_encoding_mask_seq\2', out, count=1, flags=re.S)
    if n != 1:
        die(f"failed to patch {name}")
    out = new

# 4) Replace latent_mask with env-controlled presence-vs-slot mask.
sub_once(
    r'(\s*)latent_mask = \(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1target_presence_valid = (
\1    target_entity_mask.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)
\1target_slot_valid = (
\1    entity_slot_mask.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)
\1latent_mask = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "latent_mask",
)

# 5) Replace dynamic_reference_mask target presence with structural slot under Exp34.
sub_once(
    r'(\s*)dynamic_reference_mask = \(\s*dynamic_feature_valid\[:, None, None\]\s*\.expand_as\(target_entity\)\s*\*\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1dynamic_reference_entity_valid = (
\1    entity_slot_mask if exp34_two_mask_loss else target_entity_mask
\1)
\1dynamic_reference_mask = (
\1    dynamic_feature_valid[:, None, None]
\1    .expand_as(target_entity)
\1    * dynamic_reference_entity_valid.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)''',
    "dynamic_reference_mask",
)

# 6) Replace decoded/decode-delta target_valid with slot mask under Exp34.
sub_once(
    r'(\s*)target_valid = \(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1target_valid = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "target_valid",
)

# 7) Replace main presence BCE with optional negative-class weighting.
sub_once(
    r'(\s*)presence_mask = entity_slot_mask \* valid_mask\.unsqueeze\(-1\)\s*\n\s*presence_loss = weighted_bce\(\s*presence_logits,\s*target_entity_mask,\s*presence_mask,\s*aux_weights,\s*\)',
    r'''\1presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
\1if presence_neg_class_weight != 1.0:
\1    presence_raw_bce = F.binary_cross_entropy_with_logits(
\1        presence_logits,
\1        target_entity_mask,
\1        reduction="none",
\1    )
\1    presence_time_w = aux_weights.view(1, 1, -1, 1)
\1    presence_class_w = torch.where(
\1        target_entity_mask > 0.5,
\1        torch.ones_like(target_entity_mask),
\1        torch.full_like(target_entity_mask, presence_neg_class_weight),
\1    )
\1    presence_weighted_mask = (
\1        presence_mask * presence_time_w * presence_class_w
\1    )
\1    presence_loss = (
\1        presence_raw_bce * presence_weighted_mask
\1    ).sum() / presence_weighted_mask.sum().clamp_min(1.0)
\1else:
\1    presence_loss = weighted_bce(
\1        presence_logits,
\1        target_entity_mask,
\1        presence_mask,
\1        aux_weights,
\1    )''',
    "presence_loss class weighting",
)

# 8) SIGReg target branch should follow target_encoding_mask_seq under Exp34.
sub_once(
    r'(\s*)reg_masks = torch\.cat\(\s*\[\s*observation_mask_seq,\s*target_entity_mask_seq_full\s*\],\s*dim=1,\s*\)',
    r'''\1reg_masks = torch.cat(
\1    [observation_mask_seq, target_encoding_mask_seq],
\1    dim=1,
\1)''',
    "reg_masks",
)

# 9) Stats valid mask should follow structural slot under Exp34.
sub_once(
    r'(\s*)latent_valid = \(\s*target_entity_mask\s*\*\s*entity_slot_mask\s*\*\s*valid_mask\.unsqueeze\(-1\)\s*\)',
    r'''\1latent_valid_entity = (
\1    entity_slot_mask if exp34_two_mask_loss else target_entity_mask * entity_slot_mask
\1)
\1latent_valid = latent_valid_entity * valid_mask.unsqueeze(-1)''',
    "latent_valid stats mask",
)

# 10) Add diagnostics into loss dict.
sub_once(
    r'("total_loss": total_loss,\s*)',
    r'''\1
        "exp34_two_mask_loss_enabled": torch.tensor(
            float(exp34_two_mask_loss),
            device=pred_latent.device,
        ),
        "exp35_simple_loss_enabled": torch.tensor(
            float(exp35_simple_loss),
            device=pred_latent.device,
        ),
        "presence_neg_class_weight_value": torch.tensor(
            float(presence_neg_class_weight),
            device=pred_latent.device,
        ),
        ''',
    "loss dict diagnostics",
)

OUT.write_text(out)
print(f"[ok] wrote {OUT}")

wrap = WRAP33.read_text()
wrap = wrap.replace(
    "from . import train_jepa_exp31_exp33 as _base",
    "from . import train_jepa_exp31_exp35 as _base",
)

wrap34 = wrap.replace(
    "def main() -> None:",
    'def main() -> None:\n    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")\n    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")',
)
WRAP34.write_text(wrap34)
print(f"[ok] wrote {WRAP34}")

wrap35 = wrap.replace(
    "def main() -> None:",
    'def main() -> None:\n    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")\n    os.environ.setdefault("SMAC_JEPA_EXP35_SIMPLE_LOSS", "1")\n    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")',
)
WRAP35.write_text(wrap35)
print(f"[ok] wrote {WRAP35}")
