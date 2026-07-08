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

def show_context(s: str, needle: str) -> None:
    i = s.find(needle)
    print(f"\n--- context for {needle!r}; index={i} ---", file=sys.stderr)
    if i >= 0:
        print(s[max(0, i - 500): i + 1500], file=sys.stderr)

def sub_required(s: str, pattern: str, repl: str, label: str, flags=re.S) -> str:
    new, n = re.subn(pattern, repl, s, count=1, flags=flags)
    if n != 1:
        print(f"[patch-error] {label}: expected 1 match, found {n}", file=sys.stderr)
        for needle in [
            "del detach_rollout_targets",
            'slot_mask_seq = batch["entity_slot_mask_seq"]',
            "online_target_latents_raw",
            "latent_mask =",
            "dynamic_reference_mask =",
            "target_valid =",
            "presence_loss = weighted_bce",
            "reg_masks = torch.cat",
            '"total_loss": total_loss',
        ]:
            show_context(s, needle)
        sys.exit(1)
    print(f"[ok] patched {label}")
    return new

if not BASE.exists():
    die(f"missing {BASE}")
if not WRAP33.exists():
    die(f"missing {WRAP33}")

s = BASE.read_text()
out = s

# ---------------------------------------------------------------------
# 1. Inject Exp34/Exp35 environment flags inside markov_rollout_rnn_losses.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'(\n[ \t]*)del[ \t]+detach_rollout_targets[ \t]*\n([ \t]*)entity_seq[ \t]*=[ \t]*batch\["entity_seq"\]',
    r'''\1del detach_rollout_targets
\2_exp_os = __import__("os")
\2exp34_two_mask_loss = _exp_os.environ.get("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "0") == "1"
\2exp35_simple_loss = _exp_os.environ.get("SMAC_JEPA_EXP35_SIMPLE_LOSS", "0") == "1"
\2presence_neg_class_weight = float(_exp_os.environ.get("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "1.0"))

\2# Exp35 simple-loss ablation:
\2# keep core JEPA dynamics, one-step anchor, decoder, presence, and SIGReg.
\2# remove broader auxiliary pressure after Exp34 mask cleanup is enabled.
\2if exp35_simple_loss:
\2    r2_rep_scale = 0.0
\2    r2_barlow_scale = 0.0
\2    memory_barlow_scale = 0.0
\2    delta_loss_weight = 0.0
\2    hidden_reconstruction_weight = 0.0
\2    last_seen_anchor_weight = 0.0
\2    hidden_presence_weight = 0.0
\2    reappearance_consistency_weight = 0.0
\2    inverse_dynamics_weight = 0.0

\2entity_seq = batch["entity_seq"]''',
    "env flags",
)

# ---------------------------------------------------------------------
# 2. Define target_encoding_mask_seq immediately after slot_mask_seq.
#    This is the line that failed before; now it does NOT depend on feature_valid.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*slot_mask_seq[ \t]*=[ \t]*batch\["entity_slot_mask_seq"\][ \t]*\n)',
    r'''\1    target_encoding_mask_seq = (
        slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full
    )
''',
    "target_encoding_mask_seq after slot_mask_seq",
)

# ---------------------------------------------------------------------
# 3. Target encoders / target latent normalisation use slot mask in Exp34.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'(online_target_latents_raw[ \t]*=[ \t]*model\.encoder\(\s*target_entity_seq_full,\s*)target_entity_mask_seq_full(\s*,\s*\))',
    r'\1target_encoding_mask_seq\2',
    "online target encoder mask",
)

out = sub_required(
    out,
    r'(consistency_target_latents_raw[ \t]*=[ \t]*target_encoder\(\s*target_entity_seq_full,\s*)target_entity_mask_seq_full(\s*,\s*\))',
    r'\1target_encoding_mask_seq\2',
    "EMA target encoder mask",
)

out = sub_required(
    out,
    r'(online_target_latents[ \t]*=[ \t]*r2_normalize_latent\(\s*online_target_latents_raw,\s*)target_entity_mask_seq_full(\s*,)',
    r'\1target_encoding_mask_seq\2',
    "online target latent normalize mask",
)

out = sub_required(
    out,
    r'(target_latents[ \t]*=[ \t]*r2_normalize_latent\(\s*consistency_target_latents_raw,\s*)target_entity_mask_seq_full(\s*,)',
    r'\1target_encoding_mask_seq\2',
    "EMA target latent normalize mask",
)

# ---------------------------------------------------------------------
# 4. Main latent loss mask.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)latent_mask[ \t]*=[ \t]*\(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1target_presence_valid = (
\1    target_entity_mask.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)
\1target_slot_valid = (
\1    entity_slot_mask.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)
\1latent_mask = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "latent_mask two-mask",
)

# ---------------------------------------------------------------------
# 5. Dynamic reference/event mask: dead structural slots are valid in Exp34.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)dynamic_reference_mask[ \t]*=[ \t]*\(\s*dynamic_feature_valid\[:,[ \t]*None,[ \t]*None\]\s*\.expand_as\(target_entity\)\s*\*\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1dynamic_reference_entity_valid = (
\1    entity_slot_mask if exp34_two_mask_loss else target_entity_mask
\1)
\1dynamic_reference_mask = (
\1    dynamic_feature_valid[:, None, None]
\1    .expand_as(target_entity)
\1    * dynamic_reference_entity_valid.unsqueeze(-1)
\1    * valid_mask.unsqueeze(-1).unsqueeze(-1)
\1)''',
    "dynamic_reference_mask two-mask",
)

# ---------------------------------------------------------------------
# 6. Decoder/delta/hidden target_valid mask.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)target_valid[ \t]*=[ \t]*\(\s*target_entity_mask\.unsqueeze\(-1\)\s*\*\s*valid_mask\.unsqueeze\(-1\)\.unsqueeze\(-1\)\s*\)',
    r'''\1target_valid = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "target_valid two-mask",
)

# ---------------------------------------------------------------------
# 7. Main presence loss: optional negative class weighting.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)presence_mask[ \t]*=[ \t]*entity_slot_mask[ \t]*\*[ \t]*valid_mask\.unsqueeze\(-1\)\s*\n[ \t]*presence_loss[ \t]*=[ \t]*weighted_bce\(\s*presence_logits,\s*target_entity_mask,\s*presence_mask,\s*aux_weights,\s*\)',
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
\1    presence_weighted_mask = presence_mask * presence_time_w * presence_class_w
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
    "presence negative class weighting",
)

# ---------------------------------------------------------------------
# 8. SIGReg target side uses target_encoding_mask_seq in Exp34.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)reg_masks[ \t]*=[ \t]*torch\.cat\(\s*\[\s*observation_mask_seq,\s*target_entity_mask_seq_full\s*\],\s*dim=1,\s*\)',
    r'''\1reg_masks = torch.cat(
\1    [observation_mask_seq, target_encoding_mask_seq],
\1    dim=1,
\1)''',
    "reg_masks two-mask",
)

# ---------------------------------------------------------------------
# 9. Diagnostic latent stats mask. This is not required for training,
#    but it keeps stats consistent.
# ---------------------------------------------------------------------
out, n = re.subn(
    r'([ \t]*)latent_valid[ \t]*=[ \t]*\(\s*target_entity_mask\s*\*\s*entity_slot_mask\s*\*\s*valid_mask\.unsqueeze\(-1\)\s*\)',
    r'''\1latent_valid_entity = (
\1    entity_slot_mask
\1    if exp34_two_mask_loss
\1    else target_entity_mask * entity_slot_mask
\1)
\1latent_valid = latent_valid_entity * valid_mask.unsqueeze(-1)''',
    out,
    count=1,
    flags=re.S,
)
print(f"[ok] diagnostic latent_valid patches: {n}")

# ---------------------------------------------------------------------
# 10. Add flags to logged losses.
# ---------------------------------------------------------------------
out = sub_required(
    out,
    r'([ \t]*)"total_loss":[ \t]*total_loss,',
    r'''\1"total_loss": total_loss,
\1"exp34_two_mask_loss_enabled": torch.tensor(
\1    float(exp34_two_mask_loss),
\1    device=pred_latent.device,
\1),
\1"exp35_simple_loss_enabled": torch.tensor(
\1    float(exp35_simple_loss),
\1    device=pred_latent.device,
\1),
\1"presence_neg_class_weight_value": torch.tensor(
\1    float(presence_neg_class_weight),
\1    device=pred_latent.device,
\1),''',
    "loss dict diagnostics",
)

OUT.write_text(out)
print(f"[ok] wrote {OUT}")

# ---------------------------------------------------------------------
# Create Dreamer-compatible wrappers.
# ---------------------------------------------------------------------
wrap = WRAP33.read_text()
if "train_jepa_exp31_exp33" not in wrap:
    die("Could not find train_jepa_exp31_exp33 import/reference in wrapper.")

wrap = wrap.replace("train_jepa_exp31_exp33", "train_jepa_exp31_exp35")

if "def main() -> None:" not in wrap:
    die("Could not find def main() in wrapper.")

wrap34 = wrap.replace(
    "def main() -> None:",
    '''def main() -> None:
    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")
    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")''',
    1,
)
WRAP34.write_text(wrap34)
print(f"[ok] wrote {WRAP34}")

wrap35 = wrap.replace(
    "def main() -> None:",
    '''def main() -> None:
    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")
    os.environ.setdefault("SMAC_JEPA_EXP35_SIMPLE_LOSS", "1")
    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")''',
    1,
)
WRAP35.write_text(wrap35)
print(f"[ok] wrote {WRAP35}")

print("[done] generated Exp34/Exp35 files")
