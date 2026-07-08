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

def context(s: str, needle: str) -> str:
    i = s.find(needle)
    if i < 0:
        return f"{needle!r} not found"
    return s[max(0, i - 300): i + 900]

def replace_exact(s: str, old: str, new: str, label: str, *, required: bool = True) -> str:
    n = s.count(old)
    if n != 1:
        msg = f"{label}: expected 1 exact match, found {n}"
        if required:
            print("\n[patch-error]", msg, file=sys.stderr)
            print(context(s, old[:80]), file=sys.stderr)
            sys.exit(1)
        print("[patch-warn]", msg, file=sys.stderr)
        return s
    return s.replace(old, new, 1)

def replace_regex(s: str, pattern: str, repl: str, label: str, *, required: bool = True) -> str:
    new, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        msg = f"{label}: expected 1 regex match, found {n}"
        if required:
            print("\n[patch-error]", msg, file=sys.stderr)
            for needle in [
                "del detach_rollout_targets",
                'slot_mask_seq = batch["entity_slot_mask_seq"]',
                "latent_mask =",
                "target_valid =",
                "presence_loss =",
                "reg_masks =",
                "latent_valid =",
                '"total_loss": total_loss,',
            ]:
                print("\n---", needle, "---", file=sys.stderr)
                print(context(s, needle), file=sys.stderr)
            sys.exit(1)
        print("[patch-warn]", msg, file=sys.stderr)
        return s
    return new

if not BASE.exists():
    die(f"missing {BASE}")
if not WRAP33.exists():
    die(f"missing {WRAP33}")

s = BASE.read_text()
out = s

# 1. Add env flags inside loss function.
out = replace_regex(
    out,
    r'(\n\s*)del detach_rollout_targets\s*\n(\s*)entity_seq = batch\["entity_seq"\]',
    r'''\1del detach_rollout_targets
\2_exp_os = __import__("os")
\2exp34_two_mask_loss = _exp_os.environ.get("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "0") == "1"
\2exp35_simple_loss = _exp_os.environ.get("SMAC_JEPA_EXP35_SIMPLE_LOSS", "0") == "1"
\2presence_neg_class_weight = float(_exp_os.environ.get("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "1.0"))

\2# Exp35 loss ablation: keep core dynamics / one-step / decoder / presence / SIGReg.
\2# Remove the broader auxiliary pressure after Exp34 mask cleanup is applied.
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
    "inject exp34/exp35 env flags",
)

# 2. Define target_encoding_mask_seq after slot_mask_seq is available.
out = replace_exact(
    out,
    '''    slot_mask_seq = batch["entity_slot_mask_seq"]
    feature_valid = batch["feature_valid_mask"]''',
    '''    slot_mask_seq = batch["entity_slot_mask_seq"]
    target_encoding_mask_seq = (
        slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full
    )
    feature_valid = batch["feature_valid_mask"]''',
    "insert target_encoding_mask_seq",
)

# 3. Target encoders use structural slots under Exp34.
out = replace_exact(
    out,
    '''    online_target_latents_raw = model.encoder(
        target_entity_seq_full,
        target_entity_mask_seq_full,
    )''',
    '''    online_target_latents_raw = model.encoder(
        target_entity_seq_full,
        target_encoding_mask_seq,
    )''',
    "online target encoder mask",
)

out = replace_exact(
    out,
    '''            consistency_target_latents_raw = target_encoder(
                target_entity_seq_full,
                target_entity_mask_seq_full,
            )''',
    '''            consistency_target_latents_raw = target_encoder(
                target_entity_seq_full,
                target_encoding_mask_seq,
            )''',
    "ema target encoder mask",
)

out = replace_exact(
    out,
    '''    online_target_latents = r2_normalize_latent(
        online_target_latents_raw,
        target_entity_mask_seq_full,
        enabled=r2_latent_normalize,
    )
    target_latents = r2_normalize_latent(
        consistency_target_latents_raw,
        target_entity_mask_seq_full,
        enabled=r2_latent_normalize,
    )''',
    '''    online_target_latents = r2_normalize_latent(
        online_target_latents_raw,
        target_encoding_mask_seq,
        enabled=r2_latent_normalize,
    )
    target_latents = r2_normalize_latent(
        consistency_target_latents_raw,
        target_encoding_mask_seq,
        enabled=r2_latent_normalize,
    )''',
    "target latent normalize masks",
)

# 4. Main latent mask: Exp34 can supervise structural slots, not only present slots.
out = replace_exact(
    out,
    '''    latent_mask = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )''',
    '''    target_presence_valid = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )
    target_slot_valid = (
        entity_slot_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )
    latent_mask = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "latent_mask two-mask replacement",
)

# 5. Event/change reference mask uses structural slots under Exp34.
out = replace_exact(
    out,
    '''    dynamic_reference_mask = (
        dynamic_feature_valid[:, None, None]
        .expand_as(target_entity)
        * target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )''',
    '''    dynamic_reference_entity_valid = (
        entity_slot_mask if exp34_two_mask_loss else target_entity_mask
    )
    dynamic_reference_mask = (
        dynamic_feature_valid[:, None, None]
        .expand_as(target_entity)
        * dynamic_reference_entity_valid.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )''',
    "dynamic_reference_mask two-mask replacement",
)

# 6. Decoder/delta/hidden masks use structural slots under Exp34.
out = replace_exact(
    out,
    '''    target_valid = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )''',
    '''    target_valid = target_slot_valid if exp34_two_mask_loss else target_presence_valid''',
    "target_valid two-mask replacement",
)

# 7. Presence BCE optionally weights negative/dead class.
out = replace_exact(
    out,
    '''    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(
        presence_logits,
        target_entity_mask,
        presence_mask,
        aux_weights,
    )''',
    '''    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    if presence_neg_class_weight != 1.0:
        presence_raw_bce = F.binary_cross_entropy_with_logits(
            presence_logits,
            target_entity_mask,
            reduction="none",
        )
        presence_time_w = aux_weights.view(1, 1, -1, 1)
        presence_class_w = torch.where(
            target_entity_mask > 0.5,
            torch.ones_like(target_entity_mask),
            torch.full_like(target_entity_mask, presence_neg_class_weight),
        )
        presence_weighted_mask = presence_mask * presence_time_w * presence_class_w
        presence_loss = (
            presence_raw_bce * presence_weighted_mask
        ).sum() / presence_weighted_mask.sum().clamp_min(1.0)
    else:
        presence_loss = weighted_bce(
            presence_logits,
            target_entity_mask,
            presence_mask,
            aux_weights,
        )''',
    "presence negative class weighting",
)

# 8. SIGReg target masks follow Exp34 target encoding mask.
out = replace_exact(
    out,
    '''        reg_masks = torch.cat(
            [observation_mask_seq, target_entity_mask_seq_full],
            dim=1,
        )''',
    '''        reg_masks = torch.cat(
            [observation_mask_seq, target_encoding_mask_seq],
            dim=1,
        )''',
    "reg_masks target_encoding_mask_seq",
)

# 9. Stats mask is diagnostic only. Patch if exact local format matches; do not abort.
out = replace_exact(
    out,
    '''    latent_valid = (
        target_entity_mask
        * entity_slot_mask
        * valid_mask.unsqueeze(-1)
    )''',
    '''    latent_valid_entity = (
        entity_slot_mask
        if exp34_two_mask_loss
        else target_entity_mask * entity_slot_mask
    )
    latent_valid = latent_valid_entity * valid_mask.unsqueeze(-1)''',
    "latent_valid diagnostic mask",
    required=False,
)

# 10. Add diagnostics to logged loss dict.
out = replace_exact(
    out,
    '''        "total_loss": total_loss,''',
    '''        "total_loss": total_loss,
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
        ),''',
    "loss dict diagnostics",
)

OUT.write_text(out)
print(f"[ok] wrote {OUT}")

wrap = WRAP33.read_text()
if "from . import train_jepa_exp31_exp33 as _base" not in wrap:
    die("wrapper import line not found in train_jepa_exp33_dreamer.py")

wrap = wrap.replace(
    "from . import train_jepa_exp31_exp33 as _base",
    "from . import train_jepa_exp31_exp35 as _base",
    1,
)

if "def main() -> None:" not in wrap:
    die("def main() not found in wrapper")

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

print("[ok] generated Exp34/Exp35 trainer files")
