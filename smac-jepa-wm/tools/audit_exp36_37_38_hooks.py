#!/usr/bin/env python3
from pathlib import Path

files = [p for p in [
    Path("smac_jepa/train_jepa_exp31_exp35.py"),
    Path("smac_jepa/train_jepa_exp31_exp33.py"),
] if p.exists()]
if not files:
    raise SystemExit("No trainer files found.")

text = "\n".join(p.read_text(errors="ignore") for p in files)

hooks = [
    "SMAC_JEPA_EXP34_TWO_MASK_LOSS",
    "SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT",
    "SMAC_JEPA_EXP36_ACTION_CHANGE",
    "SMAC_JEPA_CHANGED_COORD_WEIGHT",
    "SMAC_JEPA_ACTION_SHUFFLE_CONTRAST_WEIGHT",
    "SMAC_JEPA_ACTION_SHUFFLE_MARGIN",
    "SMAC_JEPA_ACTION_EFFECTIVE_WINDOW_WEIGHT",
    "SMAC_JEPA_EXP37_PROBE_CLEAN",
    "SMAC_JEPA_DECODER_WEIGHT_SCALE",
    "SMAC_JEPA_BARLOW_WEIGHT_SCALE",
    "SMAC_JEPA_REP_WEIGHT_SCALE",
    "SMAC_JEPA_SIGREG_WEIGHT_SCALE",
    "SMAC_JEPA_PROBE_ALIGN_WEIGHT",
    "SMAC_JEPA_EXP38_ANCHORED_RESIDUAL",
    "SMAC_JEPA_USE_ANCHORED_BELIEF",
    "SMAC_JEPA_ANCHOR_RESIDUAL_HIDDEN",
    "SMAC_JEPA_ANCHOR_GATE_BIAS",
    "SMAC_JEPA_ANCHOR_STABLE_WEIGHT",
    "SMAC_JEPA_ANCHOR_CHANGED_WEIGHT",
    "SMAC_JEPA_DELTA_WEIGHT",
    "SMAC_JEPA_HIDDEN_RECON_WEIGHT",
    "SMAC_JEPA_REAPPEAR_WEIGHT",
]

print("Hook audit over:", ", ".join(str(p) for p in files))
for h in hooks:
    print(f"{h:48s} {'FOUND' if h in text else 'not found'}")

print()
print("FOUND means the trainer can react to that env key.")
print("not found means the wrapper still runs, but that specific intended change may be a no-op unless already wired under another name.")
