#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("eval_outputs")
EXPS = [
    "exp36_action_change_real_5ep",
    "exp37_probe_clean_real_5ep",
    "exp38_anchor_real_5ep",
]

KEYS = {
    "native_dynamic_mae": ["native_dynamic_mae", "decoded_dynamic_mae", "rollout_native_dynamic_mae"],
    "probe_rollout_mae": ["probe_rollout_dynamic_mae", "probe_rollout_mae", "rollout_probe_dynamic_mae"],
    "presence_f1": ["presence_f1", "presence_overall_f1"],
    "presence_precision": ["presence_precision", "presence_overall_precision"],
    "presence_recall": ["presence_recall", "presence_overall_recall"],
    "action_shuffle_penalty": ["action_shuffle_penalty", "action_shuffle_dynamic_mae_penalty", "action_shuffle_decoded_mae_penalty"],
    "hidden_native_mae": ["controlled_masked_native_dynamic_mae", "controlled_hidden_dynamic_mae"],
    "hidden_probe_mae": ["controlled_masked_probe_dynamic_mae", "controlled_hidden_probe_dynamic_mae"],
    "hidden_age1_mae": ["controlled_masked_native_age1_dynamic_mae"],
    "hidden_age3_5_mae": ["controlled_masked_native_age3_5_dynamic_mae"],
    "reappearance_mae": ["controlled_reappearance_native_dynamic_mae", "controlled_reappear_native_dynamic_mae"],
    "memory_gain": ["controlled_native_memory_gain", "controlled_memory_gain"],
    "occlusion_cost": ["controlled_native_occlusion_cost", "controlled_occlusion_cost"],
    "changed_mae": ["controlled_masked_native_changed_dynamic_mae"],
    "stable_mae": ["controlled_masked_native_unchanged_dynamic_mae"],
    "last_seen_hidden_mae": ["controlled_last_seen_dynamic_mae"],
    "last_seen_changed_mae": ["controlled_last_seen_changed_dynamic_mae"],
}

def load(exp):
    d = {}
    for p in (ROOT / exp).rglob("*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(obj, dict):
            d.update({k:v for k,v in obj.items() if isinstance(v, (int,float,str,bool)) or v is None})
    return d

def pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    for k in keys:
        for kk, v in d.items():
            if kk.endswith(k):
                return v
    return None

def fmt(v):
    if v is None: return ""
    if isinstance(v, float): return f"{v:.6g}"
    return str(v)

rows = []
for exp in EXPS:
    d = load(exp)
    row = {"exp": exp}
    for col, keys in KEYS.items():
        row[col] = pick(d, keys)
    rows.append(row)

cols = ["exp"] + list(KEYS)
widths = {c:max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print(" | ".join(c.ljust(widths[c]) for c in cols))
print("-+-".join("-"*widths[c] for c in cols))
for r in rows:
    print(" | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))
