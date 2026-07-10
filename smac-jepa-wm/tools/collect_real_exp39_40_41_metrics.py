#!/usr/bin/env python3
import json
from pathlib import Path
ROOT = Path("eval_outputs")
EXPS = ["exp39_probe_action_5ep", "exp40_event_balanced_5ep", "exp41_hidden_change_gate_5ep"]
KEYS = {
 "native_dynamic_mae":["native_dynamic_mae","decoded_dynamic_mae","rollout_native_dynamic_mae"],
 "probe_rollout_mae":["probe_rollout_dynamic_mae","probe_rollout_mae","rollout_probe_dynamic_mae"],
 "presence_f1":["presence_f1","presence_overall_f1"],
 "action_shuffle_penalty":["action_shuffle_penalty","action_shuffle_dynamic_mae_penalty","action_shuffle_decoded_mae_penalty"],
 "hidden_native_mae":["controlled_masked_native_dynamic_mae","controlled_hidden_dynamic_mae"],
 "hidden_probe_mae":["controlled_masked_probe_dynamic_mae","controlled_hidden_probe_dynamic_mae"],
 "reappearance_mae":["controlled_reappearance_native_dynamic_mae","controlled_reappear_native_dynamic_mae"],
 "changed_mae":["controlled_masked_native_changed_dynamic_mae"],
 "stable_mae":["controlled_masked_native_unchanged_dynamic_mae"],
 "last_seen_changed_mae":["controlled_last_seen_changed_dynamic_mae"],
}
def load(exp):
    d={}
    for p in (ROOT/exp).rglob("*.json"):
        try: obj=json.loads(p.read_text())
        except Exception: continue
        if isinstance(obj,dict): d.update({k:v for k,v in obj.items() if isinstance(v,(int,float,str,bool)) or v is None})
    return d
def pick(d,ks):
    for k in ks:
        if k in d: return d[k]
    for k in ks:
        for kk,v in d.items():
            if kk.endswith(k): return v
    return None
def fmt(v):
    if v is None: return ""
    if isinstance(v,float): return f"{v:.6g}"
    return str(v)
rows=[]
for exp in EXPS:
    d=load(exp); row={"exp":exp}
    for c,ks in KEYS.items(): row[c]=pick(d,ks)
    rows.append(row)
cols=["exp"]+list(KEYS)
widths={c:max(len(c),*(len(fmt(r.get(c))) for r in rows)) for c in cols}
print(" | ".join(c.ljust(widths[c]) for c in cols))
print("-+-".join("-"*widths[c] for c in cols))
for r in rows: print(" | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))
