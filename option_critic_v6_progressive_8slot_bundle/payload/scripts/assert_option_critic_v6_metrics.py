#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def expected_gate(option:int,step:int)->float:
 slot=option//2
 if slot==0:return 1.0
 unlock=150000+(slot-1)*200000
 return min(max((step-unlock)/150000.0,0.0),1.0)
def expected_maturity(option:int,step:int)->float:
 slot=option//2
 if slot==0:return 1.0
 unlock=150000+(slot-1)*200000
 return min(max((step-unlock)/200000.0,0.0),1.0)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('run',type=Path); a=ap.parse_args(); p=a.run/'metrics.jsonl'
 if not p.is_file(): raise SystemExit(f'[FAIL] missing {p}')
 latest={}; step=-1
 for line in p.open():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  s=r.get('global_step',r.get('step',-1))
  if isinstance(s,(int,float)) and int(s)>=step: step=int(s); latest.update({k:v for k,v in r.items() if isinstance(v,(int,float))})
 base=(
 'train/option/legacy_behavior_losses_disabled','train/option/world_model_grad_scale',
 'train/option/manager_pg_blend','train/option/worker_pg_blend','train/option/termination_blend',
 'train/option/commitment_reselect_probability','train/option/imag_horizon',
 'train/option/source_policy_kl_mean','train/option/source_policy_kl_tail',
 'train/option/real_source_policy_kl_mean','train/option/real_source_policy_kl_tail',
 'train/option/source_manager_group_kl_mean','train/option/source_manager_group_kl_tail',
 'train/option/real_source_manager_group_kl_mean','train/option/real_source_manager_group_kl_tail',
 'train/option/high_confidence_action_flip_rate','train/option/real_source_high_confidence_action_flip_rate',
 'train/option/real_min_duration_violation_rate','train/option/real_max_duration_violation_rate',
 'train/option/real_change_without_boundary_rate','train/option/imag_min_duration_violation_rate',
 'train/option/imag_max_duration_violation_rate','train/option/imag_change_without_boundary_rate',
 )
 req=list(base)
 for i in range(8): req += [f'train/option/usage_{i}',f'train/option/real_usage_{i}',f'train/option/slot_gate_{i}',f'train/option/slot_delta_scale_{i}',f'train/option/slot_pg_blend_{i}']
 miss=[k for k in req if k not in latest]
 if miss: raise SystemExit(f'[FAIL] missing v6 metrics: {miss}')
 bad=[k for k in req if not math.isfinite(float(latest[k]))]
 if bad: raise SystemExit(f'[FAIL] nonfinite metrics: {bad}')
 if float(latest['train/option/legacy_behavior_losses_disabled'])!=1: raise SystemExit('[FAIL] legacy losses active')
 if abs(float(latest['train/option/world_model_grad_scale']))>1e-8: raise SystemExit('[FAIL] world model not frozen')
 for k in req:
  if 'violation_rate' in k or 'change_without_boundary' in k:
   if abs(float(latest[k]))>1e-8: raise SystemExit(f'[FAIL] state-machine violation {k}={latest[k]}')
 for k in ('train/imag_post_mask_invalid_sample_rate','train/real_post_mask_invalid_sample_rate'):
  if float(latest.get(k,0.0))>1e-6: raise SystemExit(f'[FAIL] invalid actions: {k}')
 for i in range(8):
  gate=float(latest[f'train/option/slot_gate_{i}']); exp=expected_gate(i,step)
  if abs(gate-exp)>2e-3: raise SystemExit(f'[FAIL] slot gate {i}: {gate} != {exp}')
  mat=float(latest[f'train/option/slot_pg_blend_{i}']); exm=expected_maturity(i,step)
  if abs(mat-exm)>2e-3: raise SystemExit(f'[FAIL] slot maturity {i}: {mat} != {exm}')
  if exp==0 and float(latest[f'train/option/usage_{i}'])>1e-7: raise SystemExit(f'[FAIL] locked option {i} has manager usage')
  if exp==0 and float(latest[f'train/option/real_usage_{i}'])>1e-7: raise SystemExit(f'[FAIL] locked option {i} has real usage')
 for k,lim in {
 'train/option/source_policy_kl_mean':.05,'train/option/real_source_policy_kl_mean':.05,
 'train/option/source_policy_kl_tail':.15,'train/option/real_source_policy_kl_tail':.15,
 'train/option/source_manager_group_kl_mean':.03,'train/option/real_source_manager_group_kl_mean':.03,
 'train/option/source_manager_group_kl_tail':.10,'train/option/real_source_manager_group_kl_tail':.10}.items():
  if float(latest[k])>lim: raise SystemExit(f'[FAIL] trust region unsafe: {k}={latest[k]}')
 if not 7<=round(float(latest['train/option/imag_horizon']))<=12: raise SystemExit('[FAIL] bad imagination horizon')
 print(f'[OK] Option-Critic v6 progressive metrics passed; latest_step={step}')
 print('[OK] eight slots present; locked slots unused; gates, source trust, and structural invariants valid')
if __name__=='__main__':main()
