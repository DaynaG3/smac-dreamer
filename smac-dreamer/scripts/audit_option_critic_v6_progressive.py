#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path
import torch
from omegaconf import OmegaConf

def fail(m): raise SystemExit(f'[FAIL] {m}')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def req(text,tokens,label):
 miss=[x for x in tokens if x not in text]
 if miss: fail(f'{label} missing contracts: {miss}')
def method(text,name):
 tree=ast.parse(text); ns=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
 if len(ns)!=1 or ns[0].end_lineno is None: fail(f'expected one {name}')
 lines=text.splitlines(True); n=ns[0]; return ''.join(lines[n.lineno-1:n.end_lineno])
def close(x,y,t=1e-12): return abs(float(x)-y)<=t

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--source-run-meta',type=Path,required=True); ap.add_argument('--require-v1-2-source',action='store_true'); a=ap.parse_args()
 repo=a.repo.resolve(); cp=Path(a.config); cp=cp if cp.is_absolute() else repo/cp
 for p in (cp,a.checkpoint,a.source_run_meta):
  if not p.is_file(): fail(f'missing {p}')
 cfg=OmegaConf.load(cp); h=cfg.hierarchical_options
 checks={
 'jepa_dense':str(cfg.world_model.backend)=='jepa' and str(cfg.reward.name)=='dense_v3',
 'eight_slots_two_groups':int(h.num_options)==8 and int(h.source_manager_group_count)==2,
 'progressive_unlock':int(h.slot_pair_unlock_initial_steps)==150000 and int(h.slot_pair_unlock_interval_steps)==200000 and int(h.slot_unlock_ramp_steps)==150000 and int(h.slot_pg_ramp_steps)==200000,
 'bounded_child_delta':close(h.slot_delta_scale_max,0.10),
 'reactive_floor':int(h.commitment_warmup_steps)==100000 and int(h.commitment_full_steps)==700000 and close(h.commitment_reselect_initial,1.0) and close(h.commitment_reselect_final,0.25),
 'staged_pg':int(h.worker_pg_warmup_steps)==20000 and int(h.worker_pg_full_steps)==150000 and int(h.manager_pg_warmup_steps)==100000 and int(h.manager_pg_full_steps)==500000,
 'late_termination':int(h.termination_warmup_steps)==400000 and int(h.termination_full_steps)==850000 and close(h.termination_max_probability_final,0.30),
 'source_trust':close(h.base_kl_scale,0.50) and close(h.manager_group_kl_scale,0.50) and close(h.action_preservation_scale,0.50),
 'no_forced_diversity':close(h.manager_collapse_scale,0.0) and close(h.manager_mi_scale,0.0) and close(h.action_diversity_scale,0.0) and close(h.residual_cosine_scale,0.0),
 'world_model_frozen':close(h.world_model_grad_scale_initial,0.0) and close(h.world_model_grad_scale_final,0.0),
 'horizon':int(h.imag_horizon_initial_max)==10 and int(h.imag_horizon_final_max)==12,
 'validation_200k':bool(cfg.validation.run_at_start) and int(cfg.validation.every)==200000,
 'fresh_uniform':str(cfg.sampling_mode)=='shuffled_round_robin' and not bool(cfg.adaptive_priority.enabled) and not bool(cfg.adaptive_priority.map.enabled) and not bool(cfg.adaptive_priority.sequence.enabled),
 }
 bad=[k for k,v in checks.items() if not v]
 if bad: fail(f'config contracts failed: {bad}')
 files={'o':repo/'external/r2dreamer/hierarchical_options.py','h':repo/'external/r2dreamer/hierarchical_dreamer.py','d':repo/'external/r2dreamer/dreamer.py','t':repo/'external/r2dreamer/trainer.py','tools':repo/'external/r2dreamer/tools.py','run':repo/'scripts/run_option_critic_v6_progressive_1m.sh','pipe':repo/'scripts/run_option_critic_v6_1m_then_exp45_pipeline.sh'}
 for p in files.values():
  if not p.is_file(): fail(f'missing installed source {p}')
 tx={k:p.read_text() for k,p in files.items()}
 req(tx['o'],('ARCHITECTURE = "dreamer_option_critic_v6_progressive_8slot"','manager_group','manager_slot','slot_gate_by_slot','slot_delta','manager_log_prob_components','slot_delta_scale_by_option'), 'options')
 req(tx['h'],('two_source_anchors_plus_six_progressive_child_slots','group_manager_loss','slot_manager_loss','selected_slot_pg_blend','target.manager_group','target.manager_slot','target.slot_delta'), 'hierarchy')
 req(tx['run'],('CURRENT_OPTION_CRITIC_V6_PROGRESSIVE_8SLOT_1M_RUN.txt','FINAL_STEP="${FINAL_STEP:-1000000}"','static_audit_option_critic_v6_progressive.sh'), 'launcher')
 req(tx['pipe'],('run_option_critic_v6_progressive_1m.sh','run_exp45_full_train_eval_resilient.sh','CONTINUE_ON_FAILURE'), 'pipeline')
 req(tx['tools'],('torch.bfloat16','.float()'), 'bf16')
 update=method(tx['d'],'update')
 if update.find('apply_hierarchy_gradient_guards(self)')<0: fail('gradient guard missing')
 grad=method(tx['d'],'_cal_grad_jepa'); l=grad.find('losses.pop(legacy_key, None)'); total=grad.find('total_loss = sum([v * self._loss_scales[k]')
 if l<0 or total<0 or l>total: fail('legacy behavior losses active')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); meta=ck.get('tactical_mixture_metadata') or {}; state=ck.get('agent_state_dict') or {}
 if a.require_v1_2_source:
  if meta.get('architecture')!='tactical_mixture_v1_2' or int(meta.get('num_tactics',-1))!=2: fail('wrong source checkpoint architecture')
  if any(k.startswith('hierarchical_options.') for k in state): fail('source already has hierarchy')
  if float(ck.get('val_macro_win_rate',-1))<0.3749: fail('source macro win below 0.375')
 print('[OK] Option-Critic v6 progressive source/config audit passed')
 print(json.dumps({'repo':str(repo),'config':str(cp),'checkpoint':{'path':str(a.checkpoint),'sha256':sha(a.checkpoint),'step':ck.get('step'),'val_macro_win_rate':ck.get('val_macro_win_rate')},'checks':checks},indent=2))
if __name__=='__main__': main()
