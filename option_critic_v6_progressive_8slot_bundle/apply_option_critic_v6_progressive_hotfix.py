#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, pathlib, shutil, subprocess
from omegaconf import OmegaConf

BUNDLE=pathlib.Path(__file__).resolve().parent
PAYLOAD=BUNDLE/'payload'
MANIFEST=BUNDLE/'MANIFEST.sha256.json'
V5_ARCH='ARCHITECTURE = "dreamer_option_critic_v5_stability"'
V6_ARCH='ARCHITECTURE = "dreamer_option_critic_v6_progressive_8slot"'
REPLACED=(
 'external/r2dreamer/hierarchical_options.py',
 'external/r2dreamer/hierarchical_dreamer.py',
 'external/r2dreamer/option_critic.py',
 'tests/test_hierarchical_options.py','tests/test_option_critic_math.py',
 'tests/test_hierarchical_auxiliary.py','tests/test_hierarchy_migration.py',
)
INTRODUCED=(
 'configs/r2_2100_jepa_option_critic_8_v6_progressive_1m.yaml',
 'scripts/audit_option_critic_v6_progressive.py',
 'scripts/static_audit_option_critic_v6_progressive.sh',
 'scripts/assert_option_critic_v6_metrics.py',
 'scripts/run_option_critic_v6_progressive_1m.sh',
 'scripts/run_option_critic_v6_1m_then_exp45_pipeline.sh',
)

def die(msg): raise SystemExit(f'[FAIL] {msg}')
def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def verify_manifest():
 m=json.loads(MANIFEST.read_text()); e=m.get('files')
 if m.get('schema_version')!=1 or not isinstance(e,dict): die('invalid manifest')
 for rel,expected in e.items():
  p=BUNDLE/rel
  if not p.is_file(): die(f'missing bundle file: {rel}')
  if sha(p)!=expected: die(f'hash mismatch: {rel}')
 print(f'[OK] verified {len(e)} v6 bundle hashes')

def build_config(repo):
 src=repo/'configs/r2_2100_jepa_option_critic_2_v5_stability_1m.yaml'
 if not src.is_file(): die(f'integrated v5 config missing: {src}')
 cfg=OmegaConf.load(src)
 if str(cfg.world_model.backend)!='jepa' or str(cfg.reward.name)!='dense_v3': die('source config contract changed')
 h=OmegaConf.to_container(cfg.hierarchical_options,resolve=True)
 h.update({
  'enabled':True,'num_options':8,'source_manager_group_count':2,
  'min_duration':1,'max_duration':8,
  'slot_manager_unimix':0.01,
  'slot_pair_unlock_initial_steps':150000,
  'slot_pair_unlock_interval_steps':200000,
  'slot_unlock_ramp_steps':150000,
  'slot_pg_ramp_steps':200000,
  'slot_delta_scale_max':0.10,
  'commitment_warmup_steps':100000,'commitment_full_steps':700000,
  'commitment_reselect_initial':1.0,'commitment_reselect_final':0.25,
  'worker_pg_warmup_steps':20000,'worker_pg_full_steps':150000,
  'manager_pg_warmup_steps':100000,'manager_pg_full_steps':500000,
  'termination_warmup_steps':400000,'termination_full_steps':850000,
  'termination_max_probability_during_ramp':0.30,
  'termination_max_probability_final':0.30,
  'termination_cap_full_steps':900000,'termination_loss_scale':0.02,
  'manager_unimix_initial':0.0,'manager_unimix_final':0.005,
  'manager_unimix_decay_steps':600000,
  'manager_collapse_scale':0.0,'manager_mi_scale':0.0,
  'action_diversity_scale':0.0,'residual_cosine_scale':0.0,
  'base_kl_target':0.002,'base_kl_tail_target':0.01,'base_kl_scale':0.50,
  'action_preservation_scale':0.50,
  'manager_group_kl_target':0.001,'manager_group_kl_tail_target':0.005,
  'manager_group_kl_scale':0.50,'manager_group_preservation_scale':0.50,
  'max_diversity_states':2048,
  'world_model_grad_scale_initial':0.0,'world_model_grad_scale_final':0.0,
  'imag_horizon_initial_max':10,'imag_horizon_final_max':12,
  'imag_horizon_window':4,'imag_horizon_ramp_steps':600000,
 })
 cfg.hierarchical_options=OmegaConf.create(h)
 cfg.tactical_mixture.enabled=False
 cfg.sampling_mode='shuffled_round_robin'
 cfg.adaptive_priority.enabled=False; cfg.adaptive_priority.map.enabled=False; cfg.adaptive_priority.sequence.enabled=False
 cfg.buffer.scratch_dir='replay'; cfg.validation.run_at_start=True; cfg.validation.every=200000
 if 'compile' in cfg: cfg.compile=False
 if 'model' in cfg and 'compile' in cfg.model: cfg.model.compile=False
 if 'wandb' in cfg: cfg.wandb.run_name='tactical_v12_option_critic_v6_progressive_8slot_1m'
 OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)
 return repo/'configs/r2_2100_jepa_option_critic_8_v6_progressive_1m.yaml',cfg

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=pathlib.Path,required=True); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
 verify_manifest(); repo=a.repo.expanduser().resolve()
 try: git_root=pathlib.Path(subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'],text=True).strip())
 except Exception: die(f'{repo} not in git worktree')
 print('[INFO] Git root:',git_root); print('[INFO] Target subtree:',repo)
 excluded=set(); pid=os.getpid()
 while pid>1 and pid not in excluded:
  excluded.add(pid)
  try: pid=int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
  except Exception: break
 active=[]
 for e in pathlib.Path('/proc').iterdir():
  if not e.name.isdigit() or int(e.name) in excluded: continue
  try: cmd=(e/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except Exception: continue
  if 'train_r2dreamer_smaclite_multimap.py' in cmd: active.append((e.name,cmd.strip()))
 if active: die(f'stop active trainer first: {active[:4]}')
 for rel in REPLACED:
  if not (repo/rel).is_file(): die(f'missing installed source: {rel}')
 text=(repo/'external/r2dreamer/hierarchical_options.py').read_text()
 if V6_ARCH in text: die('v6 already installed')
 if V5_ARCH not in text: die('expected integrated v5 source')
 if not (repo/'scripts/run_exp45_full_train_eval_resilient.sh').is_file(): die('resilient Exp45 script missing')
 for rel in REPLACED:
  p=PAYLOAD/rel
  if not p.is_file(): die(f'missing payload: {rel}')
  if p.suffix=='.py': compile(p.read_text(),str(p),'exec')
 for rel in INTRODUCED[1:]:
  p=PAYLOAD/rel
  if not p.is_file(): die(f'missing payload: {rel}')
  if p.suffix=='.py': compile(p.read_text(),str(p),'exec')
 target,cfg=build_config(repo)
 collisions=[rel for rel in INTRODUCED if (repo/rel).exists()]
 if collisions: die(f'refusing to overwrite v6 files: {collisions}')
 if a.dry_run:
  print('[OK] v6 progressive dry-run matched integrated v5, parsed payloads, and resolved 8-slot config'); return
 stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S'); backup=repo.parent/f'{repo.name}_option_critic_v6_progressive_backup_{stamp}'; backup.mkdir()
 try:
  for rel in REPLACED:
   d=backup/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/rel,d)
  for rel in REPLACED:
   d=repo/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(PAYLOAD/rel,d)
  OmegaConf.save(cfg,target)
  for rel in INTRODUCED[1:]:
   d=repo/rel; d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(PAYLOAD/rel,d); d.chmod(0o755)
  m={'schema_version':1,'repo':str(repo),'backed_up_files':list(REPLACED),'backed_up_sha256':{r:sha(backup/r) for r in REPLACED},'introduced_files':list(INTRODUCED)}
  (backup/'option_critic_v6_progressive_backup_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 except Exception:
  for rel in REPLACED:
   if (backup/rel).exists(): shutil.copy2(backup/rel,repo/rel)
  for rel in INTRODUCED:
   if (repo/rel).exists(): (repo/rel).unlink()
  raise
 print('[OK] installed Option-Critic v6 progressive 8-slot patch'); print('[OK] backup:',backup); print('[OK] config:',target)
if __name__=='__main__': main()
