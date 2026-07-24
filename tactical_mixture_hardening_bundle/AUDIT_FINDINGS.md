# Audit findings: Tactical Mixture Actor v1

## Observed run result

The current v1 run does not support continuing as-is:

- `val/micro_win_rate` declines from approximately 35.1% to 34.1% to 29.3% across the shown validations;
- all four `usage_*` curves hover near 0.25;
- `usage_max` mostly remains between about 0.25 and 0.31;
- `train/tar` is noisy without a held-out improvement.

The usage graph is not proof of four learned tactics. In v1, `usage_i` is the weighted marginal selector probability, while the KL-to-uniform auxiliary explicitly pushes those values toward 0.25. A selector that outputs nearly `[0.25, 0.25, 0.25, 0.25]` for every state produces exactly that graph while learning no state-dependent tactical decision.

## Critical and major implementation issues

### 1. Best tactical checkpoint omitted architecture metadata — critical

The original integration added tactical metadata to periodic `latest.pt` payloads but did not modify `ValidationTrainer`'s `best_val_macro_winrate.pt` payload. A best tactical checkpoint could therefore contain tactical state-dict keys without the metadata required by strict tactical resume.

**Fix:** validation best checkpoints now include `tactical_mixture_metadata`. The loader also accepts an older metadata-less v1 best checkpoint only when tactical keys load shape-strictly and no unrelated key is missing or unexpected.

### 2. Uniform replay selected an API-incompatible buffer — observed crash

The original tactical configuration correctly selected the ordinary `Buffer`, but the unified-priority runner still unconditionally invoked `replay_buffer.set_env_step()`, an `AdaptiveBuffer` method.

**Fix:** every runner-side `set_env_step` call is guarded; the source audit checks all occurrences.

### 3. Diversity auxiliary could update the inherited actor — major

The original pairwise JS effect objective consumed live `base_policy_logits`. Its gradient could change the inherited actor merely to make tactic-conditioned policies look different.

**Fix:** feature tensors, base logits, masks and weights are detached for the effect auxiliary. The auxiliary can train only the tactical residual branch.

### 4. KL-to-uniform makes the W&B usage graph misleading — major

The old `balance_loss` penalized any non-uniform marginal tactic usage. This can prevent useful specialization and explains why all four usage panels remain near 0.25.

**Fix:** the loss is now a collapse-only hinge. It activates only if one tactic exceeds 80% marginal use or effective tactic count falls below 2. It does not reward exact uniformity.

### 5. Marginal usage did not measure learned tactics — major

`usage_0...3` says how much probability each index receives on average. It cannot distinguish:

- uniform probabilities at every state;
- confident, state-specific choices that balance globally.

**Fix:** new metrics include conditional entropy, marginal entropy, mutual information, normalized mutual information, sampled usage, deterministic argmax usage and selector-logit spread.

### 6. Exact-zero residual initialization is a symmetry fixed point — major

With all tactic residual outputs exactly identical, pairwise JS divergence is zero and its first derivative at equality is also zero. Primitive policy-gradient noise may eventually break the symmetry, but the auxiliary itself cannot reliably do so.

**Fix:** the residual output layer receives a bounded `1e-2 / sqrt(hidden_dim)` symmetry-breaking initialization. In a representative 384-feature/160-logit test, the initial residual standard deviation was around 0.005 and maximum absolute shift around 0.026—small enough to preserve the inherited policy while avoiding exact symmetry.

### 7. Deterministic evaluation arbitrarily selected tactic 0 — major

A uniform categorical selector's `argmax` resolves to index 0. The original validation path therefore applied tactic 0 even when the selector had learned no preference.

**Fix:** deterministic evaluation now uses a smooth confidence gate. At uniform confidence (`1/K`), the inherited base logits are used exactly. The tactical residual is blended in continuously and reaches full strength only when selector confidence reaches 0.55.

### 8. The inherited policy was not isolated — major experimental confound

The v1 run continued to train both the base actor and the roughly 19.6M-parameter JEPA feature adapter. A validation decline could therefore come from inherited-policy drift rather than the shared latent itself.

**Fix:** the recommended hardened configuration freezes both the inherited base actor and JEPA feature adapter. The tactic selector/residual and downstream critic/reward/continuation/mask heads remain trainable. This makes the experiment a clean residual extension of the adaptive checkpoint.

### 9. Residual branch had no hard output bound — major

The residual could eventually replace rather than modify the inherited actor.

**Fix:** each residual logit is hard-bounded to `[-4, 4]` with `4*tanh(raw/4)`. A second guard penalizes residual/base RMS ratio above 1.0.

### 10. Effect-statistics implementation was unfriendly to `torch.compile` — major

The v1-style diagnostic path used dynamic `torch.nonzero`, data-dependent shape selection and Tensor-to-Python branches. These can cause graph breaks or compilation failures inside `_cal_grad_jepa`.

**Fix:** the hardened effect path uses a deterministic bounded prefix of randomized replay states, zero weights for invalid states, and branch-free empty-mask repair. It passes a `torch.compile(..., fullgraph=True)` test.

### 11. Empty predicted masks could produce fragile diagnostics — major

Softmax diagnostics over a row with no valid action can become meaningless.

**Fix:** diagnostics repair empty rows to NOOP and exclude inactive agents from weighted statistics. Production action masking remains unchanged.

### 12. Adaptive state was restored/saved even when adaptive priority was disabled — integration confusion

The tactical experiment uses uniform collection and replay, but the runner could still restore or emit adaptive state.

**Fix:** adaptive state is restored and included in checkpoints only when `_adaptive_any` is true. Source adaptive state is explicitly skipped otherwise.

### 13. JEPA adapter metric name became false after freezing — diagnostic bug

The old metric counted all adapter parameters under `trainable_adapter_parameter_count`, even if `requires_grad=False`.

**Fix:** W&B now receives both `jepa/adapter_total_parameter_count` and a true `jepa/trainable_adapter_parameter_count` filtered by `requires_grad`.

### 14. Legacy metadata could make a non-tactical checkpoint unloadable — checkpoint compatibility

After tactical methods exist in the class, a tactical-disabled checkpoint may carry metadata with `architecture=legacy`. The old loader treated any non-null metadata as a tactical checkpoint and rejected it.

**Fix:** disabled/legacy metadata is routed through the allowlisted legacy migration path. Metadata that declares legacy while containing tactical parameter keys is rejected as inconsistent.

### 15. Source-checkpoint selection was vulnerable to stale shell variables — launch safety

A generic exported `CHECKPOINT` could silently point to the original Exp40 checkpoint or the declining tactical run.

**Fix:** the launch script ignores generic `CHECKPOINT`. Only `SOURCE_CHECKPOINT` can override the adaptive run's best checkpoint, and the resolved file must be named `best_val_macro_winrate.pt` and live directly inside `CURRENT_UNIFIED_PRIORITY_RUN.txt`'s directory.

### 16. Backup/restore was incomplete for repeated installations — operational safety

Earlier restore logic knew only a fixed list of files and could overwrite an existing hardening script without preserving it.

**Fix:** the installer records every backed-up and introduced file in `hardening_backup_manifest.json`; restoration is driven by that manifest.

### 17. Duplicate JEPA adapter print was not duplicate optimization

The previous startup log printed the adapter once during JEPA construction and again while listing optimizer modules. The displayed optimizer parameter total matched a single adapter copy, so that log did not establish duplicate registration.

**Fix:** hardening still adds a runtime identity check: no optimizer parameter ID may appear twice, and every tactical parameter must appear exactly once.

### 18. Configuration validation occurred after source replacement — operational risk

The first hardening draft generated the output YAML only after replacing source files. A malformed or unresolved source config could therefore leave a partial installation.

**Fix:** the complete hardened config is now parsed, transformed and interpolation-resolved during dry-run and before any backup/write operation. The source config must live inside the target repo and match JEPA, dense-v3, horizon 5 and duration 1.

### 19. Replay paths could be accidentally reused — critical operational risk

TorchRL memmap state from an older run can be incompatible with the current replay schema. A stale absolute `buffer.scratch_dir` or an explicitly reused non-empty `RUN_DIR` could reopen old replay files.

**Fix:** the generated config forces `buffer.scratch_dir: replay`, which the runner resolves underneath the new log directory. The launcher refuses every non-empty run directory.

### 20. Mid-install filesystem failure could leave a partial patch — operational risk

A backup alone does not prevent a process from failing after only some files have been written.

**Fix:** all output is prevalidated, and the write phase is wrapped in a best-effort transaction. On failure, backed-up files are restored and newly introduced files are removed automatically; the backup remains for manual verification.

### 21. Checkpoint metadata path could disagree with checkpoint lineage — source-lineage risk

It was possible to supply the adaptive checkpoint but a `run_meta.json` from another run.

**Fix:** the launcher now requires both files to live directly inside the selected adaptive-run directory, and the audit compares critical configuration and JEPA checkpoint hashes.

## Experimental boundary retained

This bundle deliberately keeps adaptive map priority and sequence PER disabled. The new experiment answers one isolated question:

> Starting from the adaptive-PER run's best learned weights, does a hardened shared tactical latent improve a fresh uniform-replay training phase?

Running adaptive priority and tactical mixture together should be a later, separate experiment because combining them now would prevent attribution.

## What remains unproven

No static review can guarantee a better policy. The bundle can establish that:

- the mechanism is wired consistently;
- source and checkpoint contracts are enforced;
- tactic metrics mean what their names claim;
- the inherited policy is preserved and isolated;
- the action mask and world-model action dimensions remain unchanged;
- checkpoints can be resumed safely.

Only held-out validation can establish whether the method improves control performance.
