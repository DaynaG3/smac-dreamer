# Imagination-masking diagnosis: `train/imag_invalid_rate` rises to ~0.6–0.75

## Symptom
`train/imag_invalid_rate` falls early, then climbs to ~0.6–0.75, while **replay** mask
precision/recall stay high and validation win rate is flat.

## Exact cause
`imag_invalid_rate` is a **pre-mask** metric computed on actor logits that **receive no
gradient**. The masked distribution sets invalid-action logits to `-1e9` *before* softmax
([`masked_actions.MaskedMultiOneHotDist.__init__`](../../src/smacdreamer/masked_actions.py)), so
invalid actions get ~0 probability **and ~0 gradient**. Those logits are therefore never trained
and drift upward under shared-network pressure, so the **unmasked greedy** (`argmax` of the raw
logits) increasingly lands on invalid actions → the rate climbs. The **masked sampling never
picks them**, so the real signal — the masked invalid sample rate — stays 0.

Replay precision/recall stay high because they are measured on **posterior** states (the avail
head trivially reconstructs the avail that was an encoder input), whereas the imagination mask is
predicted from **open-loop prior** states, which is a different (and harder) regime.

## Affected file / function
- Metric: `src/smacdreamer/masked_actions.py::MaskedMultiOneHotDist.unmasked_invalid_rate`
- Site: `external/r2dreamer/dreamer.py::_cal_grad` (the `imag_invalid_rate` line)
- Mask source: `external/r2dreamer/dreamer.py::_predicted_action_mask` / `_imagine` (open-loop priors)

## Misleading metric or real masking bug?
**Misleading metric — not a masking bug**, *conditional on the invariant*:
```
imag_post_mask_invalid_sample_rate == 0
real_post_mask_invalid_sample_rate == 0
```
Both are now logged and asserted in tests. Sampling, log-prob, entropy and the policy loss all use
the **same masked distribution**; padded/dead slots are forced to NOOP and excluded from log-prob,
entropy and every metric (active-only normalisation). So masking itself is correct.

## Category
Primarily a **metric-definition artifact** (measuring pre-mask greedy on un-gradiented logits).
Possible secondary contributor: **long-horizon world-model drift** of the *prior* mask — the new
per-horizon diagnostic (`maskh{h}_*`) quantifies how avail-head precision/recall degrade from
posterior → 1-step prior → open-loop prior. It is **not** a tensor-alignment or padding/death bug
(verified: active-only, ordering-preserving; covered by unit tests).

The **flat validation win rate is a separate early/hard-task learning issue**; the rising rate does
not corrupt training (sampling stays masked-valid). It does indicate the actor is not forming
confident *valid* preferences.

## New diagnostics (additive only — no training-behaviour change)
- `imag_pre_mask_invalid_mass`, `imag_pre_mask_invalid_sample_rate` (= old `imag_invalid_rate`)
- `imag_post_mask_invalid_sample_rate` (**invariant = 0**), `imag_empty_mask_rate`
- `real_pre_mask_invalid_mass`, `real_post_mask_invalid_sample_rate` (**invariant = 0**)
- per-horizon `maskh{h}_{posterior|prior1|openloop}_{precision,recall,fpr}`

## Recommended fix
- **Reinterpret the metric, don't tune anything.** Treat `imag_pre_mask_invalid_*` as a *drift of
  unconstrained invalid logits* indicator, and watch `imag_post_mask_invalid_sample_rate == 0` as
  the correctness signal. Optionally retire `imag_invalid_rate` in dashboards in favour of the
  explicit pre/post names.
- If long-horizon `maskh{h}_openloop_*` precision/recall fall sharply, the *prior* mask is
  drifting; the eventual mitigation (separate change, not now) is to raise the avail-head loss
  weight or shorten the imagination horizon — **left for a follow-up**, per the no-tuning constraint.
- The flat win rate is addressed by the reward/exploration ablation, not by masking.

## Short test-run commands
```bash
# unit tests (pure torch; verifies the post-mask == 0 invariant + helpers)
python -m pytest tests/test_masked_actions.py -v

# masked structured smoke; confirm the new metrics in logs/W&B
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650.yaml --steps 2000
# expect: imag_post_mask_invalid_sample_rate == 0 and real_post_mask_invalid_sample_rate == 0
#         throughout; imag_pre_mask_invalid_* may rise (benign); compare maskh0_posterior_* (high)
#         vs maskh{>=1}_*_recall to see prior-mask drift with horizon.
```
