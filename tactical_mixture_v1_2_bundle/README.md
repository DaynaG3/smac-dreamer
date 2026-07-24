# Tactical Mixture v1.2 — Centered Trust-Region Patch

This patch is designed for a repository that already has Tactical Mixture v1.1 hardening installed.

## Why v1.1 is being replaced

The running v1.1 diagnostics showed:

- deterministic argmax tactic usage was essentially 100% tactic 1;
- sampled tactic usage looked diverse only because the selector remained stochastic;
- pairwise tactic policy JS divergence stayed around 1e-5;
- all per-tactic residual RMS values were effectively identical;
- the common residual reached roughly half the inherited actor-logit RMS;
- held-out macro win rate fell from 35.5% to 32.5% and then 31.83%.

This means v1.1 learned one large common residual actor, not distinct tactics.

## v1.2 changes

1. **Two tactics instead of four** for a cleaner first latent decomposition.
2. **Zero-mean residual across tactics** at every state. Any residual common to all tactics is mathematically removed.
3. **Residual scale 0.25** and raw residual cap 2.0.
4. **No deterministic tactical effect below 0.70 selector confidence.** The inherited policy is used exactly until the selector is confident.
5. **Selector mutual-information floor** to discourage state-invariant tactic selection.
6. **Masked KL trust region to the inherited actor**, target 0.02 nats per alive agent.
7. **Action-flip and KL metrics** for direct policy-deviation diagnosis.
8. Base actor and JEPA adapter remain frozen; adaptive priority remains disabled.

## Install

```bash
export ROOT=/home/jovyan/workspace/dreamer/combined-upload
export REPO="$ROOT/smac-dreamer"
export PY="$ROOT/.venv/bin/python"
export BUNDLE="$ROOT/tactical_mixture_v1_2_bundle"

"$PY" "$BUNDLE/install_tactical_v1_2.py" \
  --repo "$REPO" \
  --dry-run

"$PY" "$BUNDLE/install_tactical_v1_2.py" \
  --repo "$REPO"
```

## Audit

```bash
export ADAPTIVE_RUN="$(cat "$ROOT/CURRENT_UNIFIED_PRIORITY_RUN.txt")"
export SOURCE_CHECKPOINT="$ADAPTIVE_RUN/best_val_macro_winrate.pt"
export SOURCE_RUN_META="$ADAPTIVE_RUN/run_meta.json"

cd "$REPO"
REPO="$REPO" \
PY="$PY" \
CONFIG=configs/r2_2100_jepa_tactical_mixture_v1_2.yaml \
CHECKPOINT="$SOURCE_CHECKPOINT" \
SOURCE_RUN_META="$SOURCE_RUN_META" \
  bash scripts/static_audit_tactical_v1_2.sh
```

## Launch

```bash
ROOT="$ROOT" \
REPO="$REPO" \
PY="$PY" \
ADAPTIVE_RUN="$ADAPTIVE_RUN" \
FINAL_STEP=2000000 \
  bash "$REPO/scripts/run_tactical_v1_2_2m.sh"
```

## Safety metrics

Watch:

- `train/tactic/base_kl_mean`
- `train/tactic/base_kl_max`
- `train/tactic/action_flip_rate`
- `train/tactic/residual_to_base_ratio`
- `train/tactic/mutual_information_normalized`
- `train/tactic/effect_js`
- `train/tactic/argmax_usage_0`, `argmax_usage_1`
- `val/macro_win_rate`

The deterministic validation policy remains exactly the source policy while selector confidence is below 0.70.
