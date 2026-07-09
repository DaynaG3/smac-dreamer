"""Structured-observation policy visualisation for R2-Dreamer × SMAClite.

Additive, read-only tooling: load a trained checkpoint, replay deterministic SMAClite
episodes with the existing simulator + renderer, and record what *executed* actions the
agent takes (not requested logits). See ``docs/policy_visualisation.md``.

Supports ``structured`` observation checkpoints only — flat-obs checkpoints fail fast.

Module split keeps the pure-Python helpers importable without torch/pygame:
  * ``trace``   — action labels, target-focus metric, episode summary/classification.
  * ``render``  — pygame frame capture, overlay, MP4 writing (imports pygame/imageio lazily).
  * ``rollout`` — checkpoint/agent reconstruction + episode driver (imports torch + scripts).
"""
