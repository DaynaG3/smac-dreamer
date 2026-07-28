"""Optional runtime override of SMAClite's AGENT_SIGHT_RANGE for full-observability ablations.

SMAClite gates unit visibility with the module constant ``AGENT_SIGHT_RANGE`` (default 9) and ALSO
uses it as the distance/dx/dy normalization divisor. Enlarging it (e.g. to 24 on r2_2100, whose
maps span <=~21 units) makes every unit see the whole map -- at the cost of rescaling the distance
features (an accepted coupling; we do NOT edit external/smaclite to decouple them).

Propagation: the trainer sets the ``SMACLITE_SIGHT_RANGE`` env var once in the parent process from
``observation.sight_range``. Spawn children (train workers, validation env children, discovery
probes) inherit ``os.environ``; each calls :func:`maybe_override_sight_range` before its first
SMAClite env is built (from ``r2dreamer_factory._ensure_paths`` and ``map_discovery.validate_map``),
so the override is applied uniformly across training, validation, and discovery.

Absent/empty env var -> no-op (behaviour identical to today). SMAClite is imported lazily so this
module stays importable without smaclite on ``sys.path``.
"""

import os

ENV_VAR = "SMACLITE_SIGHT_RANGE"

# Guards a single log line per process (the helper may be called many times per worker).
_applied = False


def maybe_override_sight_range():
    """Override ``smaclite.env.smaclite.AGENT_SIGHT_RANGE`` from ``SMACLITE_SIGHT_RANGE``.

    Returns the applied integer sight range, or ``None`` when the env var is absent/empty (in which
    case SMAClite is left untouched). Idempotent; logs the applied value once per process.
    """
    global _applied
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    value = int(raw)
    from smaclite.env import smaclite as _smaclite  # lazy: path set up by _ensure_paths first
    _smaclite.AGENT_SIGHT_RANGE = value
    if not _applied:
        _applied = True
        print(f"[sight_range] applied AGENT_SIGHT_RANGE={value} (pid={os.getpid()})", flush=True)
    return value


def configure_train_sight_range(cfg):
    """Training-side sight-range setup from ``cfg.observation.sight_range``.

    - Present (full-visibility ablation): export ``SMACLITE_SIGHT_RANGE`` and return the int.
    - Absent: **pop** any stale ``SMACLITE_SIGHT_RANGE`` so a leaked shell export cannot silently
      turn a normal partial-observation run into full visibility; return None.

    MUST be called before any env is built (its value is inherited by spawn children). Works with
    dict or OmegaConf ``cfg``.
    """
    obs = cfg.get("observation") if cfg is not None else None
    sight_range = obs.get("sight_range") if obs else None
    if sight_range is not None:
        sight_range = int(sight_range)
        os.environ[ENV_VAR] = str(sight_range)
        print(f"  [observation] full-visibility override: AGENT_SIGHT_RANGE -> {sight_range}",
              flush=True)
    else:
        os.environ.pop(ENV_VAR, None)
        print("  [observation] default SMAClite visibility: AGENT_SIGHT_RANGE=9", flush=True)
    return sight_range


def resolve_and_export_sight_range(run_meta, cfg):
    """Reconstruct the training sight range for standalone eval and export the env var.

    Priority: ``run_meta['sight_range']`` (if present and non-null) > ``cfg.observation.sight_range``
    > ``None`` (default partial observability). When the resolved value is not None, exports
    ``SMACLITE_SIGHT_RANGE`` so every SMAClite env built afterwards (in-process probe, spawned
    discovery / eval children -- all of which call :func:`maybe_override_sight_range`) uses the same
    visibility the checkpoint was trained with. MUST be called BEFORE any env is constructed.
    Returns the resolved int, or None. Works with dict or OmegaConf ``run_meta`` / ``cfg``.
    """
    sight_range = run_meta.get("sight_range") if run_meta else None
    obs = cfg.get("observation") if cfg is not None else None
    if sight_range is None and obs:
        sight_range = obs.get("sight_range")       # .get works on dict and OmegaConf alike
    sight_range = int(sight_range) if sight_range is not None else None
    if sight_range is not None:
        os.environ[ENV_VAR] = str(sight_range)
        print(f"Reconstruction: sight_range={sight_range} -> SMACLITE_SIGHT_RANGE", flush=True)
    else:
        # Clear any stale export from the shell so a partial-obs eval is never contaminated.
        os.environ.pop(ENV_VAR, None)
        print("Reconstruction: sight_range=None -> default SMAClite partial observability", flush=True)
    return sight_range
