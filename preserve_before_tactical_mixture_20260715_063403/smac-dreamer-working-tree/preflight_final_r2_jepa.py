from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import torch


def repo_path(rel: str) -> Path:
    p = Path(rel)
    if p.exists():
        return p
    alt = Path("smac-dreamer") / rel
    if alt.exists():
        return alt
    raise FileNotFoundError(rel)


def check_static() -> None:
    wm_path = repo_path("src/smacdreamer/jepa/world_model.py")
    fa_path = repo_path("src/smacdreamer/jepa/feature_adapter.py")
    wm = wm_path.read_text()
    fa = fa_path.read_text()

    required_wm = [
        "def _seen_mask_from_memory",
        "def _belief_mask",
        "max_agents=self.max_agents",
        "CRITICAL: feature_adapter is trainable",
    ]
    for token in required_wm:
        if token not in wm:
            raise SystemExit(f"BAD: world_model.py missing token: {token}")
    print("OK: world_model has belief-mask + max_agents patch tokens")

    required_fa = [
        "slot_mlp",
        "ally slots + enemy summary + global summary",
        "allies_flat",
        "enemy_summary",
        "global_summary",
    ]
    for token in required_fa:
        if token not in fa:
            raise SystemExit(f"BAD: feature_adapter.py missing token: {token}")
    print("OK: feature_adapter has slot-preserving path tokens")

    # AST check: in get_feat, feature_adapter call must not be inside a With node.
    tree = ast.parse(wm)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FrozenJEPAWorldModel")
    get_feat = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_feat")

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.with_depth = 0
            self.bad = False
            self.calls = 0
        def visit_With(self, node):
            self.with_depth += 1
            self.generic_visit(node)
            self.with_depth -= 1
        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "feature_adapter":
                self.calls += 1
                if self.with_depth > 0:
                    self.bad = True
            self.generic_visit(node)

    v = Visitor()
    v.visit(get_feat)
    if v.calls != 1:
        raise SystemExit(f"BAD: expected one feature_adapter call in get_feat, found {v.calls}")
    if v.bad:
        raise SystemExit("BAD: feature_adapter call is still inside a with/no_grad block")
    print("OK: feature_adapter call is outside no_grad in get_feat")


def check_adapter_grad() -> None:
    import sys
    root = Path.cwd()
    for p in [root / "src", root / "smac-dreamer" / "src"]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from smacdreamer.jepa.feature_adapter import JEPAFeatureAdapter

    torch.manual_seed(0)
    b, e, latent, memory, static, out, max_agents = 4, 19, 32, 65, 12, 256, 9
    adapter = JEPAFeatureAdapter(
        latent_dim=latent,
        memory_dim=memory,
        static_dim=static,
        out_dim=out,
        max_agents=max_agents,
    )
    z = torch.randn(b, e, latent)
    mem = torch.randn(b, e, memory)
    mask = torch.ones(b, e)
    stat = torch.randn(b, static)
    feat = adapter(z, mem, mask, stat)
    loss = feat.pow(2).mean()
    loss.backward()
    grad_sum = sum(
        float(p.grad.abs().sum().item()) for p in adapter.parameters() if p.grad is not None
    )
    if grad_sum <= 0:
        raise SystemExit("BAD: adapter gradient sum is zero")
    if getattr(adapter, "max_agents", None) != max_agents:
        raise SystemExit("BAD: adapter max_agents not set")
    print(f"OK: feature adapter receives gradients; grad_sum={grad_sum:.6g}")


def check_checkpoint(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"BAD: checkpoint missing: {p}")
    ckpt = torch.load(p, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("BAD: checkpoint is not a dict")
    keys = set(ckpt.keys())
    must = {"model_state", "memory_module_state", "metadata"}
    missing = sorted(must - keys)
    if missing:
        raise SystemExit(f"BAD: checkpoint missing JEPA keys: {missing}")
    meta = dict(ckpt.get("metadata", {}))
    resolved = dict(ckpt.get("resolved_config", ckpt.get("config", {})))
    print("OK: JEPA checkpoint looks valid")
    print("  size_MB:", round(p.stat().st_size / 1024 / 1024, 2))
    print("  max_agents:", meta.get("max_agents"))
    print("  max_enemies:", meta.get("max_enemies"))
    print("  max_actions:", meta.get("max_actions"))
    print("  anchored_belief_memory:", resolved.get("anchored_belief_memory"))
    print("  action_conditioned_memory:", resolved.get("action_conditioned_memory"))
    print("  presence_rollout_mode:", resolved.get("presence_rollout_mode"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa-checkpoint", default="checkpoints/jepa/model.pt")
    args = ap.parse_args()
    check_static()
    check_adapter_grad()
    check_checkpoint(args.jepa_checkpoint)
    print("PASS: final R2-JEPA preflight passed. Launching 2M is reasonable without a 50k smoke.")


if __name__ == "__main__":
    main()
