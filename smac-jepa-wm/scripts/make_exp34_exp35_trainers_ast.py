#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

BASE = Path("smac_jepa/train_jepa_exp31_exp33.py")
WRAP33 = Path("smac_jepa/train_jepa_exp33_dreamer.py")

OUT = Path("smac_jepa/train_jepa_exp31_exp35.py")
WRAP34 = Path("smac_jepa/train_jepa_exp34_dreamer.py")
WRAP35 = Path("smac_jepa/train_jepa_exp35_dreamer.py")


def die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def parse_stmt(src: str) -> list[ast.stmt]:
    return ast.parse(src).body


def parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def is_assign_to(stmt: ast.stmt, name: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == name
    )


class Exp34Exp35Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.in_loss_fn = False
        self.counts = {
            "env_flags": 0,
            "slot_mask_insert": 0,
            "target_encoder_call_masks": 0,
            "target_normalize_masks": 0,
            "latent_mask": 0,
            "dynamic_reference_mask": 0,
            "target_valid": 0,
            "presence_loss": 0,
            "reg_masks": 0,
            "latent_valid": 0,
            "loss_dict": 0,
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != "markov_rollout_rnn_losses":
            return self.generic_visit(node)

        self.in_loss_fn = True
        node = self.generic_visit(node)
        self.in_loss_fn = False

        new_body: list[ast.stmt] = []
        inserted_env = False

        env_code = parse_stmt(
            '''
_exp_os = __import__("os")
exp34_two_mask_loss = _exp_os.environ.get("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "0") == "1"
exp35_simple_loss = _exp_os.environ.get("SMAC_JEPA_EXP35_SIMPLE_LOSS", "0") == "1"
presence_neg_class_weight = float(_exp_os.environ.get("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "1.0"))

if exp35_simple_loss:
    r2_rep_scale = 0.0
    r2_barlow_scale = 0.0
    memory_barlow_scale = 0.0
    delta_loss_weight = 0.0
    hidden_reconstruction_weight = 0.0
    last_seen_anchor_weight = 0.0
    hidden_presence_weight = 0.0
    reappearance_consistency_weight = 0.0
    inverse_dynamics_weight = 0.0
'''
        )

        for stmt in node.body:
            new_body.append(stmt)

            # Insert env flags right after the existing "del detach_rollout_targets"
            # if present. Otherwise insert after first statement.
            if not inserted_env:
                is_del_detach = (
                    isinstance(stmt, ast.Delete)
                    and any(isinstance(t, ast.Name) and t.id == "detach_rollout_targets" for t in stmt.targets)
                )
                if is_del_detach:
                    new_body.extend(env_code)
                    inserted_env = True
                    self.counts["env_flags"] += 1

            if is_assign_to(stmt, "slot_mask_seq"):
                new_body.extend(
                    parse_stmt(
                        '''
target_encoding_mask_seq = slot_mask_seq if exp34_two_mask_loss else target_entity_mask_seq_full
'''
                    )
                )
                self.counts["slot_mask_insert"] += 1

        if not inserted_env:
            node.body = env_code + node.body
            self.counts["env_flags"] += 1
        else:
            node.body = new_body

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        if not self.in_loss_fn:
            return node

        # model.encoder(target_entity_seq_full, target_entity_mask_seq_full)
        # target_encoder(target_entity_seq_full, target_entity_mask_seq_full)
        if (
            len(node.args) >= 2
            and is_name(node.args[0], "target_entity_seq_full")
            and is_name(node.args[1], "target_entity_mask_seq_full")
        ):
            func = node.func
            is_encoder_call = (
                (isinstance(func, ast.Attribute) and func.attr == "encoder")
                or (isinstance(func, ast.Name) and func.id == "target_encoder")
            )
            if is_encoder_call:
                node.args[1] = ast.Name(id="target_encoding_mask_seq", ctx=ast.Load())
                self.counts["target_encoder_call_masks"] += 1

        # r2_normalize_latent(x, target_entity_mask_seq_full, ...)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "r2_normalize_latent"
            and len(node.args) >= 2
            and is_name(node.args[1], "target_entity_mask_seq_full")
        ):
            node.args[1] = ast.Name(id="target_encoding_mask_seq", ctx=ast.Load())
            self.counts["target_normalize_masks"] += 1

        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)

        if not self.in_loss_fn:
            return node

        if is_assign_to(node, "latent_mask"):
            self.counts["latent_mask"] += 1
            return parse_stmt(
                '''
target_presence_valid = target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)
target_slot_valid = entity_slot_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)
latent_mask = target_slot_valid if exp34_two_mask_loss else target_presence_valid
'''
            )

        if is_assign_to(node, "dynamic_reference_mask"):
            self.counts["dynamic_reference_mask"] += 1
            return parse_stmt(
                '''
dynamic_reference_entity_valid = entity_slot_mask if exp34_two_mask_loss else target_entity_mask
dynamic_reference_mask = (
    dynamic_feature_valid[:, None, None].expand_as(target_entity)
    * dynamic_reference_entity_valid.unsqueeze(-1)
    * valid_mask.unsqueeze(-1).unsqueeze(-1)
)
'''
            )

        if is_assign_to(node, "target_valid"):
            self.counts["target_valid"] += 1
            return parse_stmt(
                '''
target_valid = target_slot_valid if exp34_two_mask_loss else target_presence_valid
'''
            )

        if is_assign_to(node, "presence_loss"):
            # Only replace the main weighted_bce presence loss.
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "weighted_bce":
                    self.counts["presence_loss"] += 1
                    return parse_stmt(
                        '''
if presence_neg_class_weight != 1.0:
    presence_raw_bce = F.binary_cross_entropy_with_logits(
        presence_logits,
        target_entity_mask,
        reduction="none",
    )
    presence_time_w = aux_weights.view(1, 1, -1, 1)
    presence_class_w = torch.where(
        target_entity_mask > 0.5,
        torch.ones_like(target_entity_mask),
        torch.full_like(target_entity_mask, presence_neg_class_weight),
    )
    presence_weighted_mask = presence_mask * presence_time_w * presence_class_w
    presence_loss = (
        presence_raw_bce * presence_weighted_mask
    ).sum() / presence_weighted_mask.sum().clamp_min(1.0)
else:
    presence_loss = weighted_bce(
        presence_logits,
        target_entity_mask,
        presence_mask,
        aux_weights,
    )
'''
                    )

        if is_assign_to(node, "reg_masks"):
            self.counts["reg_masks"] += 1
            return parse_stmt(
                '''
reg_masks = torch.cat(
    [observation_mask_seq, target_encoding_mask_seq],
    dim=1,
)
'''
            )

        if is_assign_to(node, "latent_valid"):
            self.counts["latent_valid"] += 1
            return parse_stmt(
                '''
latent_valid_entity = entity_slot_mask if exp34_two_mask_loss else target_entity_mask * entity_slot_mask
latent_valid = latent_valid_entity * valid_mask.unsqueeze(-1)
'''
            )

        if is_assign_to(node, "losses") and isinstance(node.value, ast.Dict):
            keys = node.value.keys
            vals = node.value.values
            insert_at = 1 if keys else 0

            keys[insert_at:insert_at] = [
                ast.Constant(value="exp34_two_mask_loss_enabled"),
                ast.Constant(value="exp35_simple_loss_enabled"),
                ast.Constant(value="presence_neg_class_weight_value"),
            ]
            vals[insert_at:insert_at] = [
                parse_expr('torch.tensor(float(exp34_two_mask_loss), device=pred_latent.device)'),
                parse_expr('torch.tensor(float(exp35_simple_loss), device=pred_latent.device)'),
                parse_expr('torch.tensor(float(presence_neg_class_weight), device=pred_latent.device)'),
            ]
            self.counts["loss_dict"] += 1
            return node

        return node


def make_wrapper(src: str, *, exp35: bool) -> str:
    out = src.replace("train_jepa_exp31_exp33", "train_jepa_exp31_exp35")

    env_lines = [
        '    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")',
        '    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")',
    ]
    if exp35:
        env_lines.insert(1, '    os.environ.setdefault("SMAC_JEPA_EXP35_SIMPLE_LOSS", "1")')

    pattern = r"def main\(\)\s*->\s*None:\s*"
    repl = "def main() -> None:\n" + "\n".join(env_lines) + "\n    "
    out, n = re.subn(pattern, repl, out, count=1)
    if n != 1:
        die("Could not patch def main() in train_jepa_exp33_dreamer.py wrapper.")
    return out


def main() -> None:
    if not BASE.exists():
        die(f"Missing {BASE}. Run this from smac-jepa-wm repo root.")
    if not WRAP33.exists():
        die(f"Missing {WRAP33}. Run this from smac-jepa-wm repo root.")

    src = BASE.read_text()
    tree = ast.parse(src, filename=str(BASE))

    transformer = Exp34Exp35Transformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    required_min = {
        "env_flags": 1,
        "slot_mask_insert": 1,
        "target_encoder_call_masks": 2,
        "target_normalize_masks": 2,
        "latent_mask": 1,
        "dynamic_reference_mask": 1,
        "target_valid": 1,
        "presence_loss": 1,
        "reg_masks": 1,
        "loss_dict": 1,
    }

    print("[patch-counts]")
    for k, v in transformer.counts.items():
        print(f"  {k}: {v}")

    bad = [
        f"{k}={transformer.counts[k]} < {minimum}"
        for k, minimum in required_min.items()
        if transformer.counts[k] < minimum
    ]
    if bad:
        die("AST patch incomplete: " + ", ".join(bad))

    generated = ast.unparse(new_tree) + "\n"
    OUT.write_text(generated)
    print(f"[ok] wrote {OUT}")

    wrap_src = WRAP33.read_text()
    WRAP34.write_text(make_wrapper(wrap_src, exp35=False))
    WRAP35.write_text(make_wrapper(wrap_src, exp35=True))
    print(f"[ok] wrote {WRAP34}")
    print(f"[ok] wrote {WRAP35}")

    # Compile immediately so you do not have to guess.
    for path in [OUT, WRAP34, WRAP35]:
        code = path.read_text()
        compile(code, str(path), "exec")
        print(f"[ok] compiled {path}")

    print("[done] Exp34/Exp35 files generated successfully.")


if __name__ == "__main__":
    main()
