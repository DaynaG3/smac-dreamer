#!/usr/bin/env python3
"""Inspect tactical metrics without imposing a long smoke test."""

from __future__ import annotations

import argparse
import json
import math
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tail", type=int, default=200)
    args = parser.parse_args()

    path = pathlib.Path(args.run_dir) / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"[FAIL] missing metrics file: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(key.startswith("train/tactic/") for key in row):
                rows.append(row)
    rows = rows[-args.tail :]
    if not rows:
        raise SystemExit("[FAIL] no train/tactic/* metrics found yet")

    keys = sorted({key for row in rows for key in row if key.startswith("train/tactic/")})
    latest = rows[-1]
    print(f"[OK] tactical metric rows: {len(rows)}")
    print(f"[OK] latest global_step: {latest.get('global_step', latest.get('step'))}")
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if len(finite) != len(values):
            raise SystemExit(f"[FAIL] non-finite values in {key}")
        print(f"{key}: latest={finite[-1]:.6g} mean={sum(finite)/len(finite):.6g}")


if __name__ == "__main__":
    main()
