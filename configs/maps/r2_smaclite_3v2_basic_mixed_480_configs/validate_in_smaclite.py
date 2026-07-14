#!/usr/bin/env python3
"""Dynamic SMAClite validation for the 3v2 basic mixed-unit map set.

Run from your smac-dreamer repository root after copying this folder in:

    PYTHONPATH=src:external/r2dreamer:external/smaclite \
      python configs/maps/r2_3v2_basic_mixed/validate_in_smaclite.py \
      --root configs/maps/r2_3v2_basic_mixed --episodes 5 --max-steps 180

The script checks that each map instantiates and runs a deterministic scripted
focus-fire policy. This is a smoke test, not a trained-policy benchmark.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from smaclite.env.util.direction import Direction


def choose_actions(env):
    avail = env.get_avail_actions()
    actions = []
    # Prefer focusing the lowest HP+shield attackable enemy. Otherwise move toward nearest enemy.
    enemy_order = sorted(env.enemies.values(), key=lambda e: (getattr(e, 'hp', 0) + getattr(e, 'shield', 0), e.id_in_faction))
    for i in range(env.n_agents):
        valid = np.flatnonzero(avail[i])
        if i not in env.agents:
            actions.append(0)
            continue
        unit = env.agents[i]
        attack_choices = [a for a in valid if a >= 6]
        if attack_choices:
            chosen = None
            for enemy in enemy_order:
                a = 6 + enemy.id_in_faction
                if a in attack_choices:
                    chosen = a
                    break
            actions.append(int(chosen if chosen is not None else min(attack_choices)))
            continue
        if env.enemies:
            nearest = min(env.enemies.values(), key=lambda e: float(np.linalg.norm(unit.pos - e.pos)))
            prefs = []
            for a in [2, 3, 4, 5]:
                if a not in valid:
                    continue
                dest = unit.pos + Direction(a - 2).dx_dy * 2
                prefs.append((float(np.linalg.norm(dest - nearest.pos)), a))
            if prefs:
                actions.append(int(min(prefs)[1]))
                continue
        actions.append(1 if 1 in valid else int(valid[0]))
    return actions


def run_one(path, seed, max_steps):
    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=str(path), seed=seed)
    obs, info = env.reset(seed=seed)
    assert np.isfinite(np.asarray(obs)).all(), path
    assert np.isfinite(env.get_state()).all(), path
    total = 0.0
    won = False
    done = False
    truncated = False
    for t in range(max_steps):
        obs, reward, done, truncated, info = env.step(choose_actions(env))
        assert np.isfinite(np.asarray(obs)).all(), path
        assert np.isfinite(float(reward)), path
        total += float(reward)
        if done or truncated:
            won = bool(info.get('battle_won', False))
            break
    env.close()
    return won, (not (done or truncated)), total, t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=180)
    args = p.parse_args()
    maps = sorted((args.root / 'configs').glob('*/*.json'))
    rows = []
    for idx, path in enumerate(maps, 1):
        wins = timeouts = 0
        returns = []
        lengths = []
        for ep in range(args.episodes):
            w, to, r, l = run_one(path, 2000 + ep, args.max_steps)
            wins += int(w)
            timeouts += int(to)
            returns.append(r)
            lengths.append(l)
        rows.append({
            'path': str(path),
            'win_rate': wins / args.episodes,
            'timeout_rate': timeouts / args.episodes,
            'mean_return': sum(returns) / len(returns),
            'mean_length': sum(lengths) / len(lengths),
        })
        print(f'[{idx:03d}/{len(maps)}] {path.name}: win={rows[-1]["win_rate"]:.2f} timeout={rows[-1]["timeout_rate"]:.2f}')
    out = args.root / 'dynamic_validation.csv'
    if rows:
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
