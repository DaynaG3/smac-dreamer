"""Batch-visualise R2-Dreamer × SMAClite episodes and select interesting cases.

Two-pass design:
  1. Evaluate every (map × seed) deterministically WITHOUT capturing frames (cheap), collect
     per-episode summaries.
  2. Select episodes by ``--select`` (wins / failures / low_enemy_damage / poor_target_focus /
     all), then rerun ONLY the selected episodes (same seed) capturing frames, and write
     MP4 + JSONL + summary for each.
A ``batch_summary.json`` index records all evaluated episodes, the selection, and output paths.

Structured checkpoints only. Records EXECUTED actions (from the env debug context).

Usage (smac-r2 conda env, project root):
    python scripts/visualize_batch.py \
        --config configs/r2_650.yaml \
        --checkpoint logs/r2dreamer/r2_650/best_val_macro_winrate.pt \
        --split blind_compositional --seeds 0,1,2 --select failures \
        --max-videos 10 --headless --output-dir results/replays/blind_comp_failures
"""

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "external" / "r2dreamer"),
           str(ROOT / "external" / "smaclite")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch-visualise R2-Dreamer × SMAClite episodes (structured checkpoints only)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-meta", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="validation",
                    help="config map split: validation | blind_iid | blind_compositional")
    ap.add_argument("--maps-dir", default=None, help="arbitrary maps folder (overrides --split)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (default: cfg eval/validation seeds, else 0)")
    ap.add_argument("--select", choices=["wins", "failures", "low_enemy_damage",
                                         "poor_target_focus", "all"], default="failures")
    ap.add_argument("--max-videos", type=int, default=10)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--output-dir", default="results/replays")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="playback fps (lower = easier to follow; 22.4 = realtime)")
    ap.add_argument("--scale", type=int, default=2,
                    help="nearest-neighbour upscale factor for readability (default: 2)")
    ap.add_argument("--hold-last-seconds", type=float, default=1.5,
                    help="freeze the final frame this long so the WIN/LOSS outcome is readable")
    ap.add_argument("--max-episode-steps", type=int, default=None)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--low-enemy-ehp-threshold", type=float, default=0.75)
    ap.add_argument("--poor-focus-threshold", type=float, default=0.5)
    ap.add_argument("--min-attack-steps-for-focus", type=int, default=5)
    return ap


def _selected(summary: dict, select: str) -> bool:
    won = bool(summary.get("battle_won", False))
    if select == "wins":
        return won
    if select == "failures":
        return not won
    if select == "low_enemy_damage":
        return bool(summary.get("low_enemy_damage", False))
    if select == "poor_target_focus":
        return bool(summary.get("poor_target_focus", False))
    if select == "all":
        return True
    return False


def main():
    args = build_parser().parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smacdreamer.visualization import rollout, render

    cfg = rollout.load_config(args.config)
    entries = rollout.resolve_map_entries(cfg, split=args.split, maps_dir=args.maps_dir)
    seeds = rollout.resolve_seeds(cfg, args.seeds)
    print(f"[visualize-batch] {len(entries)} maps × {len(seeds)} seeds "
          f"({len(entries) * len(seeds)} episodes); select={args.select}")

    ctx = rollout.load_agent(
        args.config, args.checkpoint, run_meta=args.run_meta, device=args.device,
        max_episode_steps=args.max_episode_steps, probe_entry=entries[0])

    cls_kwargs = dict(
        low_enemy_ehp_threshold=args.low_enemy_ehp_threshold,
        poor_focus_threshold=args.poor_focus_threshold,
        min_attack_steps_for_focus=args.min_attack_steps_for_focus,
    )

    # ---- Pass 1: evaluate every (map, seed), summaries only (no frames) ------------------
    evaluated = []
    for entry in entries:
        env = rollout.make_single_map_env(ctx, entry, capture=False)
        try:
            for seed in seeds:
                res = rollout.run_episode(ctx, env, map_name=entry.name, seed=seed,
                                          capture_frames=False, overlay=False, **cls_kwargs)
                evaluated.append(res.summary)
                print(f"  {entry.name:<28} seed={seed} "
                      f"{'WIN ' if res.summary['battle_won'] else 'LOSS'} "
                      f"len={res.summary['episode_length']} "
                      f"enemy_ehp={res.summary['final_enemy_ehp_frac']} "
                      f"focus={res.summary['mean_target_focus_score']} "
                      f"low_dmg={res.summary['low_enemy_damage']} "
                      f"poor_focus={res.summary['poor_target_focus']}")
        finally:
            try:
                env.close()
            except Exception:
                pass

    # ---- Select episodes -----------------------------------------------------------------
    selected = [s for s in evaluated if _selected(s, args.select)][: args.max_videos]
    print(f"[visualize-batch] selected {len(selected)}/{len(evaluated)} episodes "
          f"(capped at --max-videos={args.max_videos})")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {e.name: e for e in entries}

    # ---- Pass 2: rerun selected episodes capturing frames; save MP4 + JSONL + summary ----
    selected_records = []
    for s in selected:
        entry = by_name[s["map"]]
        seed = int(s["seed"])
        env = rollout.make_single_map_env(ctx, entry, capture=True)
        try:
            res = rollout.run_episode(ctx, env, map_name=entry.name, seed=seed,
                                      capture_frames=True, overlay=not args.no_overlay,
                                      scale=args.scale, **cls_kwargs)
        finally:
            try:
                env.close()
            except Exception:
                pass

        stem = f"{entry.name}_seed{seed}"
        mp4_path = out_dir / f"{stem}.mp4"
        jsonl_path = out_dir / f"{stem}.jsonl"
        summary_path = out_dir / f"{stem}_summary.json"
        frames = render.with_hold(res.frames, args.fps, hold_last_seconds=args.hold_last_seconds)
        render.write_mp4(mp4_path, frames, fps=args.fps)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in res.records:
                f.write(json.dumps(rec) + "\n")
        summary_path.write_text(json.dumps(res.summary, indent=2), encoding="utf-8")
        selected_records.append({
            "map": entry.name, "seed": seed,
            "selection_reason": args.select,
            "battle_won": res.summary["battle_won"],
            "low_enemy_damage": res.summary["low_enemy_damage"],
            "poor_target_focus": res.summary["poor_target_focus"],
            "mp4": str(mp4_path), "jsonl": str(jsonl_path), "summary": str(summary_path),
        })
        print(f"  wrote {mp4_path}")

    # ---- Batch index ---------------------------------------------------------------------
    batch_summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "maps_dir": args.maps_dir,
        "seeds": seeds,
        "select": args.select,
        "max_videos": args.max_videos,
        "thresholds": cls_kwargs,
        "n_evaluated": len(evaluated),
        "n_selected": len(selected_records),
        "evaluated_episodes": evaluated,
        "selected_episodes": selected_records,
    }
    index_path = out_dir / "batch_summary.json"
    index_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"[visualize-batch] wrote {index_path} "
          f"({len(selected_records)} videos, {len(evaluated)} episodes evaluated)")


if __name__ == "__main__":
    main()
