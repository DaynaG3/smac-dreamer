"""Visualise ONE R2-Dreamer × SMAClite episode (structured checkpoints only).

Runs a single (map, seed) deterministically with a trained checkpoint, optionally displays
it live and/or records an MP4, and always writes per-step JSONL + an episode summary.

The trace records the agent's EXECUTED actions (read from the env debug context), not the raw
policy logits / top-k probabilities.

Usage (smac-r2 conda env, project root):
    python scripts/visualize_episode.py \
        --config configs/r2_650.yaml \
        --checkpoint logs/r2dreamer/r2_650/best_val_macro_winrate.pt \
        --split blind_iid --map-name <map_name> --seed 0 \
        --mode both --output-dir results/replays

Headless (Kubeflow/Kaggle):
    python scripts/visualize_episode.py --config ... --checkpoint ... \
        --map-name <map> --mode record --headless
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
        description="Visualise one R2-Dreamer × SMAClite episode (structured checkpoints only)")
    ap.add_argument("--config", required=True, help="training YAML config")
    ap.add_argument("--checkpoint", required=True, help="path to a checkpoint .pt (any folder)")
    ap.add_argument("--run-meta", default=None,
                    help="run_meta.json path (default: beside the checkpoint)")
    ap.add_argument("--device", default=None, help="torch device (default: from config)")
    ap.add_argument("--split", default="validation",
                    help="config map split: validation | blind_iid | blind_compositional")
    ap.add_argument("--maps-dir", default=None,
                    help="arbitrary maps folder (overrides --split)")
    ap.add_argument("--map-name", default=None,
                    help="map to run (required unless --maps-dir/split has exactly one map)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["interactive", "record", "both"], default="both")
    ap.add_argument("--headless", action="store_true",
                    help="set SDL dummy video/audio drivers; disables the interactive window")
    ap.add_argument("--output-dir", default="results/replays")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="playback fps (lower = easier to follow; 22.4 = realtime)")
    ap.add_argument("--scale", type=int, default=2,
                    help="nearest-neighbour upscale factor for readability (default: 2)")
    ap.add_argument("--hold-last-seconds", type=float, default=1.5,
                    help="freeze the final frame this long so the WIN/LOSS outcome is readable")
    ap.add_argument("--max-episode-steps", type=int, default=None,
                    help="override the per-episode step cap (default: from config)")
    ap.add_argument("--no-overlay", action="store_true", help="disable the on-frame text overlay")
    ap.add_argument("--save-jsonl", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-summary", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main():
    args = build_parser().parse_args()

    # Headless: set SDL drivers BEFORE pygame is ever initialised (i.e. before any render).
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smacdreamer.visualization import rollout, render

    want_record = args.mode in ("record", "both")
    want_interactive = args.mode in ("interactive", "both")
    if want_interactive and args.headless:
        print("[visualize] --headless requested with interactive mode; disabling the live "
              "window (recording still proceeds if requested).")
        want_interactive = False

    cfg = rollout.load_config(args.config)
    entries = rollout.resolve_map_entries(cfg, split=args.split, maps_dir=args.maps_dir,
                                          map_name=args.map_name)
    if len(entries) != 1:
        sys.exit(
            f"Select exactly one map: {len(entries)} maps available "
            f"({[e.name for e in entries][:20]}{' ...' if len(entries) > 20 else ''}). "
            "Pass --map-name.")
    entry = entries[0]

    ctx = rollout.load_agent(
        args.config, args.checkpoint, run_meta=args.run_meta, device=args.device,
        max_episode_steps=args.max_episode_steps, probe_entry=entry)

    capture = want_record or want_interactive
    env = rollout.make_single_map_env(ctx, entry, capture=capture)

    interactive_window = None
    try:
        # Determine frame size for the interactive window from one capture if needed.
        if want_interactive:
            try:
                probe_frame = render.capture_frame(env, scale=args.scale)
                h, w = probe_frame.shape[0], probe_frame.shape[1]
                interactive_window = render.InteractiveWindow(w, h, args.fps,
                                                              title=f"{entry.name} seed={args.seed}")
            except Exception as exc:
                print(f"[visualize] could not open interactive window ({exc}); continuing "
                      "without live display.")
                interactive_window = None

        result = rollout.run_episode(
            ctx, env, map_name=entry.name, seed=args.seed,
            capture_frames=want_record, overlay=not args.no_overlay, scale=args.scale,
            interactive_window=interactive_window,
        )
    finally:
        if interactive_window is not None:
            interactive_window.close()
        try:
            env.close()
        except Exception:
            pass

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{entry.name}_seed{args.seed}"

    print(f"[visualize] {entry.name} seed={args.seed}: "
          f"{'WIN' if result.battle_won else 'LOSS'} in {result.summary['episode_length']} steps "
          f"(orig_return={result.summary['total_original_return']:.3f}, "
          f"final_enemy_ehp={result.summary['final_enemy_ehp_frac']}, "
          f"mean_focus={result.summary['mean_target_focus_score']})")

    if want_record:
        mp4_path = out_dir / f"{stem}.mp4"
        frames = render.with_hold(result.frames, args.fps, hold_last_seconds=args.hold_last_seconds)
        render.write_mp4(mp4_path, frames, fps=args.fps)
        print(f"[visualize] wrote {mp4_path} ({len(frames)} frames @ {args.fps} fps)")

    if args.save_jsonl:
        jsonl_path = out_dir / f"{stem}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in result.records:
                f.write(json.dumps(rec) + "\n")
        print(f"[visualize] wrote {jsonl_path} ({len(result.records)} steps)")

    if args.save_summary:
        summary_path = out_dir / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
        print(f"[visualize] wrote {summary_path}")


if __name__ == "__main__":
    main()
