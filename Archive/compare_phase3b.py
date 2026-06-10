"""
Phase 3B learning comparison: DreamerV3 vs valid-action random baseline.

Loads multiple evaluation result JSONs (produced by evaluate_phase3.py and
random_baseline_phase3.py), computes per-map and aggregate deltas vs the random
baseline, optionally scans training logdirs for NaN/Inf in metrics.jsonl, and
writes a Markdown comparison report.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\compare_phase3b.py ^
        --random  results\\random_phase3_30eps.json ^
        --results results\\eval_phase3_debug_5k_30eps.json ^
                  results\\eval_phase3_debug_50k_30eps.json ^
                  results\\eval_phase3_overfit_2s3z_30eps.json ^
        --labels  "5k" "50k" "overfit_2s3z" ^
        --logdirs logs\\smaclite_phase3\\debug_5k ^
                  logs\\smaclite_phase3\\debug_50k ^
                  logs\\smaclite_phase3\\overfit_2s3z ^
        --output  results\\phase3b_learning_report.md
"""

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_result(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def get_map_agg(result: dict, map_name: str):
    """Return aggregate dict for map_name, or None if not present."""
    return result.get("maps", {}).get(map_name, {}).get("aggregate")


def all_map_names(results: list) -> list:
    """Union of map names across all result JSONs, preserving first-seen order."""
    seen = {}
    for r in results:
        for name in r.get("maps", {}):
            seen[name] = None
    return list(seen)


# ---------------------------------------------------------------------------
# NaN / Inf scan
# ---------------------------------------------------------------------------

def scan_metrics_jsonl(logdir: str) -> dict:
    """Scan metrics.jsonl in logdir for NaN or ±Inf float values.

    Returns a dict with keys: path_checked, lines_scanned, nan_count, inf_count, found.
    """
    logdir_path = pathlib.Path(logdir)
    jsonl_path = logdir_path / "metrics.jsonl"
    result = {
        "path_checked": str(jsonl_path),
        "lines_scanned": 0,
        "nan_count": 0,
        "inf_count": 0,
        "found": False,
    }
    if not jsonl_path.exists():
        result["error"] = "metrics.jsonl not found"
        return result

    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            result["lines_scanned"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for v in obj.values():
                if isinstance(v, float):
                    if math.isnan(v):
                        result["nan_count"] += 1
                    elif math.isinf(v):
                        result["inf_count"] += 1

    result["found"] = result["nan_count"] > 0 or result["inf_count"] > 0
    return result


# ---------------------------------------------------------------------------
# Per-map delta computation
# ---------------------------------------------------------------------------

def compute_map_row(map_name: str, random_agg: dict, runs: list) -> dict:
    """
    Compute comparison data for one map across all runs.

    random_agg: aggregate dict from the random baseline for this map (may be None).
    runs: list of dicts with keys label, agg (may be None), checkpoint_loaded, invalid.

    Returns a dict keyed by label with per-map metrics and deltas.
    """
    row = {"map": map_name, "random": random_agg, "runs": {}}
    for run in runs:
        label = run["label"]
        agg = run["agg"]
        invalid = run["invalid"]
        if agg is None:
            row["runs"][label] = {"present": False}
            continue
        if invalid:
            row["runs"][label] = {
                "present": True,
                "invalid": True,
                "reason": run.get("invalid_reason", "checkpoint_loaded=false"),
                "win_rate": agg.get("win_rate"),
                "mean_reward": agg.get("mean_episode_reward"),
            }
            continue

        win_rate = agg.get("win_rate", 0.0)
        mean_reward = agg.get("mean_episode_reward", 0.0)
        mask_fail = agg.get("mean_masking_failure_rate", 0.0)

        r_win = random_agg["win_rate"] if random_agg else None
        r_reward = random_agg["mean_episode_reward"] if random_agg else None

        win_delta = (win_rate - r_win) if r_win is not None else None
        reward_delta = (mean_reward - r_reward) if r_reward is not None else None
        beats_random = (reward_delta > 0) if reward_delta is not None else None

        row["runs"][label] = {
            "present": True,
            "invalid": False,
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "mean_masking_failure_rate": mask_fail,
            "win_rate_delta_vs_random": win_delta,
            "reward_delta_vs_random": reward_delta,
            "beats_random_mean_reward": beats_random,
        }
    return row


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt_rate(v, decimals=3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _fmt_delta(v) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.3f}"


def build_report(
    random_result: dict,
    dreamer_results: list,
    labels: list,
    nan_scans: list,
    map_names: list,
) -> str:
    lines = []

    lines.append("# Phase 3B Learning Validation Report\n")
    lines.append(
        "Comparison of DreamerV3 checkpoints against the valid-action random baseline "
        "on the Phase 3 padded multi-map SMAClite set.\n"
    )

    # --- Run metadata ---
    lines.append("## Run Metadata\n")
    lines.append(f"- **Random baseline**: `{random_result.get('manifest', 'N/A')}`  "
                 f"episodes={random_result.get('episodes_per_map', 'N/A')}  "
                 f"seed={random_result.get('seed', 'N/A')}")
    for label, dr in zip(labels, dreamer_results):
        ckpt_loaded = dr.get("checkpoint_loaded")
        valid_str = "" if ckpt_loaded is not False else "  **⚠ INVALID — checkpoint_loaded=false**"
        lines.append(
            f"- **{label}**: logdir=`{dr.get('logdir', 'N/A')}`  "
            f"checkpoint_loaded={ckpt_loaded}{valid_str}"
        )
    lines.append("")

    # --- NaN / Inf scan ---
    if nan_scans:
        lines.append("## NaN / Inf Scan\n")
        lines.append("| Run | metrics.jsonl | Lines | NaN | Inf | Status |")
        lines.append("|-----|--------------|-------|-----|-----|--------|")
        for label, scan in nan_scans:
            if "error" in scan:
                status = f"ERROR: {scan['error']}"
            elif scan["found"]:
                status = "**FOUND — investigate**"
            else:
                status = "clean"
            lines.append(
                f"| {label} | `{pathlib.Path(scan['path_checked']).name}` "
                f"| {scan['lines_scanned']} "
                f"| {scan['nan_count']} "
                f"| {scan['inf_count']} "
                f"| {status} |"
            )
        lines.append("")

    # --- Per-map comparison table ---
    lines.append("## Per-Map Comparison\n")

    # Build column headers: Map | Random win/reward | label win/reward/Δreward/maskfail ...
    col_headers = ["Map", "Rand win", "Rand reward"]
    for label in labels:
        col_headers += [f"{label} win", f"{label} reward", f"{label} Δreward", f"{label} maskfail"]
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(col_headers)) + " |")

    # Collect rows
    all_rows = []
    for map_name in map_names:
        random_agg = get_map_agg(random_result, map_name)
        runs = []
        for label, dr in zip(labels, dreamer_results):
            agg = get_map_agg(dr, map_name)
            invalid = (dr.get("checkpoint_loaded") is False)
            runs.append({
                "label": label,
                "agg": agg,
                "invalid": invalid,
                "invalid_reason": "checkpoint_loaded=false",
            })
        row = compute_map_row(map_name, random_agg, runs)
        all_rows.append(row)

        r_win = _fmt_rate(random_agg["win_rate"] if random_agg else None)
        r_reward = _fmt_rate(random_agg["mean_episode_reward"] if random_agg else None)
        cells = [map_name, r_win, r_reward]
        for label in labels:
            run_data = row["runs"].get(label, {})
            if not run_data.get("present", False):
                cells += ["N/A", "N/A", "N/A", "N/A"]
            elif run_data.get("invalid", False):
                cells += [
                    f"~~{_fmt_rate(run_data.get('win_rate'))}~~",
                    f"~~{_fmt_rate(run_data.get('mean_reward'))}~~",
                    "INVALID",
                    "INVALID",
                ]
            else:
                cells += [
                    _fmt_rate(run_data["win_rate"]),
                    _fmt_rate(run_data["mean_reward"]),
                    _fmt_delta(run_data["reward_delta_vs_random"]),
                    _fmt_rate(run_data["mean_masking_failure_rate"], 4),
                ]
        lines.append("| " + " | ".join(cells) + " |")

    # OVERALL row using aggregate from each result
    r_overall = random_result.get("aggregate", {})
    r_win = _fmt_rate(r_overall.get("win_rate"))
    r_reward = _fmt_rate(r_overall.get("mean_episode_reward"))
    overall_cells = ["**OVERALL**", r_win, r_reward]
    for label, dr in zip(labels, dreamer_results):
        invalid = (dr.get("checkpoint_loaded") is False)
        agg = dr.get("aggregate", {})
        if not agg:
            overall_cells += ["N/A", "N/A", "N/A", "N/A"]
        elif invalid:
            overall_cells += ["INVALID", "INVALID", "INVALID", "INVALID"]
        else:
            win = agg.get("win_rate", 0.0)
            reward = agg.get("mean_episode_reward", 0.0)
            r_mean = r_overall.get("mean_episode_reward")
            delta = (reward - r_mean) if r_mean is not None else None
            overall_cells += [
                _fmt_rate(win),
                _fmt_rate(reward),
                _fmt_delta(delta),
                _fmt_rate(agg.get("mean_masking_failure_rate", 0.0), 4),
            ]
    lines.append("| " + " | ".join(overall_cells) + " |")
    lines.append("")

    # --- Acceptance checks ---
    lines.append("## Acceptance Checks\n")
    lines.append("Checks whether each valid Dreamer run beats random mean_reward per map.\n")

    # Find 50k label index (if present)
    for label, dr in zip(labels, dreamer_results):
        if dr.get("checkpoint_loaded") is False:
            lines.append(f"### {label}: SKIPPED (checkpoint_loaded=false — not valid for comparison)\n")
            continue

        lines.append(f"### {label}\n")
        lines.append("| Map | Random reward | Dreamer reward | Δreward | Result |")
        lines.append("|-----|--------------|----------------|---------|--------|")
        pass_count = 0
        fail_count = 0
        na_count = 0
        for row in all_rows:
            map_name = row["map"]
            random_agg = row["random"]
            run_data = row["runs"].get(label, {})
            if not run_data.get("present", False):
                result_str = "N/A"
                na_count += 1
                lines.append(
                    f"| {map_name} | N/A | N/A | N/A | N/A |"
                )
            elif run_data.get("invalid", False):
                lines.append(
                    f"| {map_name} | N/A | INVALID | INVALID | INVALID |"
                )
                na_count += 1
            else:
                r_reward_val = random_agg["mean_episode_reward"] if random_agg else None
                d_reward_val = run_data["mean_reward"]
                delta_val = run_data["reward_delta_vs_random"]
                beats = run_data["beats_random_mean_reward"]
                result_str = "PASS" if beats else "FAIL"
                if beats:
                    pass_count += 1
                else:
                    fail_count += 1
                lines.append(
                    f"| {map_name} | {_fmt_rate(r_reward_val)} "
                    f"| {_fmt_rate(d_reward_val)} "
                    f"| {_fmt_delta(delta_val)} "
                    f"| **{result_str}** |"
                )
        lines.append("")
        if pass_count + fail_count > 0:
            lines.append(
                f"**Summary**: {pass_count}/{pass_count + fail_count} maps beat random mean_reward"
                + (f"  ({na_count} N/A)" if na_count else "")
            )
        lines.append("")

    # --- Caveats ---
    lines.append("## Caveats\n")
    lines.append(
        "- **Single seed**: Phase 3B uses a single random seed (seed=42). "
        "Results are indicative only unless repeated with multiple seeds.\n"
        "- **Debug configuration**: Training used `--configs debug`, which reduces batch size "
        "and model capacity compared to a production configuration. "
        "These results are not the final performance configuration.\n"
        "- **Zero win rate**: A win rate of 0 is not automatic failure. "
        "Reward improvement over the random baseline shows the agent is learning to reduce "
        "casualties and deal damage even if no episode is won outright.\n"
        "- **INVALID runs**: Any run where `checkpoint_loaded=false` indicates the checkpoint "
        "could not be loaded. Metrics from such runs are not valid for learning comparison "
        "and are marked INVALID in all tables.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3B learning comparison report."
    )
    parser.add_argument("--random", required=True,
                        help="Path to random baseline result JSON.")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to Dreamer evaluation result JSONs (in label order).")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Short labels for each result (same order as --results).")
    parser.add_argument("--logdirs", nargs="*", default=[],
                        help="Training logdirs to scan for NaN/Inf in metrics.jsonl "
                             "(same order as --results, or omit to skip scan).")
    parser.add_argument("--output", default="results/phase3b_learning_report.md",
                        help="Path to write the Markdown report.")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.labels) != len(args.results):
        print(f"ERROR: --labels ({len(args.labels)}) must match --results ({len(args.results)})")
        sys.exit(1)

    random_result = load_result(args.random)
    dreamer_results = [load_result(p) for p in args.results]

    # NaN/Inf scan
    nan_scans = []
    if args.logdirs:
        if len(args.logdirs) != len(args.results):
            print(
                f"WARNING: --logdirs ({len(args.logdirs)}) does not match "
                f"--results ({len(args.results)}). Skipping NaN scan."
            )
        else:
            for label, logdir in zip(args.labels, args.logdirs):
                scan = scan_metrics_jsonl(logdir)
                nan_scans.append((label, scan))

    # Collect all map names (union across all results)
    all_results = [random_result] + dreamer_results
    map_names = all_map_names(all_results)

    # Console summary
    print(f"\nPhase 3B Comparison")
    print(f"Random baseline : {args.random}")
    for label, path in zip(args.labels, args.results):
        print(f"  {label:<18} : {path}")
    print(f"Maps            : {map_names}")
    print()

    for label, dr in zip(args.labels, dreamer_results):
        ckpt_loaded = dr.get("checkpoint_loaded")
        if ckpt_loaded is False:
            print(f"  WARNING: {label} has checkpoint_loaded=false — marking INVALID")
    print()

    if nan_scans:
        print("NaN/Inf scan:")
        for label, scan in nan_scans:
            if "error" in scan:
                print(f"  {label}: {scan['error']}")
            else:
                status = "FOUND" if scan["found"] else "clean"
                print(
                    f"  {label}: lines={scan['lines_scanned']}  "
                    f"nan={scan['nan_count']}  inf={scan['inf_count']}  [{status}]"
                )
        print()

    report = build_report(random_result, dreamer_results, args.labels, nan_scans, map_names)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
