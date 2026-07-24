"""
Action distribution plotter for DreamerV3 × SMAClite trajectory files.

Reads a per-step trajectory JSONL produced by evaluate_phase3.py
--record_trajectories and generates two plots:

  action_frequency_per_agent.png  — per-agent action frequency bar chart
  action_type_distribution.png    — aggregate idle / move / attack breakdown

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite

    python scripts\\plot_actions.py ^
        --trajectory results\\trajectories_2s_vs_1sc_200k.jsonl ^
        --n_real_agents 2 ^
        --output_dir results\\plots\\
"""

import argparse
import json
import pathlib
import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Action label helpers
# ---------------------------------------------------------------------------

_BASE_LABELS = {
    0: "noop",
    1: "stop",
    2: "move_N",
    3: "move_S",
    4: "move_E",
    5: "move_W",
}


def action_label(action_idx: int) -> str:
    return _BASE_LABELS.get(action_idx, f"attack_{action_idx - 6}")


def action_type(action_idx: int) -> str:
    if action_idx <= 1:
        return "idle"
    if action_idx <= 5:
        return "move"
    return "attack"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory(path: pathlib.Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Plot 1: per-agent action frequency
# ---------------------------------------------------------------------------

def plot_per_agent_frequency(records: list, n_real_agents: int, output_path: pathlib.Path):
    # Collect action counts per agent index.
    counts: dict[int, collections.Counter] = {
        i: collections.Counter() for i in range(n_real_agents)
    }
    for rec in records:
        actions = rec.get("actions", [])
        for i in range(min(n_real_agents, len(actions))):
            counts[i][actions[i]] += 1

    # Build sorted union of all action indices seen.
    all_actions = sorted(
        set(a for c in counts.values() for a in c)
    )
    labels = [action_label(a) for a in all_actions]

    x = np.arange(len(all_actions))
    bar_width = 0.8 / max(n_real_agents, 1)

    # Colour by action type.
    type_colors = {"idle": "#aaaaaa", "move": "#4a90d9", "attack": "#e05c5c"}
    bar_colors = [type_colors[action_type(a)] for a in all_actions]

    fig, ax = plt.subplots(figsize=(max(8, len(all_actions) * 0.9), 5))
    for i in range(n_real_agents):
        total = sum(counts[i].values()) or 1
        freqs = [counts[i].get(a, 0) / total for a in all_actions]
        offset = (i - (n_real_agents - 1) / 2) * bar_width
        bars = ax.bar(x + offset, freqs, bar_width * 0.9,
                      label=f"Agent {i}", alpha=0.85, color=bar_colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Action frequency per agent\n(grey=idle, blue=move, red=attack)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: aggregate action type distribution
# ---------------------------------------------------------------------------

def plot_action_type_distribution(records: list, n_real_agents: int, output_path: pathlib.Path):
    type_counts: collections.Counter = collections.Counter()
    for rec in records:
        actions = rec.get("actions", [])
        for i in range(min(n_real_agents, len(actions))):
            type_counts[action_type(actions[i])] += 1

    total = sum(type_counts.values()) or 1
    categories = ["idle", "move", "attack"]
    fracs = [type_counts.get(c, 0) / total for c in categories]
    colors = ["#aaaaaa", "#4a90d9", "#e05c5c"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart.
    ax = axes[0]
    ax.bar(categories, fracs, color=colors, alpha=0.85)
    ax.set_ylabel("Fraction of agent-steps")
    ax.set_title("Action type distribution (bar)")
    ax.set_ylim(0, 1.05)
    for cat, frac in zip(categories, fracs):
        ax.text(cat, frac + 0.01, f"{frac:.1%}", ha="center", fontsize=9)

    # Pie chart.
    ax2 = axes[1]
    nonzero = [(c, f, col) for c, f, col in zip(categories, fracs, colors) if f > 0]
    if nonzero:
        pie_labels, pie_fracs, pie_colors = zip(*nonzero)
        ax2.pie(pie_fracs, labels=pie_labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=90)
    ax2.set_title("Action type distribution (pie)")

    fig.suptitle(f"Aggregate over {len(records)} steps, {n_real_agents} real agent(s)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-agent action distributions from a trajectory JSONL."
    )
    parser.add_argument("--trajectory", required=True,
                        help="Path to trajectory JSONL file from evaluate_phase3.py.")
    parser.add_argument("--n_real_agents", type=int, default=2,
                        help="Number of real (non-padded) agents to plot (default: 2).")
    parser.add_argument("--output_dir", default="results/plots",
                        help="Directory to write PNG files (default: results/plots/).")
    return parser.parse_args()


def main():
    args = parse_args()
    traj_path = pathlib.Path(args.trajectory)
    output_dir = pathlib.Path(args.output_dir)

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    print(f"Loading trajectory: {traj_path}")
    records = load_trajectory(traj_path)
    print(f"  {len(records)} step records loaded")
    if not records:
        print("No records found — nothing to plot.")
        return

    n = args.n_real_agents
    stem = traj_path.stem

    plot_per_agent_frequency(
        records, n,
        output_dir / f"{stem}_action_frequency_per_agent.png",
    )
    plot_action_type_distribution(
        records, n,
        output_dir / f"{stem}_action_type_distribution.png",
    )

    print(f"\nDone. Plots written to {output_dir}/")


if __name__ == "__main__":
    main()
