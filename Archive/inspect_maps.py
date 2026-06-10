"""
Print shape profiles for all SMAClite built-in maps and any custom map files.

Usage (Anaconda Prompt / CMD):
    cd C:\\Users\\gsimru\\Documents\\smac-dreamer
    conda activate smaclite-env
    set PYTHONPATH=%cd%\\src;%cd%\\external\\dreamerv3;%cd%\\external\\smaclite
    python scripts\\inspect_maps.py
    python scripts\\inspect_maps.py --custom configs\\maps\\2s3z_v2.json configs\\maps\\2s3z_v3.json
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in [ROOT / "src", ROOT / "external" / "dreamerv3", ROOT / "external" / "smaclite"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def inspect_map_file(path: pathlib.Path) -> dict:
    """Load a custom map JSON and return its shape profile."""
    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=str(path))
    shape = {
        "name":       env.map_info.name,
        "n_agents":   env.n_agents,
        "n_enemies":  env.n_enemies,
        "n_actions":  env.n_actions,
        "obs_size":   env.obs_size,
        "state_size": env.n_agents * env.obs_size,
        "avail_size": env.n_agents * env.n_actions,
    }
    env.close()
    return shape


def inspect_builtin(name: str) -> dict:
    """Load a built-in map via gym.make and return its shape profile."""
    import smaclite  # noqa
    import gymnasium as gym
    env = gym.make(f"smaclite/{name}-v0")
    uw = env.unwrapped
    shape = {
        "name":       name,
        "n_agents":   uw.n_agents,
        "n_enemies":  uw.n_enemies,
        "n_actions":  uw.n_actions,
        "obs_size":   uw.obs_size,
        "state_size": uw.n_agents * uw.obs_size,
        "avail_size": uw.n_agents * uw.n_actions,
    }
    env.close()
    return shape


BUILTIN_MAPS = [
    "2s3z", "3s5z", "3s5z_vs_3s6z", "3s_vs_5z",
    "bane_vs_bane", "corridor", "mmm", "mmm2",
    "2s_vs_1sc", "2c_vs_64zg", "10m_vs_11m", "27m_vs_30m",
]

TARGET_SHAPE = (5, 5, 11, 80)  # Phase 2 shape profile for 2s3z

PHASE3_MAPS = {"2s3z", "3s5z", "3s5z_vs_3s6z", "2s_vs_1sc", "3s_vs_5z"}


def print_row(d: dict, compatible: bool):
    markers = []
    if compatible:
        markers.append("P2-COMPAT")
    if d['name'] in PHASE3_MAPS:
        markers.append("P3")
    tag = "  <-- " + ", ".join(markers) if markers else ""
    print(
        f"  {d['name']:<22} agents={d['n_agents']:>2}  enemies={d['n_enemies']:>2}"
        f"  actions={d['n_actions']:>3}  obs_size={d['obs_size']:>4}"
        f"  state={d['state_size']:>5}  avail={d['avail_size']:>4}{tag}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect SMAClite map shape profiles.")
    parser.add_argument(
        "--custom", nargs="*", metavar="PATH",
        help="Additional custom map JSON files to inspect.",
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("SMAClite map shape profiles")
    print(f"Target Phase 2 shape: n_agents={TARGET_SHAPE[0]}, n_enemies={TARGET_SHAPE[1]}"
          f", n_actions={TARGET_SHAPE[2]}, obs_size={TARGET_SHAPE[3]}")
    print(f"Phase 3 maps (padded multi-map): {sorted(PHASE3_MAPS)}")
    print(f"{'='*80}\n")

    print("Built-in maps:")
    for name in BUILTIN_MAPS:
        try:
            d = inspect_builtin(name)
            key = (d['n_agents'], d['n_enemies'], d['n_actions'], d['obs_size'])
            print_row(d, key == TARGET_SHAPE)
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")

    if args.custom:
        print("\nCustom maps:")
        for path_str in args.custom:
            p = pathlib.Path(path_str)
            if not p.is_absolute():
                p = ROOT / p
            try:
                d = inspect_map_file(p)
                key = (d['n_agents'], d['n_enemies'], d['n_actions'], d['obs_size'])
                print_row(d, key == TARGET_SHAPE)
            except Exception as e:
                print(f"  {path_str:<22} ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
