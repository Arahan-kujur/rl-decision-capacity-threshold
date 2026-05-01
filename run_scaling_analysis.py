"""Run scaling analysis: collapse severity and time-to-collapse across games.

Usage:
    python run_scaling_analysis.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment
from src.utils.metrics import time_to_collapse


SCALING_CONFIGS = [
    ("configs/matrix/mp_full_removal.yaml", "Matching Pennies", 1),
    ("configs/matrix/ipd_full_removal.yaml", "IPD (5 rounds)", 32),
    ("configs/full_removal.yaml", "Kuhn Poker", 12),
    ("configs/leduc/leduc_full_removal.yaml", "Leduc Poker", 288),
]


def main():
    print(f"\n{'=' * 60}")
    print("  Scaling Analysis: Collapse Across Game Complexity")
    print(f"{'=' * 60}")

    results_table = []

    for path, game_name, info_sets in SCALING_CONFIGS:
        print(f"\n>>> {game_name} ({info_sets} info sets): {path}")
        config = load_config(path)
        _, _, stat = run_experiment(config)

        _internal = {"comparisons", "_meta"}
        agents = [k for k in stat if k not in _internal]
        ql_key = next((a for a in agents if "Learning" in a or "SARSA" in a
                       or "REINFORCE" in a), agents[-1])

        post = stat[ql_key]["post_mean"]
        results_table.append({
            "game": game_name,
            "info_sets": info_sets,
            "ql_post": post,
        })

    print(f"\n{'=' * 60}")
    print("  Scaling Summary")
    print(f"{'=' * 60}")
    print(f"  {'Game':<20s}  {'Info Sets':>10s}  {'QL Post':>10s}")
    print("  " + "-" * 46)
    for row in results_table:
        print(f"  {row['game']:<20s}  {row['info_sets']:>10d}  "
              f"{row['ql_post']:>+10.4f}")


if __name__ == "__main__":
    main()
