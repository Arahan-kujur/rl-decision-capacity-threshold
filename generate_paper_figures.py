"""Generate publication-quality figures from existing experiment CSVs.

Usage:
    python generate_paper_figures.py

Produces 4 figures in report/latex/figures/:
  - algorithm_overlay.png
  - capacity_threshold.png
  - recovery_trajectory.png
  - cross_game_normalized.png
"""

from pathlib import Path
from glob import glob

from src.utils.plotting import (
    plot_algorithm_overlay,
    plot_capacity_threshold_normalized,
    plot_recovery_trajectory,
    plot_cross_game_normalized,
)
from src.config_loader import load_config
from src.experiments.runner import run_experiment
from src.utils.metrics import summarize_seed, statistical_summary

import csv
import numpy as np

FIGURES_DIR = Path("report/latex/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _find_csvs(pattern):
    """Find per-seed CSVs matching a name pattern."""
    seeds = [42, 123, 456, 789, 1024]
    paths = []
    for s in seeds:
        p = Path(f"results/raw/{pattern}_seed{s}.csv")
        if p.exists():
            paths.append(str(p))
    return paths


def generate_algorithm_overlay():
    """Figure A: Q-Learning vs SARSA vs REINFORCE on same axes."""
    print("  Generating algorithm_overlay.png...")
    csv_paths_dict = {
        "Q-Learning": _find_csvs("full_removal"),
        "SARSA": _find_csvs("sarsa_full_removal"),
        "REINFORCE": _find_csvs("reinforce_full_removal"),
    }
    csv_paths_dict = {k: v for k, v in csv_paths_dict.items() if v}

    if not csv_paths_dict:
        print("    SKIP: no algorithm CSVs found")
        return

    plot_algorithm_overlay(
        csv_paths_dict, window=200, perturbation_ep=10000,
        out_path=str(FIGURES_DIR / "algorithm_overlay.png"))
    print("    done")


def generate_capacity_threshold():
    """Figure B: Normalized capacity threshold plot."""
    print("  Generating capacity_threshold.png...")

    game_constants = {"min_reward": -2, "max_reward": 2}
    sweep_results = {}

    for cap, name in [(0, "capacity_0"), (1, "capacity_1"), (2, "capacity_2")]:
        csvs = _find_csvs(name)
        if not csvs:
            csvs = _find_csvs("full_removal" if cap == 0 else
                              "root_only" if cap == 1 else "capacity_2")
        if not csvs:
            continue
        summaries = []
        for csv_path in csvs:
            results = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append((int(row["episode"]),
                                    float(row["reward"]), row["agent"]))
            summaries.append(summarize_seed(results, 10000, 20000))
        clean = [{k: v for k, v in s.items() if k != "_meta"}
                 for s in summaries]
        sweep_results[cap] = statistical_summary(clean)

    if len(sweep_results) < 2:
        print("    SKIP: insufficient capacity data")
        return

    plot_capacity_threshold_normalized(
        sweep_results, game_constants,
        str(FIGURES_DIR / "capacity_threshold.png"))
    print("    done")


def generate_recovery_trajectory():
    """Figure C: Recovery experiment time-series."""
    print("  Generating recovery_trajectory.png...")
    csvs = _find_csvs("kuhn_recovery")
    if not csvs:
        print("    SKIP: no recovery CSVs found")
        return

    plot_recovery_trajectory(
        csvs, perturbation_ep=10000, recovery_ep=15000,
        window=200, out_path=str(FIGURES_DIR / "recovery_trajectory.png"))
    print("    done")


def generate_cross_game_normalized():
    """Figure D: Normalized collapse severity across games."""
    print("  Generating cross_game_normalized.png...")

    game_data = []

    for name, label, mn, mx in [
        ("full_removal", "Kuhn", -2, 2),
        ("leduc_full_removal", "Leduc", -13, 13),
        ("mp_full_removal", "Match. Pennies", -1, 1),
    ]:
        csvs = _find_csvs(name)
        if not csvs:
            continue
        posts = []
        for csv_path in csvs:
            results = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append((int(row["episode"]),
                                    float(row["reward"]), row["agent"]))
            summary = summarize_seed(results, 10000, 20000)
            rl_key = [k for k in summary if k != "CFR" and k != "_meta"][0]
            posts.append(summary[rl_key]["post"])
        if posts:
            game_data.append((label, np.mean(posts), mn, mx))

    if not game_data:
        print("    SKIP: no game CSVs found")
        return

    plot_cross_game_normalized(
        game_data, str(FIGURES_DIR / "cross_game_normalized.png"))
    print("    done")


def main():
    print("Generating publication figures...")
    generate_algorithm_overlay()
    generate_capacity_threshold()
    generate_recovery_trajectory()
    generate_cross_game_normalized()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
