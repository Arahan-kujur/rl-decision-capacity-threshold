"""Run algorithm comparison experiments (Q-Learning, SARSA, REINFORCE).

Usage:
    python run_algorithm_comparison.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    config_paths = sorted(Path("configs/algorithms").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/algorithms/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print(f"\n{'=' * 60}")
    print("  Algorithm comparison complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
