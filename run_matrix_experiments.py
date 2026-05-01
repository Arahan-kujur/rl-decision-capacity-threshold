"""Run matrix game experiments (IPD + Matching Pennies).

Usage:
    python run_matrix_experiments.py
"""

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run matrix game perturbation experiments.")
    parser.add_argument("--config", type=Path, nargs="+",
                        help="Config file(s). Defaults to all in configs/matrix/.")
    args = parser.parse_args()

    if args.config:
        config_paths = args.config
    else:
        config_paths = sorted(Path("configs/matrix").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/matrix/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print(f"\n{'=' * 60}")
    print("  All matrix game experiments complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
