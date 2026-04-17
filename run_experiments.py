"""Entry point: run one or all Kuhn Poker perturbation experiments.

Usage:
    python run_experiments.py                                    # run all
    python run_experiments.py --config configs/root_only.yaml    # run one
"""

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run Kuhn Poker perturbation experiments.")
    parser.add_argument(
        "--config", type=Path, nargs="+",
        help="Path(s) to YAML config file(s). Runs all configs/ if omitted.")
    args = parser.parse_args()

    if args.config:
        config_paths = args.config
    else:
        config_paths = sorted(Path("configs").glob("*.yaml"))

    if not config_paths:
        print("No config files found. Place YAML files in configs/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print("\n" + "=" * 60)
    print("  All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
