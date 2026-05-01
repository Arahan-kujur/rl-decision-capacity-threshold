"""Run deep RL (DQN) experiments.

Usage:
    python run_deep_experiments.py
    python run_deep_experiments.py --config configs/deep/dqn_kuhn_full_removal.yaml
"""

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run DQN perturbation experiments.")
    parser.add_argument("--config", type=Path, nargs="+",
                        help="Config file(s). Defaults to all in configs/deep/.")
    args = parser.parse_args()

    if args.config:
        config_paths = args.config
    else:
        config_paths = sorted(Path("configs/deep").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/deep/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print(f"\n{'=' * 60}")
    print("  All DQN experiments complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
