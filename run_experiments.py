"""Entry point: run one or all Kuhn Poker perturbation experiments.

Usage:
    python run_experiments.py                           # run all configs
    python run_experiments.py configs/full_removal.yaml  # run one
"""

import sys
from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    if len(sys.argv) > 1:
        config_paths = [Path(p) for p in sys.argv[1:]]
    else:
        config_paths = sorted(Path("configs").glob("*.yaml"))

    if not config_paths:
        print("No config files found. Place YAML files in configs/")
        sys.exit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print("\n" + "=" * 60)
    print("  All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
