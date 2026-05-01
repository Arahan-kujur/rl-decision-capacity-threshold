"""Run opponent regime comparison (self-play, fixed, mixed population).

Usage:
    python run_regime_comparison.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    config_paths = sorted(Path("configs/regimes").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/regimes/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print(f"\n{'=' * 60}")
    print("  Regime comparison complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
