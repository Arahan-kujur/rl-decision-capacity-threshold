"""Run perturbation family comparison (removal, bias, noise).

Usage:
    python run_perturbation_families.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    config_paths = sorted(Path("configs/perturbation_families").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/perturbation_families/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        run_experiment(config)

    print(f"\n{'=' * 60}")
    print("  Perturbation families comparison complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
