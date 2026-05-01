"""Run recovery experiment: perturb, collapse, then restore actions.

Usage:
    python run_recovery.py
"""

from pathlib import Path

from src.config_loader import load_config
from src.experiments.runner import run_experiment


def main():
    config_paths = sorted(Path("configs/recovery").glob("*.yaml"))

    if not config_paths:
        print("No config files found in configs/recovery/")
        raise SystemExit(1)

    for path in config_paths:
        print(f"\nLoading config: {path}")
        config = load_config(path)
        _, _, stat = run_experiment(config)

        agents = [k for k in stat if k not in {"comparisons", "_meta"}]
        print("\n  Recovery results:")
        for agent in agents:
            s = stat[agent]
            rec = s.get("post_mean", "N/A")
            print(f"    {agent}: post={rec:+.4f}" if isinstance(rec, float)
                  else f"    {agent}: {rec}")

    print(f"\n{'=' * 60}")
    print("  Recovery experiment complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
