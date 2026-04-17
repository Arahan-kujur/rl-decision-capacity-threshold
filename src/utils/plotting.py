"""Plotting utilities for experiment results."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import moving_average


def plot_results(csv_path, config, out_path):
    """Generate moving-average reward plot from an experiment CSV."""
    window = config["plot"]["window"]
    perturbation_ep = config["experiment"]["perturbation_episode"]
    name = config["experiment"]["name"]
    root_only = config["perturbation"]["root_only"]

    episodes = {"CFR": [], "Q-Learning": []}
    rewards = {"CFR": [], "Q-Learning": []}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row["agent"]
            episodes[agent].append(int(row["episode"]))
            rewards[agent].append(float(row["reward"]))

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {"CFR": "#2196F3", "Q-Learning": "#FF9800"}
    for agent in ["CFR", "Q-Learning"]:
        ma = moving_average(np.array(rewards[agent]), window)
        ax.plot(episodes[agent], ma, label=agent,
                color=colors[agent], linewidth=1.5)

    mode = "root-only" if root_only else "full"
    ax.axvline(x=perturbation_ep, color="red", linestyle="--", alpha=0.7,
               label=f"Perturbation ({mode})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Moving Avg Reward for Player 0 (window={window})")

    if root_only:
        ax.set_title(f"[{name}] P0 root bet removed — call/fold preserved")
    else:
        ax.set_title(f"[{name}] P0 bet removed at ALL nodes")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
