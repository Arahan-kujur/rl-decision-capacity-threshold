"""Plotting utilities for multi-seed experiment results."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import moving_average, bootstrap_ci

AGENT_COLORS = {
    "CFR": "#2196F3",
    "Q-Learning": "#FF9800",
    "QL-Frozen": "#4CAF50",
}


def _load_seed_csv(csv_path):
    """Load a single-seed CSV into {agent: (episodes, rewards)}."""
    episodes = {}
    rewards = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row["agent"]
            episodes.setdefault(agent, []).append(int(row["episode"]))
            rewards.setdefault(agent, []).append(float(row["reward"]))
    return episodes, rewards


def _ci_band(stacked, window, rng):
    """Compute bootstrap CI bounds at sampled points, interpolate the rest."""
    n_pts = stacked.shape[1]
    lo = np.empty(n_pts)
    hi = np.empty(n_pts)
    step = max(1, window // 4)
    idx = list(range(0, n_pts, step))
    if idx[-1] != n_pts - 1:
        idx.append(n_pts - 1)
    for i in idx:
        lo[i], hi[i] = bootstrap_ci(stacked[:, i], n_boot=5000, rng=rng)
    lo = np.interp(np.arange(n_pts), idx, lo[idx])
    hi = np.interp(np.arange(n_pts), idx, hi[idx])
    return lo, hi


# ---------------------------------------------------------------------------
# Time-series reward plot (handles N agents, any game)
# ---------------------------------------------------------------------------

def plot_results(csv_paths, config, out_path):
    """Generate multi-seed moving-average reward plot with bootstrap CI bands."""
    window = config["plot"]["window"]
    perturbation_ep = config["experiment"]["perturbation_episode"]
    name = config["experiment"]["name"]
    n_seeds = len(csv_paths)

    all_ma = {}
    episode_grid = None

    for csv_path in csv_paths:
        episodes, rewards = _load_seed_csv(csv_path)
        for agent in episodes:
            ma = moving_average(np.array(rewards[agent]), window)
            all_ma.setdefault(agent, []).append(ma)
        if episode_grid is None:
            first_agent = next(iter(episodes))
            episode_grid = np.array(episodes[first_agent])

    agents = sorted(all_ma.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)

    for agent in agents:
        color = AGENT_COLORS.get(agent, None)
        stacked = np.array(all_ma[agent])
        mean_ma = stacked.mean(axis=0)

        if n_seeds > 1:
            lo, hi = _ci_band(stacked, window, rng)
            ax.fill_between(episode_grid, lo, hi, alpha=0.18, color=color)

        label = agent if n_seeds == 1 else f"{agent} (n={n_seeds})"
        ax.plot(episode_grid, mean_ma, label=label,
                color=color, linewidth=1.5)

    pert_cfg = config["perturbation"]
    if pert_cfg.get("disabled"):
        mode = "none"
    elif pert_cfg.get("root_only"):
        mode = "root-only"
    else:
        mode = "full"

    ax.axvline(x=perturbation_ep, color="red", linestyle="--", alpha=0.7,
               label=f"Perturbation ({mode})")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Moving Avg Reward for Player 0 (window={window})")
    ax.set_title(f"[{name}] P0 reward over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Decision capacity sweep plot
# ---------------------------------------------------------------------------

def plot_capacity_sweep(sweep_results, out_path):
    """Plot post-perturbation reward vs decision capacity."""
    capacities = sorted(sweep_results.keys())
    sample = sweep_results[capacities[0]]
    _internal = {"comparisons", "_meta"}
    agents = [k for k in sample if k not in _internal]

    fig, ax = plt.subplots(figsize=(8, 5))

    for agent in agents:
        color = AGENT_COLORS.get(agent, None)
        means = [sweep_results[c][agent]["post_mean"] for c in capacities]
        lo = [sweep_results[c][agent]["post_ci"][0] for c in capacities]
        hi = [sweep_results[c][agent]["post_ci"][1] for c in capacities]
        yerr_lo = [m - l for m, l in zip(means, lo)]
        yerr_hi = [h - m for m, h in zip(means, hi)]
        ax.errorbar(capacities, means, yerr=[yerr_lo, yerr_hi],
                     label=agent, capsize=5, marker="o", color=color)

    ax.set_xlabel("Decision Capacity (P0 decision points with choice)")
    ax.set_ylabel("Post-Perturbation Mean Reward (P0)")
    ax.set_title("Decision Capacity vs Performance")
    ax.set_xticks(capacities)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Variance decomposition bar chart
# ---------------------------------------------------------------------------

def plot_variance_decomposition(var_table, out_path):
    """Stacked bar chart of variance contributions."""
    agents = sorted(var_table.keys())
    components = ["env", "policy", "interaction"]
    comp_colors = {"env": "#42A5F5", "policy": "#FFA726",
                   "interaction": "#AB47BC"}

    x = np.arange(len(agents))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(agents))

    for comp in components:
        vals = [max(var_table[a][comp], 0) for a in agents]
        ax.bar(x, vals, width, bottom=bottom, label=comp,
               color=comp_colors[comp])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Variance of Post-Perturbation Reward")
    ax.set_title("Variance Decomposition (Environment vs Policy)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Severity sweep plot
# ---------------------------------------------------------------------------

def plot_severity_sweep(sweep_results, out_path):
    """Grouped bar chart: timing x severity for QL post-perturbation reward.

    Parameters
    ----------
    sweep_results : dict[(timing_label, severity_label)] -> stat_summary
    out_path : str
    """
    timings = sorted(set(t for t, _ in sweep_results.keys()))
    severities = sorted(set(s for _, s in sweep_results.keys()))

    x = np.arange(len(timings))
    width = 0.35
    sev_colors = {"severe": "#E53935", "mild": "#43A047"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, sev in enumerate(severities):
        means = []
        errs_lo = []
        errs_hi = []
        for t in timings:
            s = sweep_results[(t, sev)]
            _internal = {"comparisons", "_meta"}
            agents = [k for k in s if k not in _internal]
            ql_key = "Q-Learning" if "Q-Learning" in agents else agents[-1]
            m = s[ql_key]["post_mean"]
            lo, hi = s[ql_key]["post_ci"]
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)
        offset = (i - len(severities) / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=[errs_lo, errs_hi], capsize=4,
               label=sev, color=sev_colors.get(sev, None))

    ax.set_xticks(x)
    ax.set_xticklabels(timings)
    ax.set_xlabel("Perturbation Timing")
    ax.set_ylabel("Q-Learning Post-Perturbation Reward (P0)")
    ax.set_title("Perturbation Timing x Severity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Cross-game comparison
# ---------------------------------------------------------------------------

def plot_cross_game(game_stats, out_path):
    """Side-by-side normalized post-perturbation reward across games.

    Parameters
    ----------
    game_stats : dict[str, stat_summary]
        Keyed by game name.
    out_path : str
    """
    games = sorted(game_stats.keys())
    _internal = {"comparisons", "_meta"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x_offset = 0
    tick_positions = []
    tick_labels = []
    bar_width = 0.3

    for game in games:
        s = game_stats[game]
        agents = [k for k in s if k not in _internal]
        for i, agent in enumerate(agents):
            color = AGENT_COLORS.get(agent, None)
            m = s[agent]["post_mean"]
            lo, hi = s[agent]["post_ci"]
            pos = x_offset + i * bar_width
            ax.bar(pos, m, bar_width, yerr=[[m - lo], [hi - m]],
                   capsize=4, color=color,
                   label=agent if x_offset == 0 else "")
            tick_positions.append(pos)
            tick_labels.append(f"{game}\n{agent}")
        x_offset += len(agents) * bar_width + 0.4

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Post-Perturbation Reward (P0)")
    ax.set_title("Cross-Game Comparison: Full Removal")
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.add(l)]
    if unique:
        ax.legend(*zip(*unique))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Publication figures (quality polish)
# ---------------------------------------------------------------------------

def plot_algorithm_overlay(csv_paths_dict, window, perturbation_ep, out_path):
    """Overlay multiple algorithms on same axes under same perturbation.

    Parameters
    ----------
    csv_paths_dict : dict[str, list[str]]
        Maps algorithm label -> list of per-seed CSV paths.
    """
    algo_colors = {
        "Q-Learning": "#FF9800",
        "SARSA": "#9C27B0",
        "REINFORCE": "#009688",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)

    for algo_label, csv_paths in csv_paths_dict.items():
        all_ma = []
        episode_grid = None
        for csv_path in csv_paths:
            episodes, rewards = _load_seed_csv(csv_path)
            rl_key = [k for k in episodes if k != "CFR"][0]
            ma = moving_average(np.array(rewards[rl_key]), window)
            all_ma.append(ma)
            if episode_grid is None:
                episode_grid = np.array(episodes[rl_key])

        stacked = np.array(all_ma)
        mean_ma = stacked.mean(axis=0)
        color = algo_colors.get(algo_label, None)

        if len(csv_paths) > 1:
            lo, hi = _ci_band(stacked, window, rng)
            ax.fill_between(episode_grid, lo, hi, alpha=0.12, color=color)

        ax.plot(episode_grid, mean_ma, label=algo_label,
                color=color, linewidth=1.8)

    ax.axvline(x=perturbation_ep, color="red", linestyle="--",
               alpha=0.7, label="Perturbation")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Moving Avg Reward (window={window})")
    ax.set_title("Algorithm Invariance Under Full Removal (Kuhn)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_capacity_threshold_normalized(sweep_results, game_constants, out_path):
    """Capacity threshold plot with normalized y-axis -- THE key figure."""
    capacities = sorted(sweep_results.keys())
    _internal = {"comparisons", "_meta"}
    mn = game_constants["min_reward"]
    mx = game_constants["max_reward"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for agent in ["CFR", "Q-Learning"]:
        if agent not in sweep_results[capacities[0]]:
            continue
        color = AGENT_COLORS.get(agent, None)
        raw_means = [sweep_results[c][agent]["post_mean"] for c in capacities]
        raw_lo = [sweep_results[c][agent]["post_ci"][0] for c in capacities]
        raw_hi = [sweep_results[c][agent]["post_ci"][1] for c in capacities]

        norm_means = [(v - mn) / (mx - mn) for v in raw_means]
        norm_lo = [(v - mn) / (mx - mn) for v in raw_lo]
        norm_hi = [(v - mn) / (mx - mn) for v in raw_hi]

        yerr_lo = [m - l for m, l in zip(norm_means, norm_lo)]
        yerr_hi = [h - m for m, h in zip(norm_means, norm_hi)]

        ax.errorbar(capacities, norm_means, yerr=[yerr_lo, yerr_hi],
                    label=agent, capsize=6, marker="o", markersize=8,
                    color=color, linewidth=2)

    ax.set_xlabel("Contingent Action Capacity", fontsize=12)
    ax.set_ylabel("Normalized Performance", fontsize=12)
    ax.set_title("The Capacity Threshold Effect", fontsize=13)
    ax.set_xticks(capacities)
    ax.set_xticklabels(["0\n(zero-contingency)", "1\n(residual)", "2\n(full)"])
    ax.axhspan(0, 0.3, alpha=0.06, color="red")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_recovery_trajectory(csv_paths, perturbation_ep, recovery_ep,
                             window, out_path):
    """Time-series: pre -> collapse -> recovery with vertical annotations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    rng = np.random.default_rng(0)

    all_ma = []
    episode_grid = None
    for csv_path in csv_paths:
        episodes, rewards = _load_seed_csv(csv_path)
        rl_key = [k for k in episodes if k != "CFR"][0]
        ma = moving_average(np.array(rewards[rl_key]), window)
        all_ma.append(ma)
        if episode_grid is None:
            episode_grid = np.array(episodes[rl_key])

    stacked = np.array(all_ma)
    mean_ma = stacked.mean(axis=0)

    if len(csv_paths) > 1:
        lo, hi = _ci_band(stacked, window, rng)
        ax.fill_between(episode_grid, lo, hi, alpha=0.18, color="#FF9800")

    ax.plot(episode_grid, mean_ma, color="#FF9800", linewidth=1.8,
            label="Q-Learning")

    ax.axvline(x=perturbation_ep, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=recovery_ep, color="green", linestyle="--", linewidth=1.5)

    y_ann = mean_ma.min() * 0.5
    ax.annotate("Actions removed", xy=(perturbation_ep + 200, y_ann),
                fontsize=9, color="red")
    ax.annotate("Actions restored", xy=(recovery_ep + 200, y_ann),
                fontsize=9, color="green")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Moving Avg Reward (window={window})")
    ax.set_title("Recovery Experiment: Collapse is Reversible")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cross_game_normalized(game_data, out_path):
    """Normalized collapse severity bar chart across games.

    Parameters
    ----------
    game_data : list of (game_name, raw_post, min_reward, max_reward)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [g[0] for g in game_data]
    normalized = [(g[1] - g[2]) / (g[3] - g[2]) for g in game_data]

    colors = ["#FF9800", "#4CAF50", "#9C27B0"]
    bars = ax.bar(range(len(names)), normalized,
                  color=colors[:len(names)], width=0.6,
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Normalized Reward\n(0 = worst, 1 = best)", fontsize=11)
    ax.set_title("Cross-Game Collapse Severity (Normalized)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5,
               label="Midpoint")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    for bar, val in zip(bars, normalized):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
