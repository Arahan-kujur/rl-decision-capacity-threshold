"""Reward metrics and summary statistics."""

import numpy as np


def moving_average(values, window):
    """Moving average with expanding window for the first `window` elements."""
    out = np.empty(len(values))
    cumsum = np.cumsum(values)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def summarize_results(results, perturbation_ep, num_episodes):
    """Mean P0 reward before/after perturbation per agent.

    Skips the first 5k episodes (burn-in) and the first 2k after
    perturbation (transition) to get stable estimates.
    """
    summary = {}
    for agent in ["CFR", "Q-Learning"]:
        pre = np.mean([r[1] for r in results
                       if r[2] == agent and 5000 <= r[0] < perturbation_ep])
        post = np.mean([r[1] for r in results
                        if r[2] == agent
                        and perturbation_ep + 2000 <= r[0] < num_episodes])
        summary[agent] = {"pre": pre, "post": post, "delta": post - pre}
    return summary
