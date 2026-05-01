"""Reward metrics, summary statistics, and statistical tests."""

import numpy as np
from scipy import stats


def moving_average(values, window):
    """Moving average with expanding window for the first `window` elements."""
    out = np.empty(len(values))
    cumsum = np.cumsum(values)
    out[:window] = cumsum[:window] / np.arange(1, window + 1)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_reward(reward, game_constants, mode="minmax"):
    """Normalize a raw reward to a game-independent scale.

    Parameters
    ----------
    reward : float or array
    game_constants : dict with min_reward, max_reward, nash_value_p0
    mode : str
        "minmax" -> [0, 1] where 0=worst, 1=best
        "nash"   -> 0.0=Nash, negative=worse, positive=better
    """
    mn = game_constants["min_reward"]
    mx = game_constants["max_reward"]
    reward = np.asarray(reward, dtype=float)
    if mode == "minmax":
        span = mx - mn
        if span == 0:
            return reward * 0.0
        return (reward - mn) / span
    elif mode == "nash":
        nash = game_constants.get("nash_value_p0")
        if nash is None:
            nash = 0.0
        span = mx - nash
        if span == 0:
            return reward * 0.0
        return (reward - nash) / span
    raise ValueError(f"Unknown normalization mode: {mode}")


# ---------------------------------------------------------------------------
# Single-seed summary
# ---------------------------------------------------------------------------

def summarize_seed(results, perturbation_ep, num_episodes, agents=None,
                   game_constants=None, recovery_ep=None):
    """Mean P0 reward before/after perturbation for a single seed.

    Uses adaptive burn-in: skips up to 25% of the pre/post window
    (capped at 5k/2k respectively) to get stable estimates.
    Supports optional recovery phase (three-phase summary).
    """
    if agents is None:
        agents = sorted(set(r[2] for r in results))

    pre_burnin = min(5000, perturbation_ep // 4)

    if recovery_ep is not None:
        post_end = recovery_ep
        post_burnin = min(2000, (recovery_ep - perturbation_ep) // 4)
    else:
        post_end = num_episodes
        post_burnin = min(2000, (num_episodes - perturbation_ep) // 4)

    summary = {}
    for agent in agents:
        pre = np.mean([r[1] for r in results
                       if r[2] == agent
                       and pre_burnin <= r[0] < perturbation_ep])
        post = np.mean([r[1] for r in results
                        if r[2] == agent
                        and perturbation_ep + post_burnin <= r[0]
                        < post_end])
        entry = {"pre": pre, "post": post, "delta": post - pre}

        if recovery_ep is not None:
            rec_burnin = min(2000, (num_episodes - recovery_ep) // 4)
            recovered = np.mean([r[1] for r in results
                                 if r[2] == agent
                                 and recovery_ep + rec_burnin <= r[0]
                                 < num_episodes])
            entry["recovered"] = recovered
            entry["recovery_delta"] = recovered - post

        if game_constants is not None:
            entry["pre_norm"] = float(normalize_reward(pre, game_constants))
            entry["post_norm"] = float(normalize_reward(post, game_constants))
            entry["delta_norm"] = entry["post_norm"] - entry["pre_norm"]
        summary[agent] = entry
    return summary


# ---------------------------------------------------------------------------
# Time-to-collapse
# ---------------------------------------------------------------------------

def time_to_collapse(results, agent, perturbation_ep, threshold,
                     window=200):
    """First episode where the moving average drops below *threshold*.

    Returns the episode number, or None if it never crosses.
    """
    rewards = [r[1] for r in results
               if r[2] == agent and r[0] >= perturbation_ep]
    if not rewards:
        return None
    ma = moving_average(np.array(rewards), window)
    episodes = [r[0] for r in results
                if r[2] == agent and r[0] >= perturbation_ep]
    for i, val in enumerate(ma):
        if val < threshold:
            return episodes[i]
    return None


def collapse_summary(all_seed_results, perturbation_ep, threshold,
                     window=200, agents=None):
    """Collapse timing statistics across seeds.

    Parameters
    ----------
    all_seed_results : list of results lists (one per seed)
    perturbation_ep : int
    threshold : float
    window : int
    agents : list of str or None

    Returns
    -------
    dict keyed by agent -> {"mean_ep", "std_ep", "fraction_collapsed",
                            "collapse_episodes"}
    """
    if agents is None:
        agents = sorted(set(r[2] for r in all_seed_results[0]))
    out = {}
    for agent in agents:
        eps = []
        for results in all_seed_results:
            t = time_to_collapse(results, agent, perturbation_ep,
                                 threshold, window)
            eps.append(t)
        valid = [e for e in eps if e is not None]
        n_collapsed = len(valid)
        out[agent] = {
            "collapse_episodes": eps,
            "fraction_collapsed": n_collapsed / len(eps),
            "mean_ep": float(np.mean(valid)) if valid else None,
            "std_ep": float(np.std(valid)) if valid else None,
        }
    return out


# ---------------------------------------------------------------------------
# Exploitability (exact, Kuhn Poker)
# ---------------------------------------------------------------------------

def compute_exploitability(policy, game_name):
    """Compute exact exploitability for a given policy in Kuhn Poker.

    Exploitability = (BR_value_p0 + BR_value_p1) / 2
    where BR_value_p0 is the value of best-responding as P0 against the policy
    playing as P1, and BR_value_p1 is the value of best-responding as P1
    against the policy playing as P0.

    Parameters
    ----------
    policy : dict mapping info_state_str -> array of action probabilities
        e.g. {"0": [0.8, 0.2], "1pb": [0.5, 0.5], ...}
    game_name : str
        Currently only "kuhn" is supported.

    Returns
    -------
    float : exploitability in raw reward units
    """
    if game_name != "kuhn":
        raise NotImplementedError(
            f"Exploitability not implemented for '{game_name}'. Only 'kuhn' is supported.")

    PASS, BET = 0, 1
    cards = [0, 1, 2]  # J, Q, K
    all_deals = [(c0, c1) for c0 in cards for c1 in cards if c0 != c1]

    def _get_policy_prob(info_state, action):
        """Get probability of action from the policy at given info state."""
        if info_state in policy:
            probs = policy[info_state]
            if hasattr(probs, '__getitem__'):
                return float(probs[action])
        return 0.5  # uniform default if info state not in policy

    def _showdown_value(c0, c1, pot):
        """P0's payoff in a showdown."""
        return pot if c0 > c1 else -pot

    def _br_value_as_p0(c0, c1):
        """Best-response value for P0 against policy playing as P1."""
        # P0 acts first. Try both actions, pick the best.
        # history: ""
        # P0 passes:
        p1_info = f"{c1}p"
        p1_bet_prob = _get_policy_prob(p1_info, BET)
        p1_pass_prob = 1.0 - p1_bet_prob

        # P0 passes -> P1 passes: showdown at pot=1 each (net +/-1)
        # P0 passes -> P1 bets -> P0 can best-respond (pass=fold or bet=call)
        val_p0_pass_p1_pass = _showdown_value(c0, c1, 1)
        # P0 passes, P1 bets: P0 at info "Xpb"
        # Best response: pick best of fold (-1) or call (showdown at 2)
        val_p0_fold = -1
        val_p0_call = _showdown_value(c0, c1, 2)
        val_p0_pass_p1_bet = max(val_p0_fold, val_p0_call)

        val_p0_passes = p1_pass_prob * val_p0_pass_p1_pass + p1_bet_prob * val_p0_pass_p1_bet

        # P0 bets:
        p1_info_b = f"{c1}b"
        p1_bet_prob_b = _get_policy_prob(p1_info_b, BET)
        p1_pass_prob_b = 1.0 - p1_bet_prob_b

        # P0 bets -> P1 passes (folds): P0 wins pot=1 -> +1
        val_p0_bet_p1_pass = 1
        # P0 bets -> P1 bets (calls): showdown at pot=2 each
        val_p0_bet_p1_bet = _showdown_value(c0, c1, 2)

        val_p0_bets = p1_pass_prob_b * val_p0_bet_p1_pass + p1_bet_prob_b * val_p0_bet_p1_bet

        return max(val_p0_passes, val_p0_bets)

    def _br_value_as_p1(c0, c1):
        """Best-response value for P1 against policy playing as P0."""
        p0_info = f"{c0}"
        p0_bet_prob = _get_policy_prob(p0_info, BET)
        p0_pass_prob = 1.0 - p0_bet_prob

        # P0 passes -> P1 best responds (pass or bet)
        # P1 passes: showdown at 1
        val_p1_pass_after_p0_pass = -_showdown_value(c0, c1, 1)
        # P1 bets -> P0 at info "Xpb"
        p0_info_pb = f"{c0}pb"
        p0_call_prob = _get_policy_prob(p0_info_pb, BET)
        p0_fold_prob = 1.0 - p0_call_prob
        # P0 folds: P1 wins +1
        # P0 calls: showdown at 2
        val_p1_bet_after_p0_pass = p0_fold_prob * 1 + p0_call_prob * (-_showdown_value(c0, c1, 2))

        val_p1_after_p0_pass = max(val_p1_pass_after_p0_pass, val_p1_bet_after_p0_pass)

        # P0 bets -> P1 best responds (pass=fold or bet=call)
        # P1 folds: -1
        val_p1_fold = -1
        # P1 calls: showdown at 2
        val_p1_call = -_showdown_value(c0, c1, 2)

        val_p1_after_p0_bet = max(val_p1_fold, val_p1_call)

        return p0_pass_prob * val_p1_after_p0_pass + p0_bet_prob * val_p1_after_p0_bet

    br_p0_total = 0.0
    br_p1_total = 0.0
    for c0, c1 in all_deals:
        br_p0_total += _br_value_as_p0(c0, c1)
        br_p1_total += _br_value_as_p1(c0, c1)

    n_deals = len(all_deals)
    br_p0_value = br_p0_total / n_deals
    br_p1_value = br_p1_total / n_deals

    return (br_p0_value + br_p1_value) / 2.0


def format_collapse_table(collapse_stats, perturbation_ep):
    """Human-readable collapse timing table."""
    lines = []
    lines.append(f"  {'Agent':14s}  {'Collapsed':>10s}  "
                 f"{'Mean ep':>10s}  {'Std ep':>10s}  "
                 f"{'Delay':>10s}")
    lines.append("  " + "-" * 62)
    for agent, s in collapse_stats.items():
        frac = f"{s['fraction_collapsed']:.0%}"
        mean_str = f"{s['mean_ep']:.0f}" if s['mean_ep'] is not None else "never"
        std_str = f"{s['std_ep']:.0f}" if s['std_ep'] is not None else "--"
        delay = ""
        if s['mean_ep'] is not None:
            delay = f"{s['mean_ep'] - perturbation_ep:.0f}"
        lines.append(f"  {agent:14s}  {frac:>10s}  {mean_str:>10s}  "
                     f"{std_str:>10s}  {delay:>10s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bootstrap & effect size helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data, n_boot=10000, ci=0.95, rng=None):
    """Bootstrap confidence interval for the mean."""
    rng = rng or np.random.default_rng(0)
    data = np.asarray(data)
    boot_means = np.array([
        data[rng.integers(0, len(data), size=len(data))].mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), \
           float(np.percentile(boot_means, 100 * (1 - alpha)))


def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-seed statistical summary
# ---------------------------------------------------------------------------

def _agent_stats(seed_summaries, agent, rng):
    """Per-agent pre/post/delta stats with CIs and paired t-test."""
    pre_vals = np.array([s[agent]["pre"] for s in seed_summaries])
    post_vals = np.array([s[agent]["post"] for s in seed_summaries])
    delta_vals = np.array([s[agent]["delta"] for s in seed_summaries])

    _, pre_post_p = stats.ttest_rel(pre_vals, post_vals)

    result = {
        "pre_mean": float(pre_vals.mean()),
        "pre_ci": bootstrap_ci(pre_vals, rng=rng),
        "post_mean": float(post_vals.mean()),
        "post_ci": bootstrap_ci(post_vals, rng=rng),
        "delta_mean": float(delta_vals.mean()),
        "delta_ci": bootstrap_ci(delta_vals, rng=rng),
        "pre_vs_post_p": float(pre_post_p),
        "cohens_d": cohens_d(post_vals, pre_vals),
        "n_seeds": len(seed_summaries),
    }

    if "post_norm" in seed_summaries[0].get(agent, {}):
        post_norm_vals = np.array(
            [s[agent]["post_norm"] for s in seed_summaries])
        result["post_norm_mean"] = float(post_norm_vals.mean())
        result["post_norm_ci"] = bootstrap_ci(post_norm_vals, rng=rng)

    return result


def _pairwise_comparison(seed_summaries, agent_a, agent_b, rng):
    """Paired comparison of post-perturbation reward between two agents."""
    a_post = np.array([s[agent_a]["post"] for s in seed_summaries])
    b_post = np.array([s[agent_b]["post"] for s in seed_summaries])
    _, p_val = stats.ttest_rel(a_post, b_post)
    diff = a_post - b_post
    return {
        "post_diff_mean": float(diff.mean()),
        "post_diff_ci": bootstrap_ci(diff, rng=rng),
        "p_value": float(p_val),
        "cohens_d": cohens_d(a_post, b_post),
    }


def statistical_summary(seed_summaries):
    """Aggregate per-seed summaries into research-grade statistics."""
    rng = np.random.default_rng(0)
    agents = sorted(seed_summaries[0].keys())
    result = {}

    for agent in agents:
        result[agent] = _agent_stats(seed_summaries, agent, rng)

    comparisons = {}
    for i, a1 in enumerate(agents):
        for a2 in agents[i + 1:]:
            key = f"{a1}_vs_{a2}"
            comparisons[key] = _pairwise_comparison(
                seed_summaries, a1, a2, rng)
    result["comparisons"] = comparisons

    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt_p(p):
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def _fmt_ci(mean, ci):
    return f"{mean:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"


def format_stat_table(stat_summary):
    """Return a human-readable string for printing to console."""
    _internal_keys = {"comparisons", "_meta"}
    agents = [k for k in stat_summary if k not in _internal_keys]
    if not agents:
        return "  (no agent data)"
    n = stat_summary[agents[0]]["n_seeds"]

    lines = [f"  Seeds: {n}"]
    lines.append(f"  {'Agent':14s}  {'Pre':>22s}  {'Post':>22s}  "
                 f"{'Delta':>22s}  {'p(pre!=post)':>12s}  {'d':>6s}")
    lines.append("  " + "-" * 108)

    for agent in agents:
        s = stat_summary[agent]
        lines.append(
            f"  {agent:14s}  {_fmt_ci(s['pre_mean'], s['pre_ci']):>22s}  "
            f"{_fmt_ci(s['post_mean'], s['post_ci']):>22s}  "
            f"{_fmt_ci(s['delta_mean'], s['delta_ci']):>22s}  "
            f"{_fmt_p(s['pre_vs_post_p']):>12s}  "
            f"{s['cohens_d']:+6.2f}")

    comparisons = stat_summary.get("comparisons", {})
    if comparisons:
        lines.append("")
        for key, c in comparisons.items():
            a1, a2 = key.split("_vs_")
            diff_str = _fmt_ci(c["post_diff_mean"], c["post_diff_ci"])
            lines.append(
                f"  {a1} vs {a2} post: diff={diff_str}  "
                f"p={_fmt_p(c['p_value'])}  d={c['cohens_d']:+.2f}")

    return "\n".join(lines)
