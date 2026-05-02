"""Minimal Policy-Space Response Oracle (PSRO) for Kuhn Poker.

Tests whether maintaining a diverse population of opponents prevents the
co-adaptation collapse observed under zero-contingency perturbation.
"""

import copy
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env.perturbed_kuhn import KuhnPokerEnv, PerturbedKuhnPoker, BET
from src.agents.q_learning_agent import QLearningAgent


def make_env(rng):
    base = KuhnPokerEnv(rng=rng)
    return PerturbedKuhnPoker(base, removed_action=BET, affected_player=0,
                              root_only=False)


def select_action_from_qtable(q_table, info_state, legal_actions, rng,
                              epsilon=0.0):
    """Greedy (or eps-greedy) action selection from a frozen Q-table."""
    if rng.random() < epsilon:
        return int(rng.choice(legal_actions))
    num_actions = 2
    q_vals = q_table.get(info_state, np.zeros(num_actions))
    masked = np.full(num_actions, -np.inf)
    for a in legal_actions:
        masked[a] = q_vals[a]
    return int(np.argmax(masked))


def snapshot_qtable(agent):
    """Return a plain-dict copy of an agent's Q-table."""
    return {k: v.copy() for k, v in agent.q.items()}


def train_best_response(env, player, opponent_qtables, episodes, rng,
                        alpha=0.1, epsilon=0.15):
    """Train a Q-learning policy as best response to a uniform mixture.

    Args:
        player: which seat (0 or 1) the learner occupies.
        opponent_qtables: list of Q-tables; one is chosen uniformly each episode.
    """
    agent = QLearningAgent(alpha=alpha, epsilon=epsilon, num_actions=2)

    for _ in range(episodes):
        opp_q = opponent_qtables[rng.integers(len(opponent_qtables))]
        env.reset()
        trajectory = []

        while not env.is_terminal:
            cp = env.current_player
            info = env.info_state_str(cp)
            legal = env.legal_actions()

            if cp == player:
                action = agent.select_action(info, legal, rng)
            else:
                action = select_action_from_qtable(opp_q, info, legal, rng,
                                                   epsilon=0.05)
            trajectory.append((cp, info, action))
            env.step(action)

        reward_p0 = env.returns[0]
        for t_player, t_info, t_action in trajectory:
            if t_player == player:
                r = reward_p0 if player == 0 else -reward_p0
                agent.q[t_info][t_action] += alpha * (
                    r - agent.q[t_info][t_action])

    return agent


def evaluate(env, p0_qtable, p1_qtables, episodes, rng):
    """Evaluate P0 vs uniform mixture of P1 population."""
    total = 0.0
    for _ in range(episodes):
        opp_q = p1_qtables[rng.integers(len(p1_qtables))]
        env.reset()
        while not env.is_terminal:
            cp = env.current_player
            info = env.info_state_str(cp)
            legal = env.legal_actions()
            if cp == 0:
                action = select_action_from_qtable(p0_qtable, info, legal, rng)
            else:
                action = select_action_from_qtable(opp_q, info, legal, rng)
            env.step(action)
        total += env.returns[0]
    return total / episodes


def run_psro(seed, episodes_per_iter=5000, num_iters=5,
             perturb_after=3, eval_episodes=2000, quiet=False):
    rng = np.random.default_rng(seed)
    env = make_env(rng)

    p1_population = [snapshot_qtable(
        QLearningAgent(num_actions=2)
    )]

    p0_agent = None
    results = []

    for it in range(1, num_iters + 1):
        if it > perturb_after:
            env.set_perturbed(True)
            tag = "perturbed"
        else:
            env.set_perturbed(False)
            tag = "normal"

        p0_agent = train_best_response(
            env, player=0, opponent_qtables=p1_population,
            episodes=episodes_per_iter, rng=rng)

        p0_snap = snapshot_qtable(p0_agent)

        new_p1 = train_best_response(
            env, player=1, opponent_qtables=[p0_snap],
            episodes=episodes_per_iter, rng=rng)
        p1_population.append(snapshot_qtable(new_p1))

        reward = evaluate(env, p0_snap, p1_population, eval_episodes, rng)
        results.append((it, tag, reward, len(p1_population)))
        if not quiet:
            print(f"  Iter {it} [{tag:>9}]  pop={len(p1_population)}  "
                  f"P0 reward = {reward:+.3f}")

    return results


import math


def run_psro_scaling(pop_sizes=None, seeds=None, episodes_per_iter=5000,
                     eval_episodes=2000):
    """Run PSRO at multiple population sizes and report a summary table."""
    if pop_sizes is None:
        pop_sizes = [3, 5, 10, 15]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    print("=" * 70)
    print("PSRO SCALING -- Kuhn Poker -- zero contingency (full bet removal)")
    print("=" * 70)

    table = {}

    for pop_size in pop_sizes:
        num_iters = pop_size
        perturb_after = math.ceil(pop_size / 2)
        final_rewards = []

        print(f"\n>>> Pop size {pop_size}  "
              f"(iters={num_iters}, perturb_after={perturb_after})")

        for seed in seeds:
            results = run_psro(seed, episodes_per_iter=episodes_per_iter,
                               num_iters=num_iters,
                               perturb_after=perturb_after,
                               eval_episodes=eval_episodes, quiet=True)
            final_reward = results[-1][2]
            final_rewards.append(final_reward)
            print(f"    Seed {seed:>4}: {final_reward:+.3f}")

        m = np.mean(final_rewards)
        s = np.std(final_rewards)
        table[pop_size] = (m, s, final_rewards)
        print(f"    -> Mean: {m:+.3f} +/- {s:.3f}")

    print("\n" + "=" * 70)
    print(f"{'Pop size':>10}    P0 reward (mean +/- std)")
    print("-" * 45)
    for ps in pop_sizes:
        m, s, _ = table[ps]
        print(f"{ps:>10}    {m:+.3f} +/- {s:.3f}")
    print("=" * 70)

    return table


def main():
    import sys

    if "--scaling" in sys.argv:
        run_psro_scaling()
        return

    seeds = [42, 123, 789]
    all_final = []

    print("=" * 60)
    print("PSRO  --  Kuhn Poker  --  full bet removal after iter 3")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_psro(seed)
        final_reward = results[-1][2]
        all_final.append(final_reward)

    mean_reward = np.mean(all_final)
    std_reward = np.std(all_final)

    print("\n" + "=" * 60)
    print("POST-PERTURBATION RESULTS (final iteration)")
    print("=" * 60)
    for seed, r in zip(seeds, all_final):
        print(f"  Seed {seed:>4}: {r:+.3f}")
    print(f"\n  Mean: {mean_reward:+.3f}  (std {std_reward:.3f})")
    print(f"\n  Baseline self-play collapse: -0.927")
    print(f"  PSRO post-perturbation:      {mean_reward:+.3f}")
    diff = mean_reward - (-0.927)
    print(f"  Improvement:                 {diff:+.3f}")

    if mean_reward > -0.5:
        print("\n  >> PSRO substantially prevents collapse under perturbation.")
    elif mean_reward > -0.927:
        print("\n  >> PSRO partially mitigates collapse.")
    else:
        print("\n  >> PSRO does NOT prevent collapse.")


if __name__ == "__main__":
    main()
