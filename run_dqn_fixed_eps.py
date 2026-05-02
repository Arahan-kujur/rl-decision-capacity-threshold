"""DQN with fixed epsilon (no decay) on Kuhn Poker with perturbation.

Compares against the standard decaying-epsilon DQN result (~-0.994).
Uses epsilon_decay=0 so _get_epsilon always returns epsilon_start=0.15.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env.perturbed_kuhn import KuhnPokerEnv, PerturbedKuhnPoker, BET
from src.agents.dqn_agent import DQNAgent


def run_dqn_kuhn(seed, num_episodes=50000, perturbation_episode=25000,
                 epsilon_start=0.15, epsilon_decay=0, eval_window=2000):
    rng = np.random.default_rng(seed)
    base = KuhnPokerEnv(rng=rng)
    env = PerturbedKuhnPoker(base, removed_action=BET, affected_player=0,
                             root_only=False)

    agent = DQNAgent(num_actions=2, game="kuhn", lr=1e-3,
                     epsilon_start=epsilon_start, epsilon_end=0.01,
                     epsilon_decay=epsilon_decay)

    rewards = []

    for ep in range(num_episodes):
        if ep == perturbation_episode:
            env.set_perturbed(True)

        env.reset()
        trajectory = []

        while not env.is_terminal:
            cp = env.current_player
            info = env.info_state_str(cp)
            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            trajectory.append((cp, info, action))
            env.step(action)

        reward_p0 = env.returns[0]
        rewards.append(reward_p0)
        agent.update(trajectory, reward_p0)

    post_rewards = rewards[perturbation_episode:]
    post_mean = np.mean(post_rewards)
    final_mean = np.mean(rewards[-eval_window:])
    return post_mean, final_mean


def main():
    seeds = [42, 123, 456]
    print("=" * 65)
    print("DQN Fixed Epsilon -- Kuhn Poker -- full bet removal at ep 25k")
    print(f"  epsilon=0.15 (fixed, no decay), 50k episodes, 3 seeds")
    print("=" * 65)

    post_means = []
    final_means = []

    for seed in seeds:
        print(f"\n  Running seed {seed}...", end="", flush=True)
        post_m, final_m = run_dqn_kuhn(seed)
        post_means.append(post_m)
        final_means.append(final_m)
        print(f"  post-pert mean={post_m:+.3f}  final-2k={final_m:+.3f}")

    overall_post = np.mean(post_means)
    overall_post_std = np.std(post_means)
    overall_final = np.mean(final_means)
    overall_final_std = np.std(final_means)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Fixed-eps DQN  post-perturbation:  {overall_post:+.3f} +/- {overall_post_std:.3f}")
    print(f"  Fixed-eps DQN  final 2k episodes:  {overall_final:+.3f} +/- {overall_final_std:.3f}")
    print(f"  Standard DQN   (decaying eps):     -0.994  (reference)")
    diff = overall_final - (-0.994)
    print(f"  Improvement (final 2k):            {diff:+.3f}")
    print("=" * 65)

    if overall_final > -0.5:
        print("  >> Fixed epsilon substantially mitigates collapse.")
    elif overall_final > -0.994:
        print("  >> Fixed epsilon partially mitigates collapse.")
    else:
        print("  >> Fixed epsilon does NOT prevent collapse.")


if __name__ == "__main__":
    main()
