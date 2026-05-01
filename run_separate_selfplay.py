"""Separate self-play: two independent QLearningAgent instances for P0 and P1.

Runs Kuhn full_removal with separate agents (P0_agent handles P0 decisions,
P1_agent handles P1 decisions, each only updates on its own transitions).
Compares post-perturbation reward against the shared-agent result (-0.927).
"""

import numpy as np
from src.env.perturbed_kuhn import KuhnPokerEnv, PerturbedKuhnPoker, BET
from src.agents.q_learning_agent import QLearningAgent

SEEDS = [42, 123, 456, 789, 1024]
NUM_EPISODES = 20000
PERTURBATION_EP = 10000
ALPHA = 0.1
EPSILON = 0.15
NUM_ACTIONS = 2


def play_episode(env, p0_agent, p1_agent, rng, cards):
    """Play one episode with separate agents for each player."""
    env.reset(cards=cards)
    p0_trajectory = []
    p1_trajectory = []

    while not env.is_terminal:
        player = env.current_player
        info = env.info_state_str(player)
        legal = env.legal_actions()

        if player == 0:
            action = p0_agent.select_action(info, legal, rng)
            p0_trajectory.append((player, info, action))
        else:
            action = p1_agent.select_action(info, legal, rng)
            p1_trajectory.append((player, info, action))

        env.step(action)

    return env.returns[0], p0_trajectory, p1_trajectory


def run_seed(seed):
    """Run a single seed and return post-perturbation mean reward for P0."""
    rng = np.random.default_rng(seed)

    base_env = KuhnPokerEnv(rng=rng)
    env = PerturbedKuhnPoker(base_env, removed_action=BET,
                             affected_player=0, root_only=False)

    p0_agent = QLearningAgent(alpha=ALPHA, epsilon=EPSILON,
                              num_actions=NUM_ACTIONS)
    p1_agent = QLearningAgent(alpha=ALPHA, epsilon=EPSILON,
                              num_actions=NUM_ACTIONS)

    rewards = []

    for ep in range(NUM_EPISODES):
        if ep == PERTURBATION_EP:
            env.set_perturbed(True)

        deck = np.array([0, 1, 2])
        rng.shuffle(deck)
        cards = (int(deck[0]), int(deck[1]))

        reward_p0, p0_traj, p1_traj = play_episode(env, p0_agent, p1_agent,
                                                    rng, cards)
        rewards.append(reward_p0)

        p0_agent.update(p0_traj, reward_p0)
        p1_agent.update(p1_traj, reward_p0)

    post_burnin = min(2000, (NUM_EPISODES - PERTURBATION_EP) // 4)
    post_rewards = rewards[PERTURBATION_EP + post_burnin:]
    return np.mean(post_rewards)


def main():
    print("Separate Self-Play: Kuhn Poker Full Removal")
    print(f"  Episodes: {NUM_EPISODES}, Perturbation at: {PERTURBATION_EP}")
    print(f"  Seeds: {SEEDS}")
    print("-" * 50)

    post_means = []
    for seed in SEEDS:
        result = run_seed(seed)
        post_means.append(result)
        print(f"  Seed {seed:>4d}: post-perturbation mean reward = {result:.4f}")

    overall_mean = np.mean(post_means)
    overall_std = np.std(post_means)
    print("-" * 50)
    print(f"  Overall: {overall_mean:.4f} +/- {overall_std:.4f}")
    print(f"  Shared-agent baseline: -0.927")
    print(f"  Difference: {overall_mean - (-0.927):+.4f}")


if __name__ == "__main__":
    main()
