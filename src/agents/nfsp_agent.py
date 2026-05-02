"""Tabular NFSP (Neural Fictitious Self-Play) agent.

Maintains two policy tables:
  1. best_response: Q-learning style (epsilon-greedy over Q-values)
  2. average_strategy: supervised-learning style (action frequency table)

On each step, plays best_response with probability eta (anticipatory
parameter) and average_strategy with probability 1-eta.
"""

import numpy as np
from collections import defaultdict


class NFSPAgent:
    """Tabular NFSP with anticipatory parameter eta.

    Same interface as QLearningAgent: select_action, update, freeze.
    """

    def __init__(self, alpha=0.1, epsilon=0.15, num_actions=2, eta=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self._frozen = False

        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.avg_strategy = defaultdict(lambda: np.zeros(self.num_actions))

    def _best_response_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state].copy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def _average_strategy_action(self, info_state, legal_actions, rng):
        counts = self.avg_strategy[info_state].copy()
        mask = np.zeros(self.num_actions)
        for a in legal_actions:
            mask[a] = 1.0
        counts *= mask
        total = counts.sum()
        if total > 0:
            probs = counts / total
            return int(rng.choice(self.num_actions, p=probs))
        return int(rng.choice(legal_actions))

    def select_action(self, info_state, legal_actions, rng):
        if self._frozen:
            return self._average_strategy_action(info_state, legal_actions, rng)
        if rng.random() < self.eta:
            return self._best_response_action(info_state, legal_actions, rng)
        return self._average_strategy_action(info_state, legal_actions, rng)

    def update(self, trajectory, reward_p0):
        if self._frozen:
            return
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[info_state][action] += self.alpha * (
                r - self.q[info_state][action])
            self.avg_strategy[info_state][action] += 1.0

    def freeze(self):
        self._frozen = True

    def get_average_policy(self):
        """Return average strategy as a normalised policy dict."""
        policy = {}
        for info_state, counts in self.avg_strategy.items():
            total = counts.sum()
            if total > 0:
                policy[info_state] = counts / total
            else:
                policy[info_state] = np.ones(self.num_actions) / self.num_actions
        return policy
