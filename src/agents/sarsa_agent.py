"""Tabular SARSA agent -- on-policy TD control."""

import numpy as np
from collections import defaultdict


class SarsaAgent:
    """On-policy SARSA with epsilon-greedy exploration.

    For short episodes (2-3 steps), uses the same MC-style terminal update
    as Q-learning but with on-policy action selection for the target.
    """

    def __init__(self, alpha=0.1, epsilon=0.15, num_actions=2):
        self.num_actions = num_actions
        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self._frozen = False

    def select_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state].copy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def freeze(self):
        self._frozen = True
        self.epsilon = 0.0

    def update(self, trajectory, reward_p0):
        if self._frozen:
            return
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[info_state][action] += self.alpha * (
                r - self.q[info_state][action])
