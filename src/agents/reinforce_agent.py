"""Tabular REINFORCE (policy gradient) agent."""

import numpy as np
from collections import defaultdict


class ReinforceAgent:
    """Tabular softmax policy gradient with baseline.

    Maintains a preference table theta[info_state][action] and updates
    via REINFORCE: theta += alpha * (R - baseline) * grad log pi.
    """

    def __init__(self, alpha=0.01, num_actions=2):
        self.num_actions = num_actions
        self.theta = defaultdict(lambda: np.zeros(self.num_actions))
        self.alpha = alpha
        self.baseline = 0.0
        self._baseline_count = 0
        self._frozen = False

    def _softmax(self, info_state, legal_actions):
        logits = self.theta[info_state].copy()
        mask = np.full(self.num_actions, -1e10)
        for a in legal_actions:
            mask[a] = logits[a]
        max_val = mask.max()
        exp_vals = np.exp(mask - max_val)
        probs = exp_vals / exp_vals.sum()
        return probs

    def select_action(self, info_state, legal_actions, rng):
        probs = self._softmax(info_state, legal_actions)
        return int(rng.choice(self.num_actions, p=probs))

    def freeze(self):
        self._frozen = True

    def update(self, trajectory, reward_p0):
        if self._frozen:
            return

        self._baseline_count += 1
        self.baseline += (reward_p0 - self.baseline) / self._baseline_count
        advantage = reward_p0 - self.baseline

        for player, info_state, action in trajectory:
            if player != 0:
                continue
            probs = self._softmax(info_state, list(range(self.num_actions)))
            grad = -probs.copy()
            grad[action] += 1.0
            self.theta[info_state] += self.alpha * advantage * grad
