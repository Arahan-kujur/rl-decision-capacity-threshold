"""Tabular PPO (Proximal Policy Optimization) agent with entropy bonus."""

import numpy as np
from collections import defaultdict


class PPOAgent:
    """Tabular softmax policy with PPO-style clipped surrogate updates.

    Stores preferences theta[info_state][action] and converts to
    probabilities via softmax.  Each episode update computes the
    clipped surrogate objective plus an entropy bonus and ascends.
    """

    def __init__(self, num_actions=2, lr=0.01, clip_eps=0.2, entropy_coef=0.01):
        self.num_actions = num_actions
        self.lr = lr
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.theta = defaultdict(lambda: np.zeros(self.num_actions))
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
        return exp_vals / exp_vals.sum()

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

        old_log_probs = []
        for player, info_state, action in trajectory:
            if player != 0:
                old_log_probs.append(None)
                continue
            probs = self._softmax(info_state, list(range(self.num_actions)))
            old_log_probs.append(np.log(probs[action] + 1e-10))

        for i, (player, info_state, action) in enumerate(trajectory):
            if player != 0:
                continue
            probs = self._softmax(info_state, list(range(self.num_actions)))
            new_log = np.log(probs[action] + 1e-10)
            old_log = old_log_probs[i]

            ratio = np.exp(new_log - old_log)
            clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            surrogate = min(ratio * advantage, clipped * advantage)

            safe = probs[probs > 1e-10]
            entropy = -np.sum(safe * np.log(safe))

            grad = -probs.copy()
            grad[action] += 1.0
            self.theta[info_state] += self.lr * (
                surrogate * grad + self.entropy_coef * entropy * grad
            )
