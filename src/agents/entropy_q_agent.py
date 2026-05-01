"""Q-learning with entropy bonus to encourage policy diversity."""

import numpy as np
from collections import defaultdict


class EntropyQLearningAgent:
    """Q-learning with entropy bonus to encourage policy diversity.

    Same interface as QLearningAgent but adds an entropy bonus to Q-value
    updates. The bonus is tau * H(pi(s)) where pi(s) = softmax(Q(s,:)) and
    H is Shannon entropy. This encourages visiting states where the policy
    retains uncertainty, promoting exploration and diverse strategies.
    """

    def __init__(self, alpha=0.1, epsilon=0.15, num_actions=2, entropy_tau=0.1):
        self.num_actions = num_actions
        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.entropy_tau = entropy_tau

    def _policy_entropy(self, info_state):
        """Compute Shannon entropy of the softmax policy at a state."""
        q_vals = self.q[info_state]
        q_shifted = q_vals - q_vals.max()
        exp_q = np.exp(q_shifted)
        probs = exp_q / exp_q.sum()
        log_probs = np.log(probs + 1e-10)
        return -float(np.sum(probs * log_probs))

    def select_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state].copy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def update(self, trajectory, reward_p0):
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            entropy_bonus = self.entropy_tau * self._policy_entropy(info_state)
            target = r + entropy_bonus
            self.q[info_state][action] += self.alpha * (target - self.q[info_state][action])
