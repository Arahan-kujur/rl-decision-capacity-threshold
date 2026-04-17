"""CFR solver and frozen-policy agent for Kuhn Poker."""

import numpy as np
from collections import defaultdict

from src.env.perturbed_kuhn import PASS, BET, NUM_ACTIONS

CARDS = [0, 1, 2]


class CFRTrainer:
    """Vanilla CFR with full game-tree traversal.

    Enumerates all 6 card deals per iteration. The cumulative average
    strategy converges to a Nash equilibrium.
    """

    def __init__(self):
        self.regret_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def _get_strategy(self, info_set):
        regrets = np.maximum(self.regret_sum[info_set], 0)
        total = regrets.sum()
        if total > 0:
            return regrets / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS

    def train(self, iterations):
        for _ in range(iterations):
            for c0 in CARDS:
                for c1 in CARDS:
                    if c0 != c1:
                        self._cfr([c0, c1], [], 1.0, 1.0)

    def _current_player(self, history):
        n = len(history)
        if n == 0:
            return 0
        if n == 1:
            return 1
        return 0

    def _is_terminal(self, history):
        return tuple(history) in {
            (PASS, PASS), (BET, PASS), (BET, BET),
            (PASS, BET, PASS), (PASS, BET, BET),
        }

    def _payoff_p0(self, cards, history):
        h = tuple(history)
        showdown = 1 if cards[0] > cards[1] else -1
        return {
            (PASS, PASS): showdown,
            (BET, PASS): 1,
            (BET, BET): 2 * showdown,
            (PASS, BET, PASS): -1,
            (PASS, BET, BET): 2 * showdown,
        }[h]

    def _cfr(self, cards, history, p0, p1):
        """Recursive tree walk. Returns expected value for player 0."""
        if self._is_terminal(history):
            return self._payoff_p0(cards, history)

        player = self._current_player(history)
        card = cards[player]
        h_str = "".join("p" if a == PASS else "b" for a in history)
        info_set = f"{card}{h_str}"

        strategy = self._get_strategy(info_set)
        util = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            if player == 0:
                util[a] = self._cfr(cards, history + [a], p0 * strategy[a], p1)
            else:
                util[a] = self._cfr(cards, history + [a], p0, p1 * strategy[a])

        node_util = strategy @ util

        opp_reach = p1 if player == 0 else p0
        sign = 1 if player == 0 else -1
        for a in range(NUM_ACTIONS):
            self.regret_sum[info_set][a] += opp_reach * sign * (util[a] - node_util)

        player_reach = p0 if player == 0 else p1
        self.strategy_sum[info_set] += player_reach * strategy

        return node_util

    def get_average_strategy(self):
        """Return the average strategy as {info_set: prob_array}."""
        policy = {}
        for info_set, s in self.strategy_sum.items():
            total = s.sum()
            if total > 0:
                policy[info_set] = s / total
            else:
                policy[info_set] = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return policy


class CFRAgent:
    """Plays using a frozen CFR-computed strategy.

    When an action is masked (illegal), its probability mass is
    redistributed to the remaining legal actions.
    """

    def __init__(self, policy):
        self.policy = policy

    def select_action(self, info_state, legal_actions, rng):
        if info_state in self.policy:
            probs = self.policy[info_state].copy()
            mask = np.zeros(NUM_ACTIONS)
            for a in legal_actions:
                mask[a] = 1.0
            probs *= mask
            total = probs.sum()
            if total > 0:
                probs /= total
                return int(rng.choice(NUM_ACTIONS, p=probs))
        return int(rng.choice(legal_actions))

    def update(self, trajectory, reward):
        pass
