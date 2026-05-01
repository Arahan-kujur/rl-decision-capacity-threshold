"""CFR solver for Liar's Dice (1 die each, 6 faces)."""

import numpy as np
from collections import defaultdict

from src.env.liars_dice import (
    CHALLENGE, NUM_ACTIONS, FACES, claim_to_qf, action_str,
)

ALL_DEALS = [(d0, d1) for d0 in range(1, 7) for d1 in range(1, 7)]


class LiarsDiceCFRTrainer:
    """Vanilla CFR for Liar's Dice with full game-tree enumeration."""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    def _get_strategy(self, info_set, legal):
        regrets = np.maximum(self.regret_sum[info_set], 0)
        mask = np.zeros(NUM_ACTIONS)
        for a in legal:
            mask[a] = 1.0
        regrets *= mask
        total = regrets.sum()
        if total > 0:
            return regrets / total
        n_legal = len(legal)
        strat = np.zeros(NUM_ACTIONS)
        for a in legal:
            strat[a] = 1.0 / n_legal
        return strat

    def _legal_actions(self, last_claim):
        actions = []
        if last_claim > 0:
            actions.append(CHALLENGE)
        for a in range(last_claim + 1, NUM_ACTIONS):
            actions.append(a)
        return actions

    def _info_set(self, dice, player, history):
        die_val = dice[player]
        h_str = ",".join(action_str(a) for a in history)
        return f"d{die_val}|{h_str}"

    def train(self, iterations):
        for it in range(iterations):
            for deal in ALL_DEALS:
                self._cfr(deal, [], 0, 1.0, 1.0)

    def _cfr(self, dice, history, last_claim, p0, p1):
        """Recursive CFR. Returns expected value for player 0."""
        player = len(history) % 2
        legal = self._legal_actions(last_claim)

        if not legal:
            return 0.0

        if len(history) > 0 and history[-1] == CHALLENGE:
            prev_claim = last_claim
            q, f = claim_to_qf(prev_claim)
            actual_count = sum(1 for d in dice if d == f)
            challenger = player
            claimer = 1 - player
            if actual_count >= q:
                return -1.0 if (len(history) - 1) % 2 == 0 else 1.0
            else:
                return 1.0 if (len(history) - 1) % 2 == 0 else -1.0

        info_set = self._info_set(dice, player, history)
        strategy = self._get_strategy(info_set, legal)
        util = np.zeros(NUM_ACTIONS)

        for a in legal:
            new_last = a if a != CHALLENGE else last_claim
            if a == CHALLENGE:
                q, f = claim_to_qf(last_claim)
                actual = sum(1 for d in dice if d == f)
                if actual >= q:
                    child_val = -1.0 if player == 0 else 1.0
                else:
                    child_val = 1.0 if player == 0 else -1.0
                util[a] = child_val
            else:
                if player == 0:
                    util[a] = self._cfr(dice, history + [a], a,
                                        p0 * strategy[a], p1)
                else:
                    util[a] = self._cfr(dice, history + [a], a,
                                        p0, p1 * strategy[a])

        node_util = sum(strategy[a] * util[a] for a in legal)

        opp_reach = p1 if player == 0 else p0
        sign = 1 if player == 0 else -1
        for a in legal:
            self.regret_sum[info_set][a] += \
                opp_reach * sign * (util[a] - node_util)

        player_reach = p0 if player == 0 else p1
        self.strategy_sum[info_set] += player_reach * strategy

        return node_util

    def get_average_strategy(self):
        policy = {}
        for info_set, s in self.strategy_sum.items():
            total = s.sum()
            if total > 0:
                policy[info_set] = s / total
            else:
                policy[info_set] = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return policy

    def nash_value_p0(self):
        policy = self.get_average_strategy()
        total_val = 0.0
        for deal in ALL_DEALS:
            total_val += self._eval(deal, [], 0, policy)
        return total_val / len(ALL_DEALS)

    def _eval(self, dice, history, last_claim, policy):
        player = len(history) % 2
        legal = self._legal_actions(last_claim)
        if not legal:
            return 0.0

        info_set = self._info_set(dice, player, history)
        strat = policy.get(info_set)
        if strat is None:
            strat = np.zeros(NUM_ACTIONS)
            for a in legal:
                strat[a] = 1.0 / len(legal)

        val = 0.0
        for a in legal:
            if a == CHALLENGE:
                q, f = claim_to_qf(last_claim)
                actual = sum(1 for d in dice if d == f)
                if actual >= q:
                    child_val = -1.0 if player == 0 else 1.0
                else:
                    child_val = 1.0 if player == 0 else -1.0
                val += strat[a] * child_val
            else:
                val += strat[a] * self._eval(dice, history + [a], a, policy)
        return val
