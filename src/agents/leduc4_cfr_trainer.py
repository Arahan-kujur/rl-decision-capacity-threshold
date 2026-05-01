"""CFR solver for Leduc4 Poker -- full game-tree enumeration (4 ranks, 12 cards)."""

import numpy as np
from collections import defaultdict

from src.env.leduc4_poker import (
    FOLD, CHECK_CALL, RAISE, NUM_ACTIONS,
    CARDS, card_rank, all_deals, RANK_NAMES, ACTION_CHARS,
)

_ALL_DEALS = all_deals()


class Leduc4CFRTrainer:
    """Vanilla CFR with full game-tree traversal for Leduc4 Poker.

    Enumerates all 1320 (p0, p1, community) deals per iteration
    (12 * 11 * 10 = 1320).
    """

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

    def train(self, iterations):
        for _ in range(iterations):
            for deal in _ALL_DEALS:
                self._cfr(deal, [], 0, 0, 1.0, 1.0)

    def _current_player(self, round_actions):
        return len(round_actions) % 2

    def _legal_actions(self, round_actions, raises):
        n = len(round_actions)
        if n == 0:
            return [CHECK_CALL, RAISE]
        last = round_actions[-1]
        if last == RAISE:
            legal = [FOLD, CHECK_CALL]
            if raises < 2:
                legal.append(RAISE)
            return legal
        if last == CHECK_CALL:
            return [CHECK_CALL, RAISE]
        return [CHECK_CALL, RAISE]

    def _is_round_over(self, round_actions):
        n = len(round_actions)
        if n < 2:
            return False
        last_two = (round_actions[-2], round_actions[-1])
        if last_two == (CHECK_CALL, CHECK_CALL):
            return True
        if last_two[0] == RAISE and last_two[1] == CHECK_CALL:
            return True
        return False

    def _has_fold(self, round_actions):
        return len(round_actions) > 0 and round_actions[-1] == FOLD

    def _compute_bets(self, r0_actions, r1_actions):
        """Compute total bets for each player from action sequences."""
        bets = [1, 1]
        for rnd, (actions, raise_size) in enumerate(
                [(r0_actions, 2), (r1_actions, 4)]):
            for i, a in enumerate(actions):
                player = i % 2
                if a == RAISE:
                    call_amount = bets[1 - player] - bets[player]
                    bets[player] += call_amount + raise_size
                elif a == CHECK_CALL:
                    call_amount = bets[1 - player] - bets[player]
                    bets[player] += call_amount
                elif a == FOLD:
                    break
        return bets

    def _info_set(self, deal, player, history_str, round_num):
        priv = card_rank(deal[player])
        if round_num >= 1:
            comm = card_rank(deal[2])
            return f"{priv},{comm}|{history_str}"
        return f"{priv}|{history_str}"

    def _history_str(self, r0_actions, r1_actions):
        parts = [ACTION_CHARS[a] for a in r0_actions]
        if r1_actions is not None:
            parts.append("/")
            parts.extend(ACTION_CHARS[a] for a in r1_actions)
        return "".join(parts)

    def _cfr(self, deal, r0_actions, round_num, r0_raises,
             p0, p1, r1_actions=None, r1_raises=0):
        """Recursive CFR walk. Returns expected value for player 0."""

        if round_num == 0:
            if self._has_fold(r0_actions):
                folder = self._current_player(r0_actions[:-1])
                bets = self._compute_bets(r0_actions, [])
                return -bets[0] if folder == 0 else bets[1]

            if self._is_round_over(r0_actions):
                return self._cfr(deal, r0_actions, 1, r0_raises,
                                 p0, p1, [], 0)

            player = self._current_player(r0_actions)
            legal = self._legal_actions(r0_actions, r0_raises)
            h_str = self._history_str(r0_actions, None)
            info_set = self._info_set(deal, player, h_str, 0)

            strategy = self._get_strategy(info_set, legal)
            util = np.zeros(NUM_ACTIONS)

            for a in legal:
                new_raises = r0_raises + (1 if a == RAISE else 0)
                if player == 0:
                    util[a] = self._cfr(deal, r0_actions + [a], 0,
                                        new_raises, p0 * strategy[a], p1)
                else:
                    util[a] = self._cfr(deal, r0_actions + [a], 0,
                                        new_raises, p0, p1 * strategy[a])

            node_util = sum(strategy[a] * util[a] for a in legal)

            opp_reach = p1 if player == 0 else p0
            sign = 1 if player == 0 else -1
            for a in legal:
                self.regret_sum[info_set][a] += \
                    opp_reach * sign * (util[a] - node_util)

            player_reach = p0 if player == 0 else p1
            self.strategy_sum[info_set] += player_reach * strategy

            return node_util

        else:
            r1_actions = r1_actions if r1_actions is not None else []

            if self._has_fold(r1_actions):
                folder = self._current_player(r1_actions[:-1])
                bets = self._compute_bets(r0_actions, r1_actions)
                return -bets[0] if folder == 0 else bets[1]

            if self._is_round_over(r1_actions):
                bets = self._compute_bets(r0_actions, r1_actions)
                r0_rank = card_rank(deal[0])
                r1_rank = card_rank(deal[1])
                comm_rank = card_rank(deal[2])
                pair0 = r0_rank == comm_rank
                pair1 = r1_rank == comm_rank
                if pair0 and not pair1:
                    return bets[0]
                if pair1 and not pair0:
                    return -bets[0]
                if r0_rank > r1_rank:
                    return bets[0]
                if r1_rank > r0_rank:
                    return -bets[0]
                return 0

            player = self._current_player(r1_actions)
            legal = self._legal_actions(r1_actions, r1_raises)
            h_str = self._history_str(r0_actions, r1_actions)
            info_set = self._info_set(deal, player, h_str, 1)

            strategy = self._get_strategy(info_set, legal)
            util = np.zeros(NUM_ACTIONS)

            for a in legal:
                new_raises = r1_raises + (1 if a == RAISE else 0)
                if player == 0:
                    util[a] = self._cfr(deal, r0_actions, 1, r0_raises,
                                        p0 * strategy[a], p1,
                                        r1_actions + [a], new_raises)
                else:
                    util[a] = self._cfr(deal, r0_actions, 1, r0_raises,
                                        p0, p1 * strategy[a],
                                        r1_actions + [a], new_raises)

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
        """Return the average strategy as {info_set: prob_array}."""
        policy = {}
        for info_set, s in self.strategy_sum.items():
            total = s.sum()
            if total > 0:
                policy[info_set] = s / total
            else:
                policy[info_set] = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        return policy

    def nash_value_p0(self):
        """Compute expected value for P0 under the average strategy."""
        policy = self.get_average_strategy()
        total_val = 0.0
        for deal in _ALL_DEALS:
            total_val += self._eval_deal(deal, policy)
        return total_val / len(_ALL_DEALS)

    def _eval_deal(self, deal, policy):
        """Evaluate a single deal under a fixed strategy profile."""
        return self._eval_node(deal, [], 0, 0, policy, None, 0)

    def _eval_node(self, deal, r0_actions, round_num, r0_raises,
                   policy, r1_actions, r1_raises):
        if round_num == 0:
            if self._has_fold(r0_actions):
                folder = self._current_player(r0_actions[:-1])
                bets = self._compute_bets(r0_actions, [])
                return -bets[0] if folder == 0 else bets[1]
            if self._is_round_over(r0_actions):
                return self._eval_node(deal, r0_actions, 1, r0_raises,
                                       policy, [], 0)
            player = self._current_player(r0_actions)
            legal = self._legal_actions(r0_actions, r0_raises)
            h_str = self._history_str(r0_actions, None)
            info_set = self._info_set(deal, player, h_str, 0)
            strat = policy.get(info_set)
            if strat is None:
                strat = np.zeros(NUM_ACTIONS)
                for a in legal:
                    strat[a] = 1.0 / len(legal)
            val = 0.0
            for a in legal:
                new_raises = r0_raises + (1 if a == RAISE else 0)
                val += strat[a] * self._eval_node(
                    deal, r0_actions + [a], 0, new_raises,
                    policy, None, 0)
            return val
        else:
            r1_actions = r1_actions if r1_actions is not None else []
            if self._has_fold(r1_actions):
                folder = self._current_player(r1_actions[:-1])
                bets = self._compute_bets(r0_actions, r1_actions)
                return -bets[0] if folder == 0 else bets[1]
            if self._is_round_over(r1_actions):
                bets = self._compute_bets(r0_actions, r1_actions)
                r0r = card_rank(deal[0])
                r1r = card_rank(deal[1])
                cr = card_rank(deal[2])
                p0_pair = r0r == cr
                p1_pair = r1r == cr
                if p0_pair and not p1_pair:
                    return bets[0]
                if p1_pair and not p0_pair:
                    return -bets[0]
                if r0r > r1r:
                    return bets[0]
                if r1r > r0r:
                    return -bets[0]
                return 0
            player = self._current_player(r1_actions)
            legal = self._legal_actions(r1_actions, r1_raises)
            h_str = self._history_str(r0_actions, r1_actions)
            info_set = self._info_set(deal, player, h_str, 1)
            strat = policy.get(info_set)
            if strat is None:
                strat = np.zeros(NUM_ACTIONS)
                for a in legal:
                    strat[a] = 1.0 / len(legal)
            val = 0.0
            for a in legal:
                new_raises = r1_raises + (1 if a == RAISE else 0)
                val += strat[a] * self._eval_node(
                    deal, r0_actions, 1, r0_raises,
                    policy, r1_actions + [a], new_raises)
            return val
