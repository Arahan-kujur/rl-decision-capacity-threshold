"""Kuhn Poker environment with configurable asymmetric action perturbation."""

import numpy as np

PASS = 0
BET = 1
NUM_ACTIONS = 2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}


class KuhnPokerEnv:
    """Kuhn Poker: 3 cards (J<Q<K), 2 players, ante 1, bet 1.

    Game tree (P=pass, B=bet):
      P0: P or B
        P -> P1: P(showdown +/-1) or B -> P0: P(fold,-1) or B(call, showdown +/-2)
        B -> P1: P(fold,+1) or B(call, showdown +/-2)
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self._cards = [0, 0]
        self._history = []
        self._done = False
        self._reward_p0 = 0

    def reset(self):
        deck = np.array([0, 1, 2])
        self.rng.shuffle(deck)
        self._cards = [int(deck[0]), int(deck[1])]
        self._history = []
        self._done = False
        self._reward_p0 = 0
        return self

    @property
    def current_player(self):
        n = len(self._history)
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 0
        return -1

    @property
    def is_root(self):
        return len(self._history) == 0

    def info_state_str(self, player):
        card = self._cards[player]
        h = "".join("p" if a == PASS else "b" for a in self._history)
        return f"{card}{h}"

    def legal_actions(self):
        return [PASS, BET]

    @property
    def is_terminal(self):
        return self._done

    @property
    def returns(self):
        return [self._reward_p0, -self._reward_p0]

    def step(self, action):
        assert not self._done, "step() called on terminal state"
        self._history.append(action)
        h = tuple(self._history)

        showdown = 1 if self._cards[0] > self._cards[1] else -1
        terminal_payoffs = {
            (PASS, PASS): showdown,
            (BET, PASS): 1,
            (BET, BET): 2 * showdown,
            (PASS, BET, PASS): -1,
            (PASS, BET, BET): 2 * showdown,
        }

        if h in terminal_payoffs:
            self._done = True
            self._reward_p0 = terminal_payoffs[h]


class PerturbedKuhnPoker:
    """Wrapper that filters an action for one player under perturbation.

    Two modes controlled by `root_only`:
      False  ->  strip action at ALL of the affected player's decision nodes
      True   ->  strip action ONLY at the opening move (root), preserving
                 call/fold decisions at later nodes like "pb"
    """

    def __init__(self, env, removed_action=BET, affected_player=0,
                 root_only=False):
        self.env = env
        self.perturbed = False
        self.removed_action = removed_action
        self.affected_player = affected_player
        self.root_only = root_only

    def set_perturbed(self, flag):
        self.perturbed = flag

    def reset(self):
        self.env.reset()
        return self

    @property
    def current_player(self):
        return self.env.current_player

    def info_state_str(self, player):
        return self.env.info_state_str(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if self.perturbed and self.env.current_player == self.affected_player:
            if not self.root_only or self.env.is_root:
                filtered = [a for a in actions if a != self.removed_action]
                if filtered:
                    return filtered
        return actions

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    def step(self, action):
        self.env.step(action)
