"""Fixed opponent agents for regime variation experiments."""

import numpy as np


class RandomAgent:
    """Uniformly random action selection."""

    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def select_action(self, info_state, legal_actions, rng):
        return int(rng.choice(legal_actions))

    def update(self, trajectory, reward_p0):
        pass


class ExploitativeAgent:
    """Always plays the last (highest-index) legal action.

    In Kuhn: always bets. In Leduc: always raises.
    Represents maximum aggression.
    """

    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def select_action(self, info_state, legal_actions, rng):
        return int(max(legal_actions))

    def update(self, trajectory, reward_p0):
        pass


class NashAgent:
    """Plays a known Nash equilibrium mixed strategy.

    Used for matrix games where Nash is analytically known.
    """

    def __init__(self, strategy, num_actions=2):
        """
        Parameters
        ----------
        strategy : dict or array-like
            If dict: maps info_state -> probability array.
            If array: single mixed strategy used for all states.
        """
        self.num_actions = num_actions
        if isinstance(strategy, dict):
            self._strategy_dict = strategy
            self._default = None
        else:
            self._strategy_dict = None
            self._default = np.asarray(strategy)

    def select_action(self, info_state, legal_actions, rng):
        if self._strategy_dict is not None:
            probs = self._strategy_dict.get(info_state)
            if probs is not None:
                mask = np.zeros(self.num_actions)
                for a in legal_actions:
                    mask[a] = probs[a]
                total = mask.sum()
                if total > 0:
                    return int(rng.choice(self.num_actions, p=mask / total))
        elif self._default is not None:
            mask = np.zeros(self.num_actions)
            for a in legal_actions:
                mask[a] = self._default[a]
            total = mask.sum()
            if total > 0:
                return int(rng.choice(self.num_actions, p=mask / total))
        return int(rng.choice(legal_actions))

    def update(self, trajectory, reward_p0):
        pass
