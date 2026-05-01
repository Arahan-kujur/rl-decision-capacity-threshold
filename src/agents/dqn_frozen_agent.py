"""DQN agent with freeze support -- stops learning after perturbation."""

from src.agents.dqn_agent import DQNAgent


class DQNFrozenAgent(DQNAgent):
    """DQN that stops updating after freeze() is called.

    Same pattern as QLearningFrozenAgent but for neural networks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._was_frozen = False

    def freeze(self):
        self._frozen = True
        self._was_frozen = True

    def update(self, trajectory, reward_p0):
        if not self._frozen:
            super().update(trajectory, reward_p0)
