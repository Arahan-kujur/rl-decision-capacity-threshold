"""DQN agent for poker and matrix games -- compatible with existing runner interface."""

import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Feature encoders
# ---------------------------------------------------------------------------

def _encode_kuhn(info_state_str, max_history=4):
    """Encode Kuhn info state to fixed-size vector.

    Format: "{card}{history}" e.g. "1pb"
    Output: card one-hot (3) + history one-hot per position (2 * max_history)
    """
    card = int(info_state_str[0])
    history = info_state_str[1:]

    features = np.zeros(3 + 2 * max_history, dtype=np.float32)
    features[card] = 1.0

    for i, ch in enumerate(history[:max_history]):
        offset = 3 + i * 2
        if ch == "p":
            features[offset] = 1.0
        elif ch == "b":
            features[offset + 1] = 1.0

    return features


def _encode_leduc(info_state_str, max_history=10):
    """Encode Leduc info state to fixed-size vector.

    Format: "{priv_rank},{comm_rank}|{history}" or "{priv_rank}|{history}"
    Output: priv one-hot (3) + comm one-hot (3) + comm_present (1) +
            history one-hot per position (3 * max_history)
    """
    parts = info_state_str.split("|")
    card_part = parts[0]
    history = parts[1] if len(parts) > 1 else ""

    cards = card_part.split(",")
    priv_rank = int(cards[0])
    comm_rank = int(cards[1]) if len(cards) > 1 else -1

    feat_size = 3 + 3 + 1 + 3 * max_history
    features = np.zeros(feat_size, dtype=np.float32)

    features[priv_rank] = 1.0
    if comm_rank >= 0:
        features[3 + comm_rank] = 1.0
        features[6] = 1.0

    action_map = {"f": 0, "c": 1, "r": 2, "/": -1}
    pos = 0
    for ch in history[:max_history]:
        if ch == "/":
            continue
        offset = 7 + pos * 3
        if ch in action_map and action_map[ch] >= 0:
            features[offset + action_map[ch]] = 1.0
        pos += 1

    return features


def _encode_generic(info_state_str, max_len=20):
    """Fallback encoder: hash-based embedding for unknown game formats."""
    features = np.zeros(max_len, dtype=np.float32)
    for i, ch in enumerate(info_state_str[:max_len]):
        features[i] = ord(ch) / 128.0
    return features


def _encode_liars_dice(info_state_str, max_history=20):
    """Encode Liar's Dice info state. Format: 'd{d1}{d2}|{history}'."""
    parts = info_state_str.split("|")
    dice_part = parts[0]
    history = parts[1] if len(parts) > 1 else ""

    d1 = int(dice_part[1]) if len(dice_part) > 1 else 0
    d2 = int(dice_part[2]) if len(dice_part) > 2 else 0

    features = np.zeros(12 + max_history, dtype=np.float32)
    if 1 <= d1 <= 6:
        features[d1 - 1] = 1.0
    if 1 <= d2 <= 6:
        features[6 + d2 - 1] = 1.0

    claims = history.split(",") if history else []
    for i, c in enumerate(claims[:max_history]):
        if c == "X":
            features[12 + i] = 1.0
        else:
            try:
                q, f = int(c[0]), int(c[2])
                features[12 + i] = (q * 6 + f) / 30.0
            except (IndexError, ValueError):
                pass

    return features


def get_encoder(game_name):
    """Return (encoder_fn, input_dim) for the given game."""
    if game_name == "kuhn":
        return _encode_kuhn, 3 + 2 * 4
    elif game_name == "leduc":
        return _encode_leduc, 3 + 3 + 1 + 3 * 10
    elif game_name in ("liars_dice", "liars_dice2"):
        return _encode_liars_dice, 12 + 20
    else:
        return _encode_generic, 20


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        dones = np.array([b[3] for b in batch], dtype=np.float32)
        return states, actions, rewards, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """Deep Q-Network agent compatible with the tabular agent interface.

    Uses a small MLP with experience replay and target network.
    For short poker episodes, uses MC-style terminal reward (no bootstrapping).
    """

    def __init__(self, num_actions=3, game="kuhn", lr=1e-3,
                 epsilon_start=0.15, epsilon_end=0.01, epsilon_decay=50000,
                 buffer_size=10000, batch_size=32, target_update=500,
                 hidden_size=64):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DQNAgent. "
                              "Install with: pip install torch")

        self.num_actions = num_actions
        self.game = game
        self.encoder, self.input_dim = get_encoder(game)

        self.epsilon = epsilon_start
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._step_count = 0

        self.batch_size = batch_size
        self.target_update = target_update
        self._episode_count = 0

        self.q_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.target_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._frozen = False

    def _get_epsilon(self):
        if self._frozen:
            return 0.0
        frac = min(1.0, self._step_count / self._epsilon_decay)
        return self._epsilon_start + frac * (self._epsilon_end - self._epsilon_start)

    def select_action(self, info_state, legal_actions, rng):
        eps = self._get_epsilon()
        if rng.random() < eps:
            return int(rng.choice(legal_actions))

        state = self.encoder(info_state)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state).unsqueeze(0))[0]

        masked = torch.full((self.num_actions,), -1e9)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(masked.argmax().item())

    def freeze(self):
        self._frozen = True

    def update(self, trajectory, reward_p0):
        if self._frozen:
            return

        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            state = self.encoder(info_state)
            self.buffer.push(state, action, r, True)
            self._step_count += 1

        self._episode_count += 1

        if len(self.buffer) >= self.batch_size:
            self._train_step()

        if self._episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _train_step(self):
        states, actions, rewards, _ = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)

        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)
        loss = nn.functional.mse_loss(q_values, rewards_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
