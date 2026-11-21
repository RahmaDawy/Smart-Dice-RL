"""Deep Q-Network (PyTorch): replay buffer + target network + epsilon-greedy.

The state is the board position, fed to the network as a one-hot vector
of length N. CPU is used by default; CUDA is optional and never required.
"""
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.board_configs import NUM_ACTIONS


class QNetwork(nn.Module):
    """Small MLP: one-hot position -> Q-value for each of the 6 dice values."""

    def __init__(self, n_states: int, n_actions: int, hidden_sizes: Sequence[int] = (128, 128)):
        super().__init__()
        layers, last = [], n_states
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = np.zeros(capacity, dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros(capacity, dtype=np.int64)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done) -> None:
        i = self.index
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i] = next_state, float(done)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        n_states: int,
        n_actions: int = NUM_ACTIONS,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.998,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        learning_starts: int = 500,
        train_every: int = 2,
        target_update_every: int = 250,
        reward_scale: float = 0.05,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.reward_scale = reward_scale
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if device == "auto" else device))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = QNetwork(n_states, n_actions, self.hidden_sizes).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    # -- helpers -------------------------------------------------------------
    def _encode(self, states) -> torch.Tensor:
        idx = torch.as_tensor(states, dtype=torch.long, device=self.device)
        return F.one_hot(idx, num_classes=self.n_states).float()

    def q_values(self, state: int) -> np.ndarray:
        with torch.no_grad():
            return self.policy_net(self._encode([state]))[0].cpu().numpy()

    # -- policy ------------------------------------------------------------
    def act(self, state: int, greedy: bool = True) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def decay_epsilon(self) -> None:
        self.epsilon = float(max(self.epsilon_min, self.epsilon * self.epsilon_decay))

    # -- learning ------------------------------------------------------------
    def remember(self, state, action, reward, next_state, terminated) -> None:
        """Store a transition. Only real terminations (goal) cut off bootstrapping."""
        self.buffer.push(state, action, reward, next_state, terminated)

    def learn(self) -> Optional[float]:
        """One gradient step on a random mini-batch; returns the loss (or None)."""
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        states = self._encode(s)
        next_states = self._encode(s2)
        actions = torch.as_tensor(a, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(r, device=self.device) * self.reward_scale
        dones = torch.as_tensor(d, device=self.device)

        q_pred = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * (1.0 - dones) * q_next
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    # -- persistence -------------------------------------------------------
    def hyperparameters(self) -> dict:
        """Plain Python numbers only, so checkpoints load with torch.load(weights_only=True)."""
        return {
            "hidden_sizes": [int(h) for h in self.hidden_sizes], "lr": float(self.lr), "gamma": float(self.gamma),
            "epsilon": float(self.epsilon), "epsilon_min": float(self.epsilon_min),
            "epsilon_decay": float(self.epsilon_decay),
            "batch_size": int(self.batch_size), "buffer_size": int(self.buffer_size),
            "learning_starts": int(self.learning_starts), "train_every": int(self.train_every),
            "target_update_every": int(self.target_update_every), "reward_scale": float(self.reward_scale),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save the trained weights (plus what is needed to rebuild the network)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.name,
                "n_states": int(self.n_states),
                "n_actions": int(self.n_actions),
                "hyperparameters": self.hyperparameters(),
                "state_dict": {k: v.cpu() for k, v in self.policy_net.state_dict().items()},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("algorithm") != cls.name:
            raise ValueError(f"{path.name} is not a DQN checkpoint")
        agent = cls(checkpoint["n_states"], checkpoint["n_actions"], device=device, **checkpoint["hyperparameters"])
        agent.policy_net.load_state_dict(checkpoint["state_dict"])
        agent.target_net.load_state_dict(checkpoint["state_dict"])
        agent.policy_net.eval()
        return agent
