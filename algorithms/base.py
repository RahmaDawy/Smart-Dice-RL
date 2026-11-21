"""Shared agent code: epsilon-greedy tabular base class and a random agent.

Every agent exposes the same minimal interface used by evaluation and the GUI:

    action = agent.act(state, greedy=True)   # state = position - 1, action in 0..5
"""
import json
from pathlib import Path
from typing import Union

import numpy as np

from configs.board_configs import NUM_ACTIONS


class RandomAgent:
    """Picks a uniformly random dice value (the 'untrained' baseline)."""

    name = "random"

    def __init__(self, n_actions: int = NUM_ACTIONS, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, greedy: bool = True) -> int:
        return int(self.rng.integers(self.n_actions))


class TabularAgent:
    """Q-table + epsilon-greedy policy shared by Q-Learning and SARSA."""

    name = "tabular"

    def __init__(
        self,
        n_states: int,
        n_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed=None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # -- policy ------------------------------------------------------------
    def act(self, state: int, greedy: bool = True) -> int:
        """Epsilon-greedy when ``greedy=False``; pure argmax when ``greedy=True``."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = self.q_table[state]
        if greedy:
            return int(np.argmax(values))  # deterministic tie-break (lowest dice)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))  # random tie-break while learning

    def decay_epsilon(self) -> None:
        self.epsilon = float(max(self.epsilon_min, self.epsilon * self.epsilon_decay))

    # -- persistence -------------------------------------------------------
    def hyperparameters(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save the Q-table (and hyper-parameters) to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"algorithm": self.name, "hyperparameters": self.hyperparameters()})
        np.savez(path, q_table=self.q_table, meta=np.array(meta))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load an agent previously written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            q_table = data["q_table"]
            meta = json.loads(str(data["meta"]))
        if meta.get("algorithm") != cls.name:
            raise ValueError(f"{path.name} holds a {meta.get('algorithm')!r} model, expected {cls.name!r}")
        agent = cls(n_states=q_table.shape[0], n_actions=q_table.shape[1], **meta["hyperparameters"])
        agent.q_table = q_table.astype(np.float64)
        return agent
