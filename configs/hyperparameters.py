"""Hyper-parameters for every algorithm and experiment.

Q-Learning and SARSA use *identical* hyper-parameters so that the comparison
between them is fair. Epsilon decays multiplicatively once per episode and is
tuned so that epsilon reaches its minimum after ~60 % of the training episodes.
"""
from copy import deepcopy
from typing import Any, Dict

ALGORITHMS = ("q_learning", "sarsa", "dqn")

TRAIN_SEED = 42          # seed used for training runs
EVAL_SEED_OFFSET = 10_000  # evaluation episodes use seeds that training never uses

# Training episodes per experiment.
EPISODES = {
    "q_learning": {1: 3000, 2: 5000, 3: 10000},
    "sarsa": {1: 3000, 2: 5000, 3: 10000},
    "dqn": {1: 1500, 2: 2500, 3: 4000},
}

# Per-episode multiplicative epsilon decay: 0.05 = decay ** (0.6 * episodes)
EPSILON_DECAY = {
    "q_learning": {1: 0.9983, 2: 0.9990, 3: 0.9995},
    "sarsa": {1: 0.9983, 2: 0.9990, 3: 0.9995},
    "dqn": {1: 0.9967, 2: 0.9980, 3: 0.99875},
}

_TABULAR_COMMON = {
    "alpha": 0.1,          # learning rate
    "gamma": 0.99,         # discount factor
    "epsilon": 1.0,        # initial exploration rate
    "epsilon_min": 0.05,   # exploration floor
}

_DQN_COMMON = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "hidden_sizes": (128, 128),
    "batch_size": 64,
    "buffer_size": 50_000,
    "learning_starts": 500,        # transitions collected before learning begins
    "train_every": 2,              # one gradient step every N environment steps
    "target_update_every": 250,    # gradient steps between target-network syncs
    "reward_scale": 0.05,          # rewards are multiplied by this inside the network loss
}


def get_hyperparameters(algo: str, experiment_id: int) -> Dict[str, Any]:
    """Return ``{"episodes": int, "agent": {...kwargs for the agent...}}``."""
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo!r}. Valid: {ALGORITHMS}")
    if experiment_id not in EPISODES[algo]:
        raise ValueError(f"Unknown experiment {experiment_id!r}")
    base = _DQN_COMMON if algo == "dqn" else _TABULAR_COMMON
    agent_kwargs = deepcopy(base)
    agent_kwargs["epsilon_decay"] = EPSILON_DECAY[algo][experiment_id]
    return {"episodes": EPISODES[algo][experiment_id], "agent": agent_kwargs}
