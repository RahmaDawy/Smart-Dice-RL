"""RL algorithms: tabular Q-Learning, SARSA and DQN (PyTorch)."""
from algorithms.base import RandomAgent, TabularAgent
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SarsaAgent

__all__ = ["RandomAgent", "TabularAgent", "QLearningAgent", "SarsaAgent"]
# DQNAgent lives in algorithms.dqn and is imported lazily so that the rest of
# the project still works if PyTorch is not installed.
