"""Tabular Q-Learning (off-policy TD control)."""
import numpy as np

from algorithms.base import TabularAgent


class QLearningAgent(TabularAgent):
    """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).

    The target uses the *greedy* next action, regardless of what the
    epsilon-greedy behaviour policy will actually do (off-policy).
    """

    name = "q_learning"

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> float:
        """One TD update. ``terminated`` (goal reached) stops bootstrapping;
        a time-limit truncation does NOT, because the game itself did not end."""
        target = reward if terminated else reward + self.gamma * float(np.max(self.q_table[next_state]))
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)
