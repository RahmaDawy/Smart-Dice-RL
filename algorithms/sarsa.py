"""Tabular SARSA (on-policy TD control)."""
from algorithms.base import TabularAgent


class SarsaAgent(TabularAgent):
    """Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a)).

    ``a'`` is the action the epsilon-greedy policy actually selects in s'
    (on-policy), so exploration risk (e.g. snakes) is reflected in the values.
    """

    name = "sarsa"

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int, terminated: bool
    ) -> float:
        target = reward if terminated else reward + self.gamma * self.q_table[next_state, next_action]
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)
