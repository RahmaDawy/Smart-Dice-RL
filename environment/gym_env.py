"""Gymnasium-compatible wrapper around :class:`GameEngine`.

* observation: ``position - 1``  (Discrete(N), 0 = start square, N-1 = goal)
* action:      Discrete(6), action ``a`` means "roll dice value a + 1"
* reset() -> (obs, info);  step(a) -> (obs, reward, terminated, truncated, info)
"""
from typing import Optional

import gymnasium as gym
from gymnasium import spaces

from configs.board_configs import NUM_ACTIONS, BoardConfig, get_board_config
from environment.board_generator import build_board
from environment.game_engine import GameEngine, StepResult


class SmartDiceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        experiment: int = 1,
        config: Optional[BoardConfig] = None,
        slip_prob: float = 0.0,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.config = config if config is not None else get_board_config(experiment)
        self.board = build_board(self.config)
        self.engine = GameEngine(
            self.board,
            max_steps=max_steps if max_steps is not None else self.config.max_steps,
            slip_prob=slip_prob,
        )
        self.observation_space = spaces.Discrete(self.config.size)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def action_to_dice(action: int) -> int:
        return int(action) + 1

    def _info(self, result: Optional[StepResult] = None) -> dict:
        info = {
            "position": self.engine.position,
            "moves": self.engine.moves,
            "total_reward": self.engine.total_reward,
            "goal": self.board.goal,
            "max_steps": self.engine.max_steps,
        }
        if result is not None:
            info.update(
                dice=result.dice,
                requested_dice=result.requested_dice,
                slipped=result.slipped,
                event=result.event,
                previous_position=result.previous_position,
            )
        return info

    # -- Gymnasium API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        position = self.engine.reset(seed=engine_seed)
        return position - 1, self._info()

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected an integer in [0, {NUM_ACTIONS - 1}]")
        result = self.engine.step(self.action_to_dice(action))
        return result.position - 1, result.reward, result.terminated, result.truncated, self._info(result)
