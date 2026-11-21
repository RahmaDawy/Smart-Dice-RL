"""Core game engine: the single implementation of the game rules + rewards.

The Gymnasium wrapper, the evaluation code, the BFS baseline and the Pygame
GUI all drive this class, so there is exactly one version of the rules.
"""
import random
from dataclasses import dataclass
from typing import Optional

from configs.board_configs import (
    REWARD_GOAL,
    REWARD_INVALID,
    REWARD_LADDER,
    REWARD_NORMAL,
    REWARD_SNAKE,
)
from environment.board_generator import (
    DICE_VALUES,
    EVENT_GOAL,
    EVENT_INVALID,
    EVENT_LADDER,
    EVENT_SNAKE,
    Board,
)

EVENT_REWARDS = {
    "normal": REWARD_NORMAL,
    EVENT_SNAKE: REWARD_SNAKE,
    EVENT_LADDER: REWARD_LADDER,
    EVENT_GOAL: REWARD_GOAL,
    EVENT_INVALID: REWARD_INVALID,
    "hold": REWARD_NORMAL,   # extension only: skipping a turn costs like a normal square
}


@dataclass(frozen=True)
class StepResult:
    """Everything that happened during one turn."""

    previous_position: int
    position: int
    requested_dice: int     # dice value the agent/human asked for
    dice: int               # dice value actually applied (differs only if "slipped")
    slipped: bool
    event: str              # normal | snake | ladder | goal | invalid
    reward: float
    terminated: bool        # goal reached
    truncated: bool         # maximum episode length reached without winning
    moves: int              # turns taken so far (invalid moves count)
    total_reward: float

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class GameEngine:
    """One game of Smart Dice.

    ``slip_prob`` is an optional robustness feature (default 0.0 = the
    deterministic core project): with that probability the dice shows a random
    value instead of the chosen one.
    """

    def __init__(self, board: Board, max_steps: int, slip_prob: float = 0.0, seed: Optional[int] = None):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError("slip_prob must be in [0, 1]")
        self.board = board
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self._rng = random.Random(seed)
        self.position = board.start
        self.moves = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self, seed: Optional[int] = None) -> int:
        """Start a new game; returns the start position (square 1)."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.position = self.board.start
        self.moves = 0
        self.total_reward = 0.0
        self.done = False
        return self.position

    def step(self, dice: int) -> StepResult:
        """Play one turn with the requested dice value (1..6)."""
        if self.done:
            raise RuntimeError("episode is finished; call reset() first")
        if dice not in DICE_VALUES:
            raise ValueError(f"dice must be an integer from 1 to 6, got {dice!r}")

        applied = dice
        slipped = False
        if self.slip_prob > 0.0 and self._rng.random() < self.slip_prob:
            applied = self._rng.choice(DICE_VALUES)
            slipped = applied != dice

        previous = self.position
        new_position, event = self.board.resolve(previous, applied)
        reward = EVENT_REWARDS[event]

        self.position = new_position
        self.moves += 1                       # invalid moves consume a turn too
        self.total_reward += reward
        terminated = event == EVENT_GOAL
        truncated = (not terminated) and self.moves >= self.max_steps
        self.done = terminated or truncated

        return StepResult(
            previous_position=previous,
            position=new_position,
            requested_dice=dice,
            dice=applied,
            slipped=slipped,
            event=event,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            moves=self.moves,
            total_reward=self.total_reward,
        )

    def hold(self) -> StepResult:
        """Skip the move (used only by the random-dice extension).

        The token stays, the turn is consumed and the normal-square cost applies.
        The core game never calls this method.
        """
        if self.done:
            raise RuntimeError("episode is finished; call reset() first")
        reward = EVENT_REWARDS["hold"]
        self.moves += 1
        self.total_reward += reward
        truncated = self.moves >= self.max_steps
        self.done = truncated
        return StepResult(
            previous_position=self.position, position=self.position, requested_dice=0, dice=0, slipped=False,
            event="hold", reward=reward, terminated=False, truncated=truncated, moves=self.moves,
            total_reward=self.total_reward,
        )
