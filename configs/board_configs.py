"""Board configurations and the reward function (single source of truth).

The three experiments required by the assignment are defined here. Every
module (training, evaluation, GUI, tests) builds its board from these
configs, so the game rules can never diverge between training and playing.
"""
from dataclasses import dataclass
from typing import Dict

# ----------------------------------------------------------------------------
# Reward function (professor's specification)
# ----------------------------------------------------------------------------
# Precedence when several things could apply to one move:
#   1. invalid move (overshoot)  -> -10   (token does not move, turn consumed)
#   2. reaching the goal square  -> +100  (always wins over any other reward)
#   3. ladder                    -> +20
#   4. snake                     -> -20
#   5. normal square             -> -1
# Ladders never end on the goal square, so the goal reward can never be
# "hidden" behind a ladder bonus.
REWARD_GOAL = 100.0
REWARD_LADDER = 20.0
REWARD_NORMAL = -1.0
REWARD_SNAKE = -20.0
REWARD_INVALID = -10.0

# The agent picks a dice value 1..6, so there are 6 discrete actions
# (action index a  <->  dice value a + 1).
NUM_ACTIONS = 6


@dataclass(frozen=True)
class BoardConfig:
    """Everything needed to rebuild a board and an episode deterministically."""

    experiment_id: int
    size: int            # number of squares N
    num_snakes: int
    num_ladders: int
    seed: int            # seed used by the board generator (fixed => fair comparison)
    max_steps: int       # maximum episode length (episode is truncated afterwards)

    @property
    def name(self) -> str:
        return f"Experiment {self.experiment_id} ({self.size} squares)"


BOARD_CONFIGS: Dict[int, BoardConfig] = {
    1: BoardConfig(experiment_id=1, size=30, num_snakes=3, num_ladders=3, seed=101, max_steps=20),
    2: BoardConfig(experiment_id=2, size=50, num_snakes=5, num_ladders=5, seed=202, max_steps=30),
    3: BoardConfig(experiment_id=3, size=100, num_snakes=8, num_ladders=8, seed=303, max_steps=60),
}

SIZE_TO_EXPERIMENT: Dict[int, int] = {cfg.size: k for k, cfg in BOARD_CONFIGS.items()}


def get_board_config(experiment_id: int) -> BoardConfig:
    """Return the config for experiment 1, 2 or 3 (raises a clear error otherwise)."""
    try:
        return BOARD_CONFIGS[int(experiment_id)]
    except (KeyError, ValueError, TypeError):
        valid = ", ".join(str(k) for k in BOARD_CONFIGS)
        raise ValueError(f"Unknown experiment {experiment_id!r}. Valid experiments: {valid}") from None
