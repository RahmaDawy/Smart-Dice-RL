"""Game environment package (rules, board generation, Gymnasium wrapper)."""
from environment.board_generator import Board, build_board, generate_board
from environment.game_engine import GameEngine, StepResult
from environment.gym_env import SmartDiceEnv

__all__ = ["Board", "build_board", "generate_board", "GameEngine", "StepResult", "SmartDiceEnv"]
