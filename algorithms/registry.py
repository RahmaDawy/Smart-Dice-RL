"""Find, load and build agents by name (used by evaluation and the GUI)."""
from pathlib import Path
from typing import Optional

from algorithms.base import RandomAgent
from configs.board_configs import get_board_config
from configs.hyperparameters import get_hyperparameters
from configs.paths import MODELS_DIR

AGENT_KEYS = ("random", "q_learning", "sarsa", "dqn")
DISPLAY_NAMES = {"random": "Random", "q_learning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN"}
MODEL_SUFFIX = {"q_learning": ".npz", "sarsa": ".npz", "dqn": ".pt"}


class ModelMissingError(FileNotFoundError):
    """Raised when a trained model file does not exist yet."""


def model_path(algo: str, experiment_id: int, models_dir: Optional[Path] = None) -> Path:
    if algo not in MODEL_SUFFIX:
        raise ValueError(f"Unknown algorithm {algo!r}")
    return Path(models_dir or MODELS_DIR) / f"{algo}_exp{experiment_id}{MODEL_SUFFIX[algo]}"


def model_exists(algo: str, experiment_id: int, models_dir: Optional[Path] = None) -> bool:
    return model_path(algo, experiment_id, models_dir).exists()


def train_command(algo: str, experiment_id: int) -> str:
    return f"python run_project.py train --algo {algo} --experiment {experiment_id}"


def load_agent(algo: str, experiment_id: int, models_dir: Optional[Path] = None):
    """Load a trained agent. Raises ModelMissingError / ImportError with clear messages."""
    if algo == "random":
        return RandomAgent(seed=0)
    path = model_path(algo, experiment_id, models_dir)
    if not path.exists():
        raise ModelMissingError(
            f"No trained {DISPLAY_NAMES[algo]} model for the {get_board_config(experiment_id).size}-square "
            f"board ({path.name}). Training is required: {train_command(algo, experiment_id)}"
        )
    if algo == "dqn":
        from algorithms.dqn import DQNAgent  # lazy: needs PyTorch
        return DQNAgent.load(path)
    from algorithms.q_learning import QLearningAgent
    from algorithms.sarsa import SarsaAgent
    return (QLearningAgent if algo == "q_learning" else SarsaAgent).load(path)


def make_untrained_agent(algo: str, experiment_id: int, seed: int = 0):
    """A freshly initialised agent (epsilon = initial value), i.e. 'before training'."""
    if algo == "random":
        return RandomAgent(seed=seed)
    n_states = get_board_config(experiment_id).size
    kwargs = get_hyperparameters(algo, experiment_id)["agent"]
    if algo == "dqn":
        from algorithms.dqn import DQNAgent
        return DQNAgent(n_states, seed=seed, **kwargs)
    from algorithms.q_learning import QLearningAgent
    from algorithms.sarsa import SarsaAgent
    return (QLearningAgent if algo == "q_learning" else SarsaAgent)(n_states, seed=seed, **kwargs)
