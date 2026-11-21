"""Smoke tests for training, evaluation, model saving/loading and metrics."""
import numpy as np
import pytest

from algorithms.base import RandomAgent
from algorithms.q_learning import QLearningAgent
from algorithms.registry import ModelMissingError, load_agent, model_path
from algorithms.sarsa import SarsaAgent
from evaluation.evaluate import evaluate_agent
from evaluation.metrics import convergence_episode, moving_average
from training.common import train


@pytest.mark.parametrize("algo, cls", [("q_learning", QLearningAgent), ("sarsa", SarsaAgent)])
def test_tabular_training_saves_and_loads(tmp_path, algo, cls):
    agent, df = train(algo, 1, episodes=40, models_dir=tmp_path, csv_dir=tmp_path, verbose=False)
    assert len(df) == 40 and {"reward", "moves", "success", "epsilon", "elapsed_seconds"} <= set(df.columns)
    assert (tmp_path / f"training_{algo}_exp1.csv").exists()
    path = model_path(algo, 1, tmp_path)
    assert path.exists()
    loaded = cls.load(path)
    np.testing.assert_array_equal(loaded.q_table, agent.q_table)
    assert loaded.alpha == agent.alpha and loaded.gamma == agent.gamma
    assert [loaded.act(s) for s in range(30)] == [agent.act(s) for s in range(30)]
    assert type(load_agent(algo, 1, tmp_path)) is cls


def test_dqn_training_saves_and_loads(tmp_path):
    pytest.importorskip("torch")
    from algorithms.dqn import DQNAgent
    agent, df = train("dqn", 1, episodes=120, models_dir=tmp_path, csv_dir=tmp_path, verbose=False)
    assert len(df) == 120 and agent.learn_steps > 0
    loaded = DQNAgent.load(model_path("dqn", 1, tmp_path))
    np.testing.assert_allclose(loaded.q_values(5), agent.q_values(5), rtol=1e-5, atol=1e-6)
    assert [loaded.act(s) for s in range(30)] == [agent.act(s) for s in range(30)]


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelMissingError, match="Training is required"):
        load_agent("q_learning", 2, tmp_path)


def test_q_learning_learns_on_the_30_square_board(tmp_path):
    agent, _ = train("q_learning", 1, models_dir=tmp_path, csv_dir=tmp_path, verbose=False)
    result = evaluate_agent(agent, 1, episodes=10)
    assert result["success"].mean() == 1.0
    random_result = evaluate_agent(RandomAgent(seed=0), 1, episodes=50)
    assert result["reward"].mean() > random_result["reward"].mean()


def test_training_is_reproducible(tmp_path):
    _, a = train("q_learning", 1, episodes=50, models_dir=tmp_path, csv_dir=tmp_path, verbose=False, seed=5)
    _, b = train("q_learning", 1, episodes=50, models_dir=tmp_path, csv_dir=tmp_path, verbose=False, seed=5)
    assert a["reward"].tolist() == b["reward"].tolist()


def test_moving_average_and_convergence():
    assert moving_average([1, 2, 3, 4], 2).tolist() == [1.0, 1.5, 2.5, 3.5]
    rewards = list(np.linspace(-100, 100, 300)) + [100.0] * 400
    conv = convergence_episode(rewards)
    assert conv is not None and 300 <= conv <= 450
    assert convergence_episode([0.0] * 50) is None   # too short to judge
