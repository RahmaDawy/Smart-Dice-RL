"""Tests for the optional extensions (random dice, dynamic board, multiplayer, hold())."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.board_generator import Board
from environment.game_engine import GameEngine
from extensions.dynamic_board import FEATURE_DIM, DynamicBoardEnv, board_features, evaluate_dynamic, train_dynamic
from extensions.multiplayer import play_race, tournament
from extensions.random_dice import RandomDiceEnv, evaluate_policy, always_move, train_random_dice


def test_engine_hold_costs_a_turn_and_keeps_position():
    eng = GameEngine(Board(size=30), max_steps=3)
    eng.reset()
    r = eng.hold()
    assert (r.position, r.reward, r.event, r.moves) == (1, -1.0, "hold", 1)
    eng.hold()
    assert eng.hold().truncated


# --- random dice ---------------------------------------------------------------------
def test_random_dice_env_api():
    env = RandomDiceEnv(experiment=1)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs) and env.action_space.n == 2 and 1 <= info["dice"] <= 6
    obs, reward, term, trunc, info = env.step(1)       # hold
    assert info["position"] == 1 and reward == -1.0
    with pytest.raises(ValueError):
        env.step(2)
    check_env(RandomDiceEnv(experiment=1), skip_render_check=True)


def test_random_dice_is_stochastic_but_reproducible():
    def rolls(seed):
        env = RandomDiceEnv(experiment=1)
        env.reset(seed=seed)
        return [env.step(1)[4]["next_dice"] for _ in range(10)]
    assert rolls(1) == rolls(1) and rolls(1) != rolls(2)


def test_random_dice_training_and_eval_smoke(tmp_path):
    agent, df = train_random_dice("q_learning", 1, episodes=200, models_dir=tmp_path, csv_dir=tmp_path, verbose=False)
    assert len(df) == 200 and agent.q_table.shape == (30 * 6, 2)
    assert (tmp_path / "ext_random_dice_q_learning_exp1.npz").exists()
    stats = evaluate_policy(RandomDiceEnv(1), always_move, 20, seed=0)
    assert 0 <= stats["success_rate"] <= 1 and stats["hold_rate"] == 0


# --- dynamic board -------------------------------------------------------------------
def test_dynamic_env_changes_board_each_episode_and_is_reproducible():
    env = DynamicBoardEnv(experiment=1)
    seeds = []
    for _ in range(5):
        obs, info = env.reset()
        assert obs.shape == (FEATURE_DIM,) and env.observation_space.contains(obs)
        seeds.append(info["board_seed"])
    assert len(set(seeds)) > 1
    env.reset(options={"board_seed": 5}); a = dict(env.board.snakes)
    env.reset(options={"board_seed": 5}); assert a == env.board.snakes
    check_env(DynamicBoardEnv(experiment=1), skip_render_check=True)


def test_board_features_describe_outcomes():
    board = Board(size=30, snakes={7: 2}, ladders={4: 14})
    f = board_features(board, 1)
    assert f.shape == (FEATURE_DIM,)
    # dice 3 -> ladder (index 2 of the one-hot), dice 6 -> snake (index 1)
    assert f[(3 - 1) * 6 + 2] == 1.0 and f[(6 - 1) * 6 + 1] == 1.0


def test_dynamic_training_and_heldout_eval_smoke(tmp_path):
    agent, df = train_dynamic(1, episodes=80, models_dir=tmp_path, csv_dir=tmp_path, verbose=False)
    assert len(df) == 80
    table = evaluate_dynamic(1, boards=5, models_dir=tmp_path)
    assert "BFS optimum (reference)" in set(table["policy"])
    assert (table["success_rate"].between(0, 1)).all()


# --- multiplayer ---------------------------------------------------------------------
class _FixedDice:
    def __init__(self, dice):
        self.dice = dice
    def act(self, state, greedy=True):
        return self.dice - 1


def test_race_first_mover_wins_ties_and_alternates():
    board = Board(size=13)                      # two moves of 6 are enough
    a, b = _FixedDice(6), _FixedDice(6)
    assert play_race(board, [a, b], 20, first=0)["winner"] == 0
    assert play_race(board, [a, b], 20, first=1)["winner"] == 1
    res = tournament(_FixedDice(6), _FixedDice(6), 1, games=10)
    assert res["wins_a"] + res["wins_b"] + res["draws"] == 10


def test_race_draw_when_nobody_finishes():
    board = Board(size=30)
    assert play_race(board, [_FixedDice(1), _FixedDice(1)], 3)["winner"] is None


def test_race_faster_player_wins():
    board = Board(size=31)                      # 1 + 5*6 = 31: dice 6 finishes in 5 moves, dice 1 needs 30
    assert play_race(board, [_FixedDice(1), _FixedDice(6)], 50, first=0)["winner"] == 1
