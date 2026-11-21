"""Rules of the game: movement, snakes, ladders, invalid moves, winning, max length."""
import pytest

from configs.board_configs import BOARD_CONFIGS
from environment.board_generator import Board, build_board, generate_board
from environment.game_engine import GameEngine
from evaluation.bfs_baseline import bfs_shortest_path


@pytest.fixture
def board():
    # snakes: 7->2, 20->11   ladders: 4->14, 25->27   (no chains, no shared squares)
    return Board(size=30, snakes={7: 2, 20: 11}, ladders={4: 14, 25: 27})


@pytest.fixture
def engine(board):
    eng = GameEngine(board, max_steps=50)
    eng.reset()
    return eng


def test_normal_move(engine):
    r = engine.step(1)
    assert (r.position, r.reward, r.event) == (2, -1.0, "normal")
    assert not r.done and r.moves == 1


def test_ladder_is_applied_immediately(engine):
    r = engine.step(3)  # 1 + 3 = 4 -> ladder to 14
    assert (r.position, r.reward, r.event) == (14, 20.0, "ladder")


def test_snake_is_applied_immediately(engine):
    r = engine.step(6)  # 1 + 6 = 7 -> snake to 2
    assert (r.position, r.reward, r.event) == (2, -20.0, "snake")


def test_invalid_move_stays_and_costs_a_turn(engine):
    engine.position = 28
    r = engine.step(6)  # 34 > 30
    assert (r.position, r.reward, r.event) == (28, -10.0, "invalid")
    assert r.moves == 1 and not r.terminated


def test_winning_gives_goal_reward(engine):
    engine.position = 24
    r = engine.step(6)
    assert (r.position, r.reward, r.event) == (30, 100.0, "goal")
    assert r.terminated and not r.truncated


def test_exact_roll_needed_near_goal(engine):
    engine.position = 29
    assert engine.step(2).event == "invalid"
    assert engine.step(1).event == "goal"


def test_max_episode_length_truncates(board):
    eng = GameEngine(board, max_steps=3)
    eng.reset()
    results = [eng.step(1) for _ in range(3)]
    assert not results[1].done
    assert results[2].truncated and not results[2].terminated


def test_goal_on_last_allowed_step_is_termination_not_truncation(board):
    eng = GameEngine(board, max_steps=1)
    eng.reset()
    eng.position = 29
    r = eng.step(1)
    assert r.terminated and not r.truncated


def test_step_after_done_and_bad_dice(engine):
    with pytest.raises(ValueError):
        engine.step(7)
    with pytest.raises(ValueError):
        engine.step(0)
    engine.position = 29
    engine.step(1)
    with pytest.raises(RuntimeError):
        engine.step(1)


def test_reset_restores_start(engine):
    engine.step(1)
    assert engine.reset() == 1 and engine.moves == 0 and engine.total_reward == 0


def test_total_reward_accumulates(engine):
    engine.step(1)   # -1
    engine.step(1)   # -1  (square 3)
    engine.step(1)   # square 4 -> ladder +20
    assert engine.total_reward == 18.0


@pytest.mark.parametrize("exp", list(BOARD_CONFIGS))
def test_generated_boards_follow_the_rules(exp):
    cfg = BOARD_CONFIGS[exp]
    b = build_board(cfg)
    assert len(b.snakes) == cfg.num_snakes and len(b.ladders) == cfg.num_ladders
    starts = list(b.snakes) + list(b.ladders)
    ends = list(b.snakes.values()) + list(b.ladders.values())
    assert len(set(starts)) == len(starts)                 # no square is both snake and ladder start
    assert not set(starts) & set(ends)                      # no chained transitions
    assert all(2 <= s <= cfg.size - 1 for s in starts + ends)
    assert all(h > t for h, t in b.snakes.items()) and all(t > bt for bt, t in b.ladders.items())
    assert bfs_shortest_path(b) is not None                 # solvable


def test_board_generation_is_reproducible():
    assert generate_board(50, 5, 5, seed=7) == generate_board(50, 5, 5, seed=7)
    assert generate_board(50, 5, 5, seed=7) != generate_board(50, 5, 5, seed=8)


def test_bfs_minimum_moves_on_plain_board():
    assert bfs_shortest_path(Board(size=13))["min_moves"] == 2   # 12 squares / 6 per roll
