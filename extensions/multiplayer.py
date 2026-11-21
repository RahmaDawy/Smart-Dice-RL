"""Extension 3 - MULTIPLE PLAYERS (a race on the same board).

Two players alternate turns on the SAME board, each with their own token and
their own GameEngine (same rules, same rewards). The first token to reach the
goal wins. Tokens do not interact (no capturing). If neither player reaches the
goal within the maximum episode length it is a draw.

Players are any objects with ``act(state, greedy)`` (the trained agents, the
random agent). In the Pygame game (mode "Race vs AI") a human plays against an
agent with the same code path.

Note: the core game is deterministic and a trained agent plays the same optimal
line every time, so two equally good agents tie on moves and the player who
starts wins - games therefore alternate who starts.
"""
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from algorithms.registry import AGENT_KEYS, DISPLAY_NAMES, ModelMissingError, load_agent
from configs.board_configs import BOARD_CONFIGS, get_board_config
from configs.paths import CSV_DIR, ensure_dirs
from environment.board_generator import Board, build_board
from environment.game_engine import GameEngine


def play_race(board: Board, players: List, max_steps: int, first: int = 0) -> dict:
    """Play one two-player race. Returns winner (0/1/None), moves and rewards per player."""
    engines = [GameEngine(board, max_steps) for _ in players]
    for e in engines:
        e.reset()
    turn, winner = first, None
    while winner is None and not all(e.done for e in engines):
        engine = engines[turn]
        if not engine.done:
            action = players[turn].act(engine.position - 1, greedy=True)
            if engine.step(int(action) + 1).terminated:
                winner = turn
        turn = 1 - turn
    return {"winner": winner, "moves": [e.moves for e in engines], "rewards": [e.total_reward for e in engines]}


def tournament(agent_a, agent_b, experiment_id: int, games: int = 100) -> dict:
    """Alternate the starting player; returns win counts for A and B."""
    cfg = get_board_config(experiment_id)
    board = build_board(cfg)
    wins = [0, 0]
    draws = 0
    moves = [0.0, 0.0]
    for g in range(games):
        first = g % 2
        result = play_race(board, [agent_a, agent_b], cfg.max_steps, first=first)
        if result["winner"] is None:
            draws += 1
        else:
            wins[result["winner"]] += 1
        moves[0] += result["moves"][0]
        moves[1] += result["moves"][1]
    return {"games": games, "wins_a": wins[0], "wins_b": wins[1], "draws": draws,
            "avg_moves_a": moves[0] / games, "avg_moves_b": moves[1] / games}


def run_multiplayer(experiments: Iterable[int] = tuple(BOARD_CONFIGS), games: int = 100, models_dir=None,
                    csv_dir=None, save: bool = True) -> pd.DataFrame:
    """Round-robin between all available agents (random, Q-Learning, SARSA, DQN)."""
    rows = []
    for exp in experiments:
        agents = {}
        for key in AGENT_KEYS:
            try:
                agents[key] = load_agent(key, exp, models_dir)
            except (ModelMissingError, ImportError) as err:
                print(f"[skip] {key} / exp {exp}: {err}")
        keys = list(agents)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                stats = tournament(agents[a], agents[b], exp, games)
                rows.append({"experiment": exp, "board_size": BOARD_CONFIGS[exp].size,
                             "agent_a": DISPLAY_NAMES[a], "agent_b": DISPLAY_NAMES[b], **stats})
    table = pd.DataFrame(rows)
    if save and not table.empty:
        ensure_dirs()
        table.to_csv(Path(csv_dir or CSV_DIR) / "extension_multiplayer.csv", index=False)
    return table
