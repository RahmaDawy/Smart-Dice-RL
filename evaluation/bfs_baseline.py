"""BFS baseline (NOT reinforcement learning).

BFS searches the full, known, deterministic state graph and returns the
minimum number of moves from the start to the goal. It is used ONLY as a
reference ("what is the best possible number of moves?"). The RL agents never
call this module and never use a precomputed path: they learn from rewards.

Limitations of BFS
------------------
* It needs the complete transition model (the board layout and the rules) in
  advance. RL agents only need to interact with the environment.
* It assumes deterministic transitions. If the dice were random (or the
  environment dynamic: snakes/ladders moving between games), there is no
  single path to find - the problem becomes an MDP and needs expectations
  (value iteration) or learning (RL), not a plain shortest-path search.
* It minimises the number of moves, not the reward. With rewards such as
  +20 for ladders and -20 for snakes the reward-optimal policy can differ.
* The state space must be small enough to enumerate explicitly.

    python -m evaluation.bfs_baseline
"""
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.board_configs import BOARD_CONFIGS  # noqa: E402
from configs.paths import CSV_DIR, ensure_dirs  # noqa: E402
from environment.board_generator import DICE_VALUES, EVENT_INVALID, Board, build_board  # noqa: E402
from environment.game_engine import GameEngine  # noqa: E402


def bfs_shortest_path(board: Board) -> Optional[Dict]:
    """Shortest sequence of dice values from start to goal (None if unreachable)."""
    start, goal = board.start, board.goal
    parent = {start: None}  # position -> (previous position, dice)
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        for dice in DICE_VALUES:
            new_pos, event = board.resolve(pos, dice)
            if event != EVENT_INVALID and new_pos not in parent:
                parent[new_pos] = (pos, dice)
                queue.append(new_pos)
    if goal not in parent:
        return None
    dice_seq: List[int] = []
    positions = [goal]
    node = goal
    while parent[node] is not None:
        node, dice = parent[node]
        dice_seq.append(dice)
        positions.append(node)
    dice_seq.reverse()
    positions.reverse()
    return {"min_moves": len(dice_seq), "dice_sequence": dice_seq, "positions": positions}


def replay_reward(board: Board, dice_sequence: List[int], max_steps: int) -> float:
    """Total reward of a dice sequence, computed with the real game engine."""
    engine = GameEngine(board, max_steps=max_steps)
    engine.reset()
    for dice in dice_sequence:
        engine.step(dice)
    return engine.total_reward


def run_bfs_baseline(save: bool = True) -> pd.DataFrame:
    """BFS result for each of the three experiment boards."""
    rows = []
    for exp_id, cfg in BOARD_CONFIGS.items():
        board = build_board(cfg)
        result = bfs_shortest_path(board)
        rows.append({
            "experiment": exp_id,
            "board_size": cfg.size,
            "bfs_min_moves": result["min_moves"],
            "bfs_total_reward": replay_reward(board, result["dice_sequence"], cfg.max_steps),
            "bfs_dice_sequence": " ".join(map(str, result["dice_sequence"])),
            "bfs_path": " -> ".join(map(str, result["positions"])),
        })
    df = pd.DataFrame(rows)
    if save:
        ensure_dirs()
        df.to_csv(CSV_DIR / "bfs_baseline.csv", index=False)
    return df


def main() -> None:
    df = run_bfs_baseline()
    print(df.drop(columns=["bfs_path"]).to_string(index=False))
    print(f"\nSaved {CSV_DIR / 'bfs_baseline.csv'}")
    print("Note: BFS is a planning baseline on the known deterministic board - it is not used by the RL agents.")


if __name__ == "__main__":
    main()
