"""Board data structure and reproducible board generation.

``Board.resolve`` is the ONLY place where the movement rules live; the game
engine, BFS baseline and board generator all call it.
"""
import math
import random
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple

from configs.board_configs import BoardConfig

DICE_VALUES = (1, 2, 3, 4, 5, 6)

# Event names returned by Board.resolve / GameEngine.step
EVENT_NORMAL = "normal"
EVENT_SNAKE = "snake"
EVENT_LADDER = "ladder"
EVENT_GOAL = "goal"
EVENT_INVALID = "invalid"


@dataclass(frozen=True)
class Board:
    """A Snakes-and-Ladders board.

    ``snakes`` maps head square -> tail square (tail < head).
    ``ladders`` maps bottom square -> top square (top > bottom).
    """

    size: int
    snakes: Dict[int, int] = field(default_factory=dict)
    ladders: Dict[int, int] = field(default_factory=dict)

    @property
    def goal(self) -> int:
        return self.size

    @property
    def start(self) -> int:
        return 1

    def resolve(self, position: int, dice: int) -> Tuple[int, str]:
        """Apply one dice move (no rewards) and return ``(new_position, event)``.

        * overshooting the goal -> ``(position, "invalid")``
        * landing on the goal   -> ``(size, "goal")``
        * landing on a snake head / ladder bottom -> jump immediately, once
          (there are no chained transitions by construction).
        """
        if dice not in DICE_VALUES:
            raise ValueError(f"dice must be an integer from 1 to 6, got {dice!r}")
        target = position + dice
        if target > self.size:
            return position, EVENT_INVALID
        if target == self.size:
            return target, EVENT_GOAL
        if target in self.snakes:
            return self.snakes[target], EVENT_SNAKE
        if target in self.ladders:
            return self.ladders[target], EVENT_LADDER
        return target, EVENT_NORMAL


def _goal_reachable(board: Board) -> bool:
    """Small BFS used only to guarantee generated boards are solvable."""
    seen = {board.start}
    queue = deque([board.start])
    while queue:
        pos = queue.popleft()
        if pos == board.goal:
            return True
        for dice in DICE_VALUES:
            new_pos, event = board.resolve(pos, dice)
            if event != EVENT_INVALID and new_pos not in seen:
                seen.add(new_pos)
                queue.append(new_pos)
    return False


def board_columns(size: int) -> int:
    """Number of columns used when the board is drawn (also used for layout-aware generation)."""
    return 6 if size <= 30 else 10


def _grid_xy(square: int, cols: int) -> Tuple[float, float]:
    """(column, row) of a square in the zig-zag layout (square 1 bottom-left)."""
    row, col = divmod(square - 1, cols)
    if row % 2 == 1:
        col = cols - 1 - col
    return float(col), float(row)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _segment_gap(s1, s2) -> float:
    """Distance between two drawn snake/ladder lines (0 if they cross)."""
    (a, b), (c, d) = s1, s2
    d1, d2, d3, d4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(_point_seg_dist(a, c, d), _point_seg_dist(b, c, d), _point_seg_dist(c, a, b), _point_seg_dist(d, a, b))


def _assign_kinds(rng, num_snakes, num_ladders, feasible):
    """Decide which band gets a snake and which a ladder.

    With equal counts every pair of neighbouring bands holds one snake and one
    ladder (random order), so both are spread evenly. Bands where a snake could
    not drop (too close to the start) or a ladder could not climb (too close to
    the goal) are avoided.
    """
    total = num_snakes + num_ladders
    for _ in range(500):
        if num_snakes == num_ladders:
            kinds = []
            for _pair in range(total // 2):
                pair = ["ladder", "snake"]
                rng.shuffle(pair)
                kinds += pair
        else:
            kinds = ["ladder"] * num_ladders + ["snake"] * num_snakes
            rng.shuffle(kinds)
        if all(feasible[i][kind] for i, kind in enumerate(kinds)):
            return kinds
    return None


def generate_board(size: int, num_snakes: int, num_ladders: int, seed: int) -> Board:
    """Generate a valid board deterministically from ``seed``.

    Guarantees:
      * snakes/ladders only use squares 2..N-1 (never start or goal);
      * every square is used at most once (so a square is never both a snake
        and a ladder start, and there are no chained transitions);
      * snake drops and ladder climbs are between ``max(3, N // 10)`` and ``N // 4`` squares;
      * snake heads and ladder bottoms are spread over the whole board: squares
        2..N-1 are cut into one band per snake/ladder and each band gets one;
      * the goal can be reached from the start.
    """
    if size < 12:
        raise ValueError("board size must be at least 12")
    min_jump = max(3, size // 10)
    max_jump = max(min_jump + 1, size // 4)
    rng = random.Random(seed)

    total = num_snakes + num_ladders
    squares = list(range(2, size))                                   # squares 2 .. N-1
    band_size = len(squares) / total
    bands = [squares[int(i * band_size): int((i + 1) * band_size)] for i in range(total)]
    feasible = [{"snake": any(sq - min_jump >= 2 for sq in zone),
                 "ladder": any(sq + min_jump <= size - 1 for sq in zone)} for zone in bands]

    for _attempt in range(2000):
        kinds = _assign_kinds(rng, num_snakes, num_ladders, feasible)
        if kinds is None:
            break
        free = set(squares)
        snakes: Dict[int, int] = {}
        ladders: Dict[int, int] = {}
        ok = True
        for kind, zone in zip(kinds, bands):
            starts = [sq for sq in zone if sq in free]
            if not starts:
                ok = False
                break
            start = rng.choice(starts)
            if kind == "ladder":
                ends = [e for e in free if min_jump <= e - start <= max_jump]
            else:
                ends = [e for e in free if min_jump <= start - e <= max_jump]
            if not ends:
                ok = False
                break
            end = rng.choice(sorted(ends))
            free.discard(start)
            free.discard(end)
            (ladders if kind == "ladder" else snakes)[start] = end
        if not ok:
            continue
        board = Board(size=size, snakes=snakes, ladders=ladders)
        if _goal_reachable(board):
            return board
    raise RuntimeError("could not generate a valid board; relax the constraints")


@lru_cache(maxsize=None)
def build_board(config: BoardConfig) -> Board:
    """Return the (cached, reproducible) board for a ``BoardConfig``."""
    return generate_board(config.size, config.num_snakes, config.num_ladders, config.seed)