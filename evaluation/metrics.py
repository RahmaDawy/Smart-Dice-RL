"""Metric helpers shared by evaluation, comparison and plotting."""
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Convergence criterion (documented in the README):
#   1. plateau  = mean reward of the LAST 10 % of the training episodes
#   2. threshold = plateau - 10 % * |plateau|
#   3. rolling  = mean reward over a window of 100 episodes
#   The convergence episode is the first episode e at which the rolling mean
#   is >= threshold AND stays >= threshold for the next 50 episodes.
#   If this never happens the function returns None (reported as "not converged").
# ---------------------------------------------------------------------------
WINDOW = 100
TOLERANCE = 0.10
PATIENCE = 50
FINAL_FRACTION = 0.10


def moving_average(values: Sequence[float], window: int = WINDOW) -> np.ndarray:
    """Trailing moving average; the first ``window-1`` entries use a shorter window."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty_like(arr)
    for i in range(arr.size):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def convergence_episode(
    rewards: Sequence[float],
    window: int = WINDOW,
    tolerance: float = TOLERANCE,
    patience: int = PATIENCE,
    final_fraction: float = FINAL_FRACTION,
) -> Optional[int]:
    """Return the 1-based convergence episode, or None if never converged."""
    arr = np.asarray(rewards, dtype=float)
    n = arr.size
    if n < window + patience:
        return None
    tail = max(1, int(n * final_fraction))
    plateau = arr[-tail:].mean()
    threshold = plateau - tolerance * abs(plateau)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    rolling = (csum[window:] - csum[:-window]) / window        # rolling[i] ends at episode i+window
    ok = (rolling >= threshold).astype(int)
    run = np.convolve(ok, np.ones(patience, dtype=int), mode="valid")   # run[i] = sum ok[i:i+patience]
    hits = np.flatnonzero(run == patience)
    return int(hits[0] + window) if hits.size else None


def summarize_episodes(rewards: Sequence[float], moves: Sequence[float], successes: Sequence[int]) -> dict:
    """Aggregate per-episode results into the evaluation metrics."""
    rewards, moves, successes = map(lambda x: np.asarray(x, dtype=float), (rewards, moves, successes))
    successful_moves = moves[successes == 1]
    return {
        "episodes": int(rewards.size),
        "success_rate": float(successes.mean()) if rewards.size else float("nan"),
        "avg_reward": float(rewards.mean()) if rewards.size else float("nan"),
        "std_reward": float(rewards.std()) if rewards.size else float("nan"),
        "avg_moves": float(moves.mean()) if rewards.size else float("nan"),
        "avg_moves_successful": float(successful_moves.mean()) if successful_moves.size else float("nan"),
    }
