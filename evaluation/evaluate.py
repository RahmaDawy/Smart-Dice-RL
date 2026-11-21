"""Evaluate trained agents on episodes that were never used for training.

* Agents act greedily (no exploration).
* Evaluation seeds start at EVAL_SEED_OFFSET, which training never uses.
* The core game is deterministic, so a greedy agent plays the SAME game every
  episode (std = 0). To get a meaningful spread we ALSO evaluate under a noisy
  dice (``slip_prob`` = 0.1: 10 % of rolls show a random value instead of the
  chosen one). Both settings are saved in the CSV (column ``slip_prob``).
* A random agent is evaluated as a reference.

    python -m evaluation.evaluate --episodes 100
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from algorithms.registry import AGENT_KEYS, DISPLAY_NAMES, ModelMissingError, load_agent  # noqa: E402
from configs.board_configs import BOARD_CONFIGS  # noqa: E402
from configs.hyperparameters import EVAL_SEED_OFFSET  # noqa: E402
from configs.paths import CSV_DIR, ensure_dirs  # noqa: E402
from environment.gym_env import SmartDiceEnv  # noqa: E402
from evaluation.metrics import summarize_episodes  # noqa: E402

DEFAULT_SLIP_PROBS = (0.0, 0.1)


def run_episode(env: SmartDiceEnv, agent, greedy: bool = True, seed: Optional[int] = None) -> dict:
    """Play one full episode and return reward / moves / success."""
    state, _ = env.reset(seed=seed)
    total, moves, terminated = 0.0, 0, False
    while True:
        action = agent.act(state, greedy=greedy)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        moves += 1
        if terminated or truncated:
            return {"reward": total, "moves": moves, "success": int(terminated)}


def evaluate_agent(agent, experiment_id: int, episodes: int = 100, slip_prob: float = 0.0,
                   seed_offset: int = EVAL_SEED_OFFSET) -> pd.DataFrame:
    """Evaluate one agent; returns a per-episode DataFrame."""
    env = SmartDiceEnv(experiment=experiment_id, slip_prob=slip_prob)
    rows = []
    for i in range(episodes):
        result = run_episode(env, agent, greedy=True, seed=seed_offset + experiment_id * 1000 + i)
        rows.append({"episode": i + 1, **result})
    return pd.DataFrame(rows)


def evaluate_all(agents: Iterable[str] = AGENT_KEYS, experiments: Iterable[int] = tuple(BOARD_CONFIGS),
                 episodes: int = 100, slip_probs: Iterable[float] = DEFAULT_SLIP_PROBS,
                 save: bool = True) -> pd.DataFrame:
    """Evaluate every available agent on every experiment; saves two CSV files."""
    summary_rows, episode_frames = [], []
    for exp in experiments:
        for key in agents:
            try:
                agent = load_agent(key, exp)
            except (ModelMissingError, ImportError) as err:
                print(f"[skip] {DISPLAY_NAMES[key]} / exp {exp}: {err}")
                continue
            for slip in slip_probs:
                df = evaluate_agent(agent, exp, episodes, slip)
                stats = summarize_episodes(df["reward"], df["moves"], df["success"])
                summary_rows.append({"experiment": exp, "board_size": BOARD_CONFIGS[exp].size,
                                     "algorithm": key, "slip_prob": slip, **stats})
                episode_frames.append(df.assign(experiment=exp, algorithm=key, slip_prob=slip))
    summary = pd.DataFrame(summary_rows)
    if save and not summary.empty:
        ensure_dirs()
        summary.to_csv(CSV_DIR / "evaluation_summary.csv", index=False)
        pd.concat(episode_frames).to_csv(CSV_DIR / "evaluation_episodes.csv", index=False)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained agents (greedy policy, unseen episodes).")
    parser.add_argument("--episodes", type=int, default=100, help="evaluation episodes per setting")
    parser.add_argument("--experiments", nargs="+", type=int, choices=list(BOARD_CONFIGS), default=list(BOARD_CONFIGS))
    parser.add_argument("--agents", nargs="+", choices=AGENT_KEYS, default=list(AGENT_KEYS))
    args = parser.parse_args(argv)
    summary = evaluate_all(args.agents, args.experiments, args.episodes)
    if summary.empty:
        print("Nothing evaluated: no trained models found. Train first (see README).")
        return
    print(summary.round(2).to_string(index=False))
    print(f"\nSaved {CSV_DIR / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
