"""Shared training loops for Q-Learning, SARSA and DQN.

Every algorithm learns purely by interacting with ``SmartDiceEnv``.
Per-episode metrics are written to ``results/csv/training_<algo>_exp<k>.csv``
and the trained model to ``models/``.
"""
import argparse
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from algorithms.registry import DISPLAY_NAMES, model_path
from configs.board_configs import BOARD_CONFIGS
from configs.hyperparameters import ALGORITHMS, TRAIN_SEED, get_hyperparameters
from configs.paths import CSV_DIR, MODELS_DIR, ensure_dirs
from environment.gym_env import SmartDiceEnv

CSV_COLUMNS = ["episode", "reward", "moves", "success", "epsilon", "elapsed_seconds"]


def training_csv_path(algo: str, experiment_id: int, csv_dir: Optional[Path] = None) -> Path:
    return Path(csv_dir or CSV_DIR) / f"training_{algo}_exp{experiment_id}.csv"


def _build_agent(algo: str, n_states: int, agent_kwargs: dict, seed: int):
    if algo == "q_learning":
        from algorithms.q_learning import QLearningAgent
        return QLearningAgent(n_states, seed=seed, **agent_kwargs)
    if algo == "sarsa":
        from algorithms.sarsa import SarsaAgent
        return SarsaAgent(n_states, seed=seed, **agent_kwargs)
    from algorithms.dqn import DQNAgent
    return DQNAgent(n_states, seed=seed, **agent_kwargs)


# --- one-episode loops (one per algorithm family) ----------------------------
def _episode_q_learning(env, agent, state):
    total, moves, terminated = 0.0, 0, False
    while True:
        action = agent.act(state, greedy=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, terminated)
        state, total, moves = next_state, total + reward, moves + 1
        if terminated or truncated:
            return total, moves, terminated


def _episode_sarsa(env, agent, state):
    total, moves = 0.0, 0
    action = agent.act(state, greedy=False)
    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = agent.act(next_state, greedy=False)  # on-policy: the action we will really take
        agent.update(state, action, reward, next_state, next_action, terminated)
        state, action, total, moves = next_state, next_action, total + reward, moves + 1
        if terminated or truncated:
            return total, moves, terminated


def _episode_dqn(env, agent, state, step_counter):
    total, moves = 0.0, 0
    while True:
        action = agent.act(state, greedy=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.remember(state, action, reward, next_state, terminated)
        step_counter[0] += 1
        if step_counter[0] % agent.train_every == 0:
            agent.learn()
        state, total, moves = next_state, total + reward, moves + 1
        if terminated or truncated:
            return total, moves, terminated


def train(
    algo: str,
    experiment_id: int,
    episodes: Optional[int] = None,
    seed: int = TRAIN_SEED,
    models_dir: Optional[Path] = None,
    csv_dir: Optional[Path] = None,
    episode_scale: float = 1.0,
    verbose: bool = True,
    save: bool = True,
) -> Tuple[object, pd.DataFrame]:
    """Train ``algo`` on experiment ``experiment_id``; returns (agent, metrics DataFrame).

    ``episode_scale`` < 1 shortens training for quick tests; epsilon decay is
    rescaled so epsilon still reaches its floor at the same fraction of training.
    """
    if algo not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo!r}. Valid: {ALGORITHMS}")
    ensure_dirs()
    hp = get_hyperparameters(algo, experiment_id)
    n_episodes = int(episodes) if episodes is not None else max(1, int(hp["episodes"] * episode_scale))
    agent_kwargs = dict(hp["agent"])
    if episodes is None and episode_scale != 1.0:
        agent_kwargs["epsilon_decay"] = agent_kwargs["epsilon_decay"] ** (1.0 / episode_scale)
    elif episodes is not None:
        agent_kwargs["epsilon_decay"] = agent_kwargs["epsilon_decay"] ** (hp["episodes"] / n_episodes)

    env = SmartDiceEnv(experiment=experiment_id)
    agent = _build_agent(algo, env.observation_space.n, agent_kwargs, seed)
    step_counter = [0]
    rows: List[dict] = []
    report_every = max(1, n_episodes // 10)
    start = time.perf_counter()

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed if episode == 1 else None)
        epsilon_used = agent.epsilon
        if algo == "q_learning":
            total, moves, success = _episode_q_learning(env, agent, state)
        elif algo == "sarsa":
            total, moves, success = _episode_sarsa(env, agent, state)
        else:
            total, moves, success = _episode_dqn(env, agent, state, step_counter)
        agent.decay_epsilon()
        rows.append({
            "episode": episode, "reward": total, "moves": moves, "success": int(success),
            "epsilon": epsilon_used, "elapsed_seconds": time.perf_counter() - start,
        })
        if verbose and (episode % report_every == 0 or episode == n_episodes):
            recent = rows[-100:]
            print(f"[{DISPLAY_NAMES[algo]:>10} | exp {experiment_id}] episode {episode:>6}/{n_episodes} "
                  f"avg reward {np.mean([r['reward'] for r in recent]):8.1f}  "
                  f"success {np.mean([r['success'] for r in recent]):5.2f}  "
                  f"eps {agent.epsilon:.3f}  time {rows[-1]['elapsed_seconds']:6.1f}s", flush=True)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if save:
        path = agent.save(model_path(algo, experiment_id, models_dir))
        csv_path = training_csv_path(algo, experiment_id, csv_dir)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"   saved model -> {path}\n   saved metrics -> {csv_path}")
    return agent, df


def parse_experiments(value: str) -> List[int]:
    if value == "all":
        return list(BOARD_CONFIGS)
    exp = int(value)
    if exp not in BOARD_CONFIGS:
        raise argparse.ArgumentTypeError(f"experiment must be 1, 2, 3 or 'all' (got {value})")
    return [exp]


def run_cli(algo: str, argv: Optional[Iterable[str]] = None) -> None:
    """Command-line entry point shared by train_q_learning/train_sarsa/train_dqn."""
    parser = argparse.ArgumentParser(description=f"Train {DISPLAY_NAMES[algo]} on the Smart Dice board game.")
    parser.add_argument("--experiment", "-e", default="all", help="1, 2, 3 or 'all' (default: all)")
    parser.add_argument("--episodes", type=int, default=None, help="override number of training episodes")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED, help=f"random seed (default {TRAIN_SEED})")
    parser.add_argument("--quick", action="store_true", help="short smoke-test run (10%% of the episodes)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    for exp in parse_experiments(str(args.experiment)):
        train(algo, exp, episodes=args.episodes, seed=args.seed, episode_scale=0.1 if args.quick else 1.0)
