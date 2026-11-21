"""Extension 1 - RANDOM DICE (stochastic transitions).

The core game lets the agent pick the dice value. Here the ENVIRONMENT rolls the
dice, so the six "choose a dice value" actions no longer make sense. The
decision problem is reformulated:

    state  = (position, rolled dice)            -> Discrete(N * 6)
    action = 0: MOVE by the rolled value        -> Discrete(2)
             1: HOLD (skip the move, reward -1, a new die is rolled next turn)

Same board, same rewards (hold costs like a normal square). A smart agent learns
to HOLD when the roll would land on a snake or overshoot the goal.
The transition is now stochastic (the next dice roll is random).
"""
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SarsaAgent
from configs.board_configs import BOARD_CONFIGS, get_board_config
from configs.hyperparameters import EVAL_SEED_OFFSET, TRAIN_SEED
from configs.paths import CSV_DIR, MODELS_DIR, PLOTS_DIR, ensure_dirs
from environment.board_generator import build_board
from environment.game_engine import GameEngine
from evaluation.metrics import moving_average

ACTION_MOVE, ACTION_HOLD = 0, 1
N_ACTIONS = 2


class RandomDiceEnv(gym.Env):
    """Gymnasium environment with a random dice (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, experiment: int = 2, max_steps: Optional[int] = None):
        super().__init__()
        self.config = get_board_config(experiment)
        self.board = build_board(self.config)
        self.engine = GameEngine(self.board, max_steps or self.config.max_steps)
        self.observation_space = spaces.Discrete(self.config.size * 6)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.dice = 1

    def _obs(self) -> int:
        return (self.engine.position - 1) * 6 + (self.dice - 1)

    def _roll(self) -> None:
        self.dice = int(self.np_random.integers(1, 7))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.engine.reset()
        self._roll()
        return self._obs(), {"position": 1, "dice": self.dice}

    def step(self, action):
        action = int(action)
        if action not in (ACTION_MOVE, ACTION_HOLD):
            raise ValueError("action must be 0 (move) or 1 (hold)")
        result = self.engine.step(self.dice) if action == ACTION_MOVE else self.engine.hold()
        info = {"position": result.position, "event": result.event, "moves": result.moves, "rolled": self.dice}
        self._roll()                      # the next (random) roll the agent will see
        info["next_dice"] = self.dice
        return self._obs(), result.reward, result.terminated, result.truncated, info


# --- policies / evaluation ------------------------------------------------------
def always_move(_obs) -> int:
    """The classic game: no decision, always play the roll."""
    return ACTION_MOVE


def evaluate_policy(env: RandomDiceEnv, policy: Callable[[int], int], episodes: int, seed: int) -> dict:
    rewards, moves, wins, holds = [], [], [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, n, terminated = 0.0, 0, False
        while True:
            action = policy(obs)
            holds += action == ACTION_HOLD
            obs, reward, terminated, truncated, _ = env.step(action)
            total, n = total + reward, n + 1
            if terminated or truncated:
                break
        rewards.append(total), moves.append(n), wins.append(int(terminated))
    return {"episodes": episodes, "success_rate": float(np.mean(wins)), "avg_moves": float(np.mean(moves)),
            "avg_reward": float(np.mean(rewards)), "hold_rate": holds / max(1, sum(moves))}


# --- training -------------------------------------------------------------------
def random_dice_model_path(algo: str, experiment_id: int, models_dir=None) -> Path:
    return Path(models_dir or MODELS_DIR) / f"ext_random_dice_{algo}_exp{experiment_id}.npz"


def train_random_dice(algo: str, experiment_id: int = 2, episodes: int = 20000, seed: int = TRAIN_SEED,
                      models_dir=None, csv_dir=None, verbose: bool = True, save: bool = True):
    """Tabular Q-Learning / SARSA on the (position, dice) state space."""
    if algo not in ("q_learning", "sarsa"):
        raise ValueError("algo must be 'q_learning' or 'sarsa'")
    env = RandomDiceEnv(experiment_id)
    cls = QLearningAgent if algo == "q_learning" else SarsaAgent
    agent = cls(env.observation_space.n, n_actions=N_ACTIONS, alpha=0.1, gamma=0.99, epsilon=1.0,
                epsilon_min=0.05, epsilon_decay=0.05 ** (1.0 / (0.6 * episodes)), seed=seed)
    rows, start = [], time.perf_counter()
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed if ep == 1 else None)
        eps_used, total, n = agent.epsilon, 0.0, 0
        action = agent.act(state, greedy=False)
        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            if algo == "q_learning":
                agent.update(state, action, reward, next_state, terminated)
                next_action = agent.act(next_state, greedy=False)
            else:
                next_action = agent.act(next_state, greedy=False)
                agent.update(state, action, reward, next_state, next_action, terminated)
            state, action, total, n = next_state, next_action, total + reward, n + 1
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rows.append({"episode": ep, "reward": total, "moves": n, "success": int(terminated), "epsilon": eps_used,
                     "elapsed_seconds": time.perf_counter() - start})
        if verbose and ep % max(1, episodes // 5) == 0:
            recent = rows[-200:]
            print(f"[random dice | {algo} | exp {experiment_id}] episode {ep}/{episodes} "
                  f"avg reward {np.mean([r['reward'] for r in recent]):.1f}", flush=True)
    df = pd.DataFrame(rows)
    if save:
        ensure_dirs()
        agent.save(random_dice_model_path(algo, experiment_id, models_dir))
        df.to_csv(Path(csv_dir or CSV_DIR) / f"extension_random_dice_training_{algo}_exp{experiment_id}.csv", index=False)
    return agent, df


def run_random_dice(experiment_id: int = 2, episodes: int = 20000, eval_episodes: int = 1000,
                    models_dir=None, csv_dir=None, plots_dir=None, save: bool = True) -> pd.DataFrame:
    """Train both algorithms, evaluate them against baselines, save CSV + plot."""
    agents, logs = {}, {}
    for algo in ("q_learning", "sarsa"):
        agents[algo], logs[algo] = train_random_dice(algo, experiment_id, episodes, models_dir=models_dir,
                                                      csv_dir=csv_dir, save=save)
    env = RandomDiceEnv(experiment_id)
    rng = np.random.default_rng(0)
    policies: Dict[str, Callable] = {
        "always_move (classic game)": always_move,
        "random decisions": lambda obs: int(rng.integers(N_ACTIONS)),
        "Q-Learning (move/hold)": lambda obs: agents["q_learning"].act(obs, greedy=True),
        "SARSA (move/hold)": lambda obs: agents["sarsa"].act(obs, greedy=True),
    }
    rows = []
    for name, policy in policies.items():
        stats = evaluate_policy(env, policy, eval_episodes, seed=EVAL_SEED_OFFSET)
        rows.append({"experiment": experiment_id, "board_size": env.config.size, "policy": name, **stats})
    table = pd.DataFrame(rows)
    if save:
        table.to_csv(Path(csv_dir or CSV_DIR) / f"extension_random_dice_exp{experiment_id}.csv", index=False)
        _plot_random_dice(logs, table, experiment_id, Path(plots_dir or PLOTS_DIR))
    return table


def _plot_random_dice(logs, table, experiment_id, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for algo, color in (("q_learning", "#1f77b4"), ("sarsa", "#d62728")):
        df = logs[algo]
        axes[0].plot(df["episode"], moving_average(df["reward"], 200), color=color, label=algo)
    base = table[table["policy"].str.startswith("always_move")]["avg_reward"].iloc[0]
    axes[0].axhline(base, color="k", linestyle="--", label="always move (classic game)")
    axes[0].set(title="Training reward (moving avg)", xlabel="Episode", ylabel="Reward")
    axes[0].legend()
    for ax, col, title in ((axes[1], "avg_reward", "Evaluation: average reward"),
                           (axes[2], "success_rate", "Evaluation: success rate")):
        ax.bar(range(len(table)), table[col], color=["#7f7f7f", "#bbbbbb", "#1f77b4", "#d62728"])
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table["policy"], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Extension: random dice - {BOARD_CONFIGS[experiment_id].name}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"extension_random_dice_exp{experiment_id}.png", dpi=130)
    plt.close(fig)
