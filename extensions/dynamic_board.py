"""Extension 2 - DYNAMIC BOARD (snakes and ladders change between episodes).

A position-only agent (like the core Q-Learning agent) memorises ONE board and
cannot adapt. To adapt, the agent must SEE the board. Reformulation:

    observation = for each dice value 1..6, what would happen from here
                  (one-hot: normal / snake / ladder / goal / invalid, plus the
                  signed distance moved) + position + distance to goal
                  -> a 38-dimensional vector, independent of the specific board
    action      = dice value (same 6 actions as the core project)

A DQN trained on a fresh random board every episode learns a general rule
("take ladders, avoid snakes, don't overshoot") and is tested on boards that
were never used for training. It is compared with the position-only
Q-Learning agent from the core project and with a random agent; BFS gives the
optimum for each test board.
"""
import time
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from algorithms.base import RandomAgent
from algorithms.dqn import DQNAgent, ReplayBuffer
from algorithms.registry import ModelMissingError, load_agent
from configs.board_configs import BOARD_CONFIGS, get_board_config
from configs.hyperparameters import TRAIN_SEED
from configs.paths import CSV_DIR, MODELS_DIR, PLOTS_DIR, ensure_dirs
from environment.board_generator import DICE_VALUES, Board, generate_board
from environment.game_engine import GameEngine
from evaluation.bfs_baseline import bfs_shortest_path, replay_reward
from evaluation.metrics import moving_average

EVENTS = ("normal", "snake", "ladder", "goal", "invalid")
FEATURE_DIM = 6 * (len(EVENTS) + 1) + 2
TRAIN_SEED_LIMIT = 1_000_000_000       # training boards use seeds below this value ...
EVAL_BOARD_SEED_START = 1_000_000_000  # ... held-out test boards use seeds from here on


def board_features(board: Board, position: int) -> np.ndarray:
    """Board-independent description of what each dice value would do from ``position``."""
    n = board.size
    feats = []
    for dice in DICE_VALUES:
        new_pos, event = board.resolve(position, dice)
        one_hot = [1.0 if event == e else 0.0 for e in EVENTS]
        feats += one_hot + [(new_pos - position) / n]
    feats += [(position - 1) / (n - 1), min((n - position) / 12.0, 1.0)]
    return np.asarray(feats, dtype=np.float32)


class DynamicBoardEnv(gym.Env):
    """A new random board (same size / counts as the experiment) at every reset."""

    metadata = {"render_modes": []}

    def __init__(self, experiment: int = 2, max_steps: Optional[int] = None):
        super().__init__()
        self.config = get_board_config(experiment)
        self.max_steps = max_steps or self.config.max_steps
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(FEATURE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.board: Optional[Board] = None
        self.engine: Optional[GameEngine] = None
        self.board_seed: Optional[int] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and "board_seed" in options:
            self.board_seed = int(options["board_seed"])
        else:
            self.board_seed = int(self.np_random.integers(0, TRAIN_SEED_LIMIT))
        c = self.config
        self.board = generate_board(c.size, c.num_snakes, c.num_ladders, self.board_seed)
        self.engine = GameEngine(self.board, self.max_steps)
        self.engine.reset()
        return board_features(self.board, 1), {"board_seed": self.board_seed, "position": 1}

    def step(self, action):
        result = self.engine.step(int(action) + 1)
        info = {"board_seed": self.board_seed, "position": result.position, "event": result.event}
        return board_features(self.board, result.position), result.reward, result.terminated, result.truncated, info


class VectorReplayBuffer(ReplayBuffer):
    """Replay buffer that stores feature vectors instead of state indices."""

    def __init__(self, capacity: int, dim: int):
        super().__init__(capacity)
        self.states = np.zeros((capacity, dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, dim), dtype=np.float32)


class VectorDQNAgent(DQNAgent):
    """The core DQN (replay + target network) with a feature-vector input."""

    name = "dqn_dynamic"

    def __init__(self, n_features: int = FEATURE_DIM, n_actions: int = 6, **kwargs):
        super().__init__(n_features, n_actions, **kwargs)
        self.buffer = VectorReplayBuffer(self.buffer_size, n_features)

    def _encode(self, states):
        import torch
        return torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)


DYNAMIC_HPARAMS = dict(hidden_sizes=(128, 128), lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                       batch_size=64, buffer_size=50_000, learning_starts=500, train_every=2,
                       target_update_every=250, reward_scale=0.05)


def dynamic_model_path(experiment_id: int, models_dir=None) -> Path:
    return Path(models_dir or MODELS_DIR) / f"ext_dynamic_dqn_exp{experiment_id}.pt"


def train_dynamic(experiment_id: int = 2, episodes: int = 6000, seed: int = TRAIN_SEED, models_dir=None,
                  csv_dir=None, verbose: bool = True, save: bool = True):
    env = DynamicBoardEnv(experiment_id)
    agent = VectorDQNAgent(seed=seed, epsilon_decay=0.05 ** (1.0 / (0.6 * episodes)), **DYNAMIC_HPARAMS)
    rows, start, steps = [], time.perf_counter(), 0
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed if ep == 1 else None)
        eps_used, total, n = agent.epsilon, 0.0, 0
        while True:
            action = agent.act(state, greedy=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated)
            steps += 1
            if steps % agent.train_every == 0:
                agent.learn()
            state, total, n = next_state, total + reward, n + 1
            if terminated or truncated:
                break
        agent.decay_epsilon()
        rows.append({"episode": ep, "reward": total, "moves": n, "success": int(terminated), "epsilon": eps_used,
                     "elapsed_seconds": time.perf_counter() - start})
        if verbose and ep % max(1, episodes // 5) == 0:
            recent = rows[-200:]
            print(f"[dynamic board | DQN | exp {experiment_id}] episode {ep}/{episodes} "
                  f"avg reward {np.mean([r['reward'] for r in recent]):.1f}  success "
                  f"{np.mean([r['success'] for r in recent]):.2f}", flush=True)
    df = pd.DataFrame(rows)
    if save:
        ensure_dirs()
        agent.save(dynamic_model_path(experiment_id, models_dir))
        df.to_csv(Path(csv_dir or CSV_DIR) / f"extension_dynamic_training_exp{experiment_id}.csv", index=False)
    return agent, df


def load_dynamic_agent(experiment_id: int = 2, models_dir=None) -> VectorDQNAgent:
    path = dynamic_model_path(experiment_id, models_dir)
    if not path.exists():
        raise ModelMissingError(f"{path.name} not found. Run: python run_project.py extensions --which dynamic")
    return VectorDQNAgent.load(path)


# --- evaluation on held-out boards -----------------------------------------------
def play_on_board(board: Board, max_steps: int, choose_dice: Callable[[GameEngine], int]) -> dict:
    engine = GameEngine(board, max_steps)
    engine.reset()
    res = None
    while not engine.done:
        res = engine.step(choose_dice(engine))
    return {"success": int(res.terminated), "moves": res.moves, "reward": res.total_reward}


def evaluate_dynamic(experiment_id: int = 2, boards: int = 100, models_dir=None) -> pd.DataFrame:
    """Compare policies on ``boards`` held-out random boards."""
    cfg = get_board_config(experiment_id)
    dyn = load_dynamic_agent(experiment_id, models_dir)
    chooser = {
        "dynamic DQN (sees the board)": lambda e: dyn.act(board_features(e.board, e.position), greedy=True) + 1,
    }
    rnd = RandomAgent(seed=0)
    chooser["random"] = lambda e: rnd.act(e.position - 1) + 1
    try:
        static = load_agent("q_learning", experiment_id, models_dir)
        chooser["static Q-Learning (core agent, position only)"] = lambda e: static.act(e.position - 1) + 1
    except (ModelMissingError, FileNotFoundError):
        print("[note] core Q-Learning model not found; skipping the static baseline.")
        static = None

    test_boards = [generate_board(cfg.size, cfg.num_snakes, cfg.num_ladders, EVAL_BOARD_SEED_START + i)
                   for i in range(boards)]
    bfs = [bfs_shortest_path(b) for b in test_boards]
    rows = []
    for name, fn in chooser.items():
        results = [play_on_board(b, cfg.max_steps, fn) for b in test_boards]
        rows.append({"experiment": experiment_id, "board_size": cfg.size, "policy": name, "test_boards": boards,
                     "success_rate": np.mean([r["success"] for r in results]),
                     "avg_moves": np.mean([r["moves"] for r in results]),
                     "avg_reward": np.mean([r["reward"] for r in results])})
    if static is not None:   # sanity check: the static agent on the ONE board it was trained on
        from environment.board_generator import build_board
        r = play_on_board(build_board(cfg), cfg.max_steps, chooser["static Q-Learning (core agent, position only)"])
        rows.append({"experiment": experiment_id, "board_size": cfg.size,
                     "policy": "static Q-Learning on its own fixed board", "test_boards": 1,
                     "success_rate": r["success"], "avg_moves": r["moves"], "avg_reward": r["reward"]})
    rows.append({"experiment": experiment_id, "board_size": cfg.size, "policy": "BFS optimum (reference)",
                 "test_boards": boards, "success_rate": 1.0,
                 "avg_moves": np.mean([p["min_moves"] for p in bfs]),
                 "avg_reward": np.mean([replay_reward(b, p["dice_sequence"], cfg.max_steps)
                                        for b, p in zip(test_boards, bfs)])})
    return pd.DataFrame(rows)


def run_dynamic(experiment_id: int = 2, episodes: int = 6000, boards: int = 100, models_dir=None, csv_dir=None,
                plots_dir=None, save: bool = True) -> pd.DataFrame:
    _, log = train_dynamic(experiment_id, episodes, models_dir=models_dir, csv_dir=csv_dir, save=save)
    table = evaluate_dynamic(experiment_id, boards, models_dir)
    if save:
        table.to_csv(Path(csv_dir or CSV_DIR) / f"extension_dynamic_board_exp{experiment_id}.csv", index=False)
        _plot_dynamic(log, table, experiment_id, Path(plots_dir or PLOTS_DIR))
    return table


def _plot_dynamic(log, table, experiment_id, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(log["episode"], moving_average(log["reward"], 200), color="#2ca02c")
    axes[0].set(title="Training reward (new board every episode)", xlabel="Episode", ylabel="Reward (moving avg)")
    axes[0].grid(alpha=0.3)
    for ax, col, title in ((axes[1], "success_rate", "Held-out boards: success rate"),
                           (axes[2], "avg_moves", "Held-out boards: average moves")):
        ax.bar(range(len(table)), table[col], color="#2ca02c")
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([p.replace(" (", "\n(") for p in table["policy"]], rotation=25, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Extension: dynamic board - {BOARD_CONFIGS[experiment_id].name}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"extension_dynamic_board_exp{experiment_id}.png", dpi=130)
    plt.close(fig)
