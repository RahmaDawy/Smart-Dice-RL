"""Create all result plots from the saved CSV files.

    python -m visualization.training_plots
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no window needed; works on every machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from algorithms.registry import DISPLAY_NAMES  # noqa: E402
from configs.board_configs import BOARD_CONFIGS  # noqa: E402
from configs.hyperparameters import ALGORITHMS  # noqa: E402
from configs.paths import PLOTS_DIR, ensure_dirs  # noqa: E402
from evaluation.metrics import moving_average  # noqa: E402
from training.common import training_csv_path  # noqa: E402

COLORS = {"q_learning": "#1f77b4", "sarsa": "#d62728", "dqn": "#2ca02c", "random": "#7f7f7f", "bfs": "#000000"}
PANELS = (("reward", "Total reward per episode"), ("moves", "Moves per episode"),
          ("success", "Success rate"))


def _load(algo: str, exp: int):
    path = training_csv_path(algo, exp)
    return pd.read_csv(path) if path.exists() else None


def _window(n: int) -> int:
    return int(min(100, max(5, n // 10)))


def _panel_figure(exp: int, algos, title: str, filename: str, show_raw: bool = False) -> bool:
    data = {a: _load(a, exp) for a in algos}
    data = {a: d for a, d in data.items() if d is not None}
    if not data:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (col, label) in zip(axes, PANELS):
        for algo, df in data.items():
            w = _window(len(df))
            if show_raw:
                ax.plot(df["episode"], df[col], color=COLORS[algo], alpha=0.12, linewidth=0.6)
            ax.plot(df["episode"], moving_average(df[col], w), color=COLORS[algo], linewidth=1.8,
                    label=DISPLAY_NAMES[algo])
        ax.set_xlabel("Training episode")
        ax.set_ylabel(label + (" (moving avg)" if col != "success" else " (moving avg)"))
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
        if col == "success":
            ax.set_ylim(-0.02, 1.02)
    fig.suptitle(f"{title} - {BOARD_CONFIGS[exp].name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=130)
    plt.close(fig)
    return True


def plot_board_size_comparison() -> bool:
    """Compare boards 30/50/100 for each algorithm (evaluation metrics + convergence)."""
    from evaluation.compare_models import build_comparison_table
    table = build_comparison_table(save=True)
    if table.empty:
        return False
    det = table[(table["slip_prob"] == 0.0) & (table["algorithm"].isin(ALGORITHMS))]
    if det.empty:
        return False
    metrics = (("success_rate", "Evaluation success rate"), ("avg_moves", "Evaluation avg. moves"),
               ("avg_reward", "Evaluation avg. reward"), ("convergence_episode", "Convergence episode"))
    sizes = [BOARD_CONFIGS[e].size for e in BOARD_CONFIGS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    width = 0.25
    for ax, (col, label) in zip(axes.ravel(), metrics):
        for i, algo in enumerate(ALGORITHMS):
            sub = det[det["algorithm"] == algo].set_index("board_size")
            vals = [sub[col].get(s, np.nan) for s in sizes]
            ax.bar(np.arange(len(sizes)) + (i - 1) * width, vals, width, color=COLORS[algo], label=DISPLAY_NAMES[algo])
        if col == "avg_moves":
            bfs = det.drop_duplicates("board_size").set_index("board_size")["bfs_min_moves"]
            ax.plot(np.arange(len(sizes)), [bfs.get(s, np.nan) for s in sizes], "k--o", label="BFS minimum")
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels([f"{s} squares" for s in sizes])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Comparison across board sizes (deterministic evaluation)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "board_size_comparison.png", dpi=130)
    plt.close(fig)
    return True


def generate_all_plots() -> list:
    """Generate every plot that the available CSV files allow; returns the file names written."""
    ensure_dirs()
    written = []
    for exp in BOARD_CONFIGS:
        for algo in ALGORITHMS:
            name = f"training_{algo}_exp{exp}.png"
            if _panel_figure(exp, [algo], DISPLAY_NAMES[algo], name, show_raw=True):
                written.append(name)
        for algos, title, name in (
            (("q_learning", "sarsa"), "Q-Learning vs SARSA", f"qlearning_vs_sarsa_exp{exp}.png"),
            (ALGORITHMS, "All algorithms", f"all_algorithms_exp{exp}.png"),
        ):
            if _panel_figure(exp, algos, title, name):
                written.append(name)
    if plot_board_size_comparison():
        written.append("board_size_comparison.png")
    return written


def main() -> None:
    written = generate_all_plots()
    if not written:
        print("No training CSV files found. Train first (see README).")
        return
    print(f"Wrote {len(written)} plots to {PLOTS_DIR}:")
    for name in written:
        print("  -", name)


if __name__ == "__main__":
    main()
