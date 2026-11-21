"""Build comparison tables: RL algorithms vs each other and vs the BFS minimum.

Reads the training CSVs, evaluation_summary.csv and runs the BFS baseline, then
writes ``results/csv/comparison_table.csv`` and ``results/csv/convergence_summary.csv``.

    python -m evaluation.compare_models
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from algorithms.registry import DISPLAY_NAMES  # noqa: E402
from configs.board_configs import BOARD_CONFIGS  # noqa: E402
from configs.hyperparameters import ALGORITHMS  # noqa: E402
from configs.paths import CSV_DIR, ensure_dirs  # noqa: E402
from evaluation.bfs_baseline import run_bfs_baseline  # noqa: E402
from evaluation.metrics import convergence_episode  # noqa: E402
from training.common import training_csv_path  # noqa: E402


def build_convergence_table(save: bool = True) -> pd.DataFrame:
    """Convergence episode, training time and final training stats per (algorithm, experiment)."""
    rows = []
    for exp, cfg in BOARD_CONFIGS.items():
        for algo in ALGORITHMS:
            path = training_csv_path(algo, exp)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            conv = convergence_episode(df["reward"].to_numpy())
            last = df.tail(max(1, len(df) // 10))
            rows.append({
                "experiment": exp, "board_size": cfg.size, "algorithm": algo,
                "training_episodes": len(df),
                "convergence_episode": conv if conv is not None else float("nan"),
                "converged": conv is not None,
                "training_time_s": float(df["elapsed_seconds"].iloc[-1]),
                "final_train_reward": float(last["reward"].mean()),
                "final_train_success_rate": float(last["success"].mean()),
            })
    table = pd.DataFrame(rows)
    if save and not table.empty:
        ensure_dirs()
        table.to_csv(CSV_DIR / "convergence_summary.csv", index=False)
    return table


def build_comparison_table(save: bool = True) -> pd.DataFrame:
    """Merge evaluation results, convergence info and the BFS minimum into one table."""
    eval_path = CSV_DIR / "evaluation_summary.csv"
    if not eval_path.exists():
        return pd.DataFrame()
    evaluation = pd.read_csv(eval_path)
    bfs = run_bfs_baseline(save=True)[["experiment", "bfs_min_moves", "bfs_total_reward"]]
    conv = build_convergence_table(save=True)
    table = evaluation.merge(bfs, on="experiment", how="left")
    if not conv.empty:
        table = table.merge(conv[["experiment", "algorithm", "training_episodes", "convergence_episode",
                                  "training_time_s"]], on=["experiment", "algorithm"], how="left")
    table["moves_over_bfs"] = table["avg_moves"] - table["bfs_min_moves"]
    table = table.sort_values(["slip_prob", "experiment", "algorithm"]).reset_index(drop=True)
    if save:
        ensure_dirs()
        table.to_csv(CSV_DIR / "comparison_table.csv", index=False)
    return table


def main() -> None:
    table = build_comparison_table()
    if table.empty:
        print("No evaluation results found. Run: python run_project.py evaluate")
        return
    shown = table.copy()
    shown["algorithm"] = shown["algorithm"].map(DISPLAY_NAMES)
    cols = ["experiment", "board_size", "algorithm", "slip_prob", "success_rate", "avg_moves", "avg_reward",
            "bfs_min_moves", "convergence_episode", "training_time_s"]
    print(shown[[c for c in cols if c in shown.columns]].round(2).to_string(index=False))
    print(f"\nSaved {CSV_DIR / 'comparison_table.csv'} and {CSV_DIR / 'convergence_summary.csv'}")


if __name__ == "__main__":
    main()
