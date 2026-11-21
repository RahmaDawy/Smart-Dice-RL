"""Run the optional extensions.

    python run_project.py extensions --which all
    python -m extensions.cli --which dynamic --experiment 2
    python -m extensions.cli --which all --quick
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.board_configs import BOARD_CONFIGS  # noqa: E402

pd.set_option("display.width", 200)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Smart Dice RL - optional extensions")
    parser.add_argument("--which", choices=["all", "random_dice", "dynamic", "multiplayer"], default="all")
    parser.add_argument("--experiment", type=int, choices=list(BOARD_CONFIGS), default=2,
                        help="board for random_dice / dynamic (default 2 = 50 squares)")
    parser.add_argument("--quick", action="store_true", help="short smoke-test run (10%% of the episodes)")
    args = parser.parse_args(argv)
    scale = 0.1 if args.quick else 1.0

    if args.which in ("all", "random_dice"):
        from extensions.random_dice import run_random_dice
        print("\n=== Extension: random dice (move / hold) ===")
        print(run_random_dice(args.experiment, episodes=int(20000 * scale)).round(3).to_string(index=False))
    if args.which in ("all", "dynamic"):
        from extensions.dynamic_board import run_dynamic
        print("\n=== Extension: dynamic board ===")
        print(run_dynamic(args.experiment, episodes=int(6000 * scale)).round(3).to_string(index=False))
    if args.which in ("all", "multiplayer"):
        from extensions.multiplayer import run_multiplayer
        print("\n=== Extension: multiplayer race (needs the core models) ===")
        table = run_multiplayer(games=100)
        print(table.round(2).to_string(index=False) if not table.empty else "No trained models found.")


if __name__ == "__main__":
    main()
