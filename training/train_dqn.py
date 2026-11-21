"""Train DQN on one or all board experiments.

Run from the project root:
    python -m training.train_dqn --experiment all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "python training/train_dqn.py"

from training.common import run_cli  # noqa: E402


def main(argv=None) -> None:
    run_cli("dqn", argv)


if __name__ == "__main__":
    main()
