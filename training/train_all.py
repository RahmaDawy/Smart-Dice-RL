"""Train every algorithm on every board experiment.

    python -m training.train_all                # full run
    python -m training.train_all --quick        # fast smoke test (10 % of the episodes)
    python -m training.train_all --algos q_learning sarsa --experiments 1 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.board_configs import BOARD_CONFIGS  # noqa: E402
from configs.hyperparameters import ALGORITHMS, TRAIN_SEED  # noqa: E402
from training.common import train  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train all algorithms on all boards.")
    parser.add_argument("--algos", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--experiments", nargs="+", type=int, choices=list(BOARD_CONFIGS), default=list(BOARD_CONFIGS))
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--quick", action="store_true", help="short smoke-test run (10%% of the episodes)")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    for exp in args.experiments:
        for algo in args.algos:
            train(algo, exp, seed=args.seed, episode_scale=0.1 if args.quick else 1.0)
    print(f"\nAll training finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
