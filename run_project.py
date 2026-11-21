"""One entry point for the whole project (run from the project root).

    python run_project.py train --algo all --experiment all     # train
    python run_project.py train --algo dqn --experiment 2       # train one
    python run_project.py evaluate                              # greedy evaluation + BFS + comparison tables
    python run_project.py plots                                 # all PNG plots
    python run_project.py game                                  # Pygame game (menu)
    python run_project.py test                                  # pytest
    python run_project.py all                                   # train + evaluate + plots
    python run_project.py all --quick                           # fast smoke run (10 % of the episodes)
    python run_project.py extensions --which all                # optional extras: random dice, dynamic board, multiplayer
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_train(args) -> None:
    from configs.board_configs import BOARD_CONFIGS
    from configs.hyperparameters import ALGORITHMS
    from training.common import train
    algos = list(ALGORITHMS) if args.algo == "all" else [args.algo]
    exps = list(BOARD_CONFIGS) if args.experiment == "all" else [int(args.experiment)]
    for exp in exps:
        for algo in algos:
            train(algo, exp, episode_scale=0.1 if args.quick else 1.0)


def cmd_evaluate(args) -> None:
    from evaluation import compare_models, evaluate
    summary = evaluate.evaluate_all(episodes=args.episodes)
    if summary.empty:
        print("No trained models found. Run: python run_project.py train")
        return
    compare_models.main()


def cmd_plots(_args) -> None:
    from visualization import training_plots
    training_plots.main()


def cmd_game(_args) -> None:
    from game.main import main
    main([])


def cmd_test(_args) -> None:
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT))


def cmd_extensions(args) -> None:
    from extensions.cli import main as extensions_main
    extensions_main(["--which", args.which, "--experiment", str(args.experiment)] + (["--quick"] if args.quick else []))


def cmd_all(args) -> None:
    cmd_train(argparse.Namespace(algo="all", experiment="all", quick=args.quick))
    cmd_evaluate(argparse.Namespace(episodes=args.episodes))
    cmd_plots(args)
    print("\nDone. Launch the game with:  python run_project.py game")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Dice RL project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train agents")
    p.add_argument("--algo", choices=["all", "q_learning", "sarsa", "dqn"], default="all")
    p.add_argument("--experiment", choices=["all", "1", "2", "3"], default="all")
    p.add_argument("--quick", action="store_true", help="10%% of the episodes (smoke test)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate trained agents and build comparison tables")
    p.add_argument("--episodes", type=int, default=100)
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("plots", help="generate all plots").set_defaults(func=cmd_plots)
    sub.add_parser("game", help="launch the Pygame game").set_defaults(func=cmd_game)
    sub.add_parser("test", help="run the unit tests").set_defaults(func=cmd_test)

    p = sub.add_parser("extensions", help="optional extras (random dice, dynamic board, multiplayer)")
    p.add_argument("--which", choices=["all", "random_dice", "dynamic", "multiplayer"], default="all")
    p.add_argument("--experiment", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_extensions)

    p = sub.add_parser("all", help="train + evaluate + plots")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
