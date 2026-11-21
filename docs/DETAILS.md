# Smart Dice RL - Snakes & Ladders with Reinforcement Learning

A university Reinforcement Learning project. The board is a Snakes-and-Ladders game, but **the agent chooses the dice value (1-6)**. Three RL algorithms (Q-Learning, SARSA, DQN) learn to reach the goal by interacting with a Gymnasium-compatible environment. A separate BFS baseline gives the minimum possible number of moves, and a Pygame game lets you play or watch the trained agents.

---------------------------------------------------------------------

## 1. Project overview

| Experiment | Board size | Snakes | Ladders | Max episode length |
|---|---:|---:|---:|---:|
| 1 | 30 | 3 | 3 | 20 moves |
| 2 | 50 | 5 | 5 | 30 moves |
| 3 | 100 | 8 | 8 | 60 moves |

Rules (all implemented once, in `environment/`):

* Start at square 1, goal is square N.
* Snakes move you back, ladders move you forward, applied immediately after landing.
* A square is never both a snake and a ladder start, and there are **no chained transitions** (no snake/ladder ends on another snake/ladder start).
* Overshooting the goal is an **invalid move**: you stay where you are, receive -10, and the turn is still consumed.
* An episode ends when the goal is reached (*terminated*) or when the maximum episode length is hit (*truncated*).
* Boards are generated reproducibly from fixed seeds in `configs/board_configs.py`, so every algorithm sees exactly the same board.

The maximum episode lengths are deliberately tight (a bit above what a random agent needs on average) so that "success" is a meaningful metric. You can change them in `configs/board_configs.py`.

## 2. Architecture

```
Smart-Dice-RL/
  configs/        board_configs.py (boards + rewards), hyperparameters.py, paths.py
  environment/    board_generator.py (Board + rules), game_engine.py (GameEngine), gym_env.py (Gymnasium wrapper)
  algorithms/     q_learning.py, sarsa.py, dqn.py, base.py (shared tabular code, random agent), registry.py (load/save by name)
  training/       common.py (shared loops), train_q_learning.py, train_sarsa.py, train_dqn.py, train_all.py
  evaluation/     evaluate.py, bfs_baseline.py, compare_models.py, metrics.py
  visualization/  training_plots.py (matplotlib), game_renderer.py (Pygame drawing)
  game/           main.py (game loop), menu.py (start menu), race.py (extension: human vs AI race)
  extensions/     random_dice.py, dynamic_board.py, multiplayer.py, cli.py  (optional extras, see section 14)
  models/         trained models  (q_learning_exp1.npz, sarsa_exp1.npz, dqn_exp1.pt, ...)
  results/csv/    all metrics as CSV      results/plots/   all figures as PNG
  tests/          pytest tests
  run_project.py  one entry point for everything
```

Design rules that keep everything consistent:

* `Board.resolve(position, dice)` is the **only** place where movement rules live. The engine, BFS, board generator and GUI all use it.
* `GameEngine` computes rewards. `SmartDiceEnv` (Gymnasium) wraps it; the GUI drives the same `GameEngine` and the same `BoardConfig`s. There is no second copy of the rules.
* Every agent has the same interface: `agent.act(state, greedy)`, `save(path)`, `load(path)`.
* Extra files beyond the requested structure: `configs/paths.py`, `algorithms/base.py`, `algorithms/registry.py`, `training/common.py`, `tests/conftest.py` (they avoid duplicated code).

## 3. Installation (Windows 11 + VS Code)

Requirements: Python 3.10 - 3.12 (64-bit) from python.org. Tick "Add Python to PATH" in the installer. No GPU is needed (CPU is used; PyTorch from PyPI is CPU-only on Windows).

Open the extracted `Smart-Dice-RL` folder in VS Code (File > Open Folder), then open a terminal (Ctrl+`):

```powershell
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# If PowerShell blocks the script, run this once in the same terminal and activate again:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code press Ctrl+Shift+P > "Python: Select Interpreter" > choose `.venv`.

Dependencies: numpy, pandas, matplotlib (results), gymnasium (environment API), pygame (game), torch (DQN), pytest (tests).

## 4. Commands (all run from the project root)

```powershell
# Everything in one go: train all 3 algorithms on all 3 boards, evaluate, make plots
python run_project.py all
python run_project.py all --quick            # 10 % of the episodes (fast smoke test)

# Train
python run_project.py train --algo all --experiment all
python run_project.py train --algo q_learning --experiment 1
python -m training.train_q_learning --experiment all
python -m training.train_sarsa --experiment 2
python -m training.train_dqn --experiment 3 --episodes 2000
python -m training.train_all

# Evaluate (greedy policy, unseen episodes) + BFS baseline + comparison tables
python run_project.py evaluate
python -m evaluation.evaluate --episodes 200
python -m evaluation.bfs_baseline
python -m evaluation.compare_models

# Plots
python run_project.py plots
python -m visualization.training_plots

# Play / watch
python run_project.py game
python -m game.main --board 50 --agent sarsa --mode ai     # skip the menu

# Tests
python run_project.py test
python -m pytest -q

# Optional extensions (section 14)
python run_project.py extensions --which all
python run_project.py extensions --which dynamic --experiment 2
python -m game.main --board 50 --agent sarsa --mode race
```

### Reproducing the experiments

Everything is seeded: boards (`seed` in `board_configs.py`), training (`TRAIN_SEED = 42`), evaluation (`EVAL_SEED_OFFSET`). Running `python run_project.py all` regenerates every model, CSV and plot. Training times depend on your CPU (tabular methods take seconds, DQN a few minutes in total).

## 5. Playing the game

`python run_project.py game` opens the menu. Choose:

* **Board size**: 30 / 50 / 100 squares (Experiments 1-3).
* **Agent**: Random, Q-Learning, SARSA, DQN.
* **Mode**:
  * **Human** - you can **choose** the dice value (buttons or keys 1-6) **or roll a random dice** (the *Random* button or `Space`) - use whichever you like on every turn. The same rewards as in training apply.
  * **AI Demo** - the selected agent loads its trained model and chooses the dice itself. Pause/Resume (Space) and Step buttons are available.
  * **Race vs AI** (extension, section 14) - you and the agent take turns on the same board; first to the goal wins.
  * **Before / After** - first one episode with the *untrained* agent (fresh agent, epsilon = 1.0, i.e. random exploration), then one episode with the *trained* agent (greedy), on the same board. The panel shows both results.

The side panel shows position, moves, last and total reward, the dice and the game status. The Board/Agent/Mode selectors can be changed during the game (the game restarts). Keys: `R` restart, `Esc` menu, `Space` pause (AI modes) or random roll (Human / Race), `1-6` choose dice (Human / Race).

If a model is missing, the game shows **"MODEL MISSING - TRAINING REQUIRED"** with the exact command to run. Models are loaded from `models/`.

## 6. The MDP

* **State**: the current square, `s = position - 1` in `{0, ..., N-1}` (the Gymnasium observation is `Discrete(N)`).
* **Action**: the dice value the agent asks for, `a` in `{0..5}` meaning dice `a + 1` (`Discrete(6)`).
* **Transition**: deterministic. `s' = resolve(s, dice)` (move, then snake/ladder jump; overshoot = stay).
* **Reward**: see below. **Discount** gamma = 0.99.
* **Terminal**: the goal square. **Truncation**: maximum episode length (the learner still bootstraps from the next state in that case, because the game itself did not end).

### Reward function and precedence

| Event | Reward |
|---|---:|
| Reach the goal | +100 |
| Ladder | +20 |
| Normal square | -1 |
| Snake | -20 |
| Invalid move (overshoot) | -10 |

Precedence: an invalid move is evaluated first (token stays, -10). Otherwise, **reaching the goal always gives +100**, whatever else is on that square. Then ladder (+20), snake (-20) and normal square (-1). Ladders never end on the goal square, so the goal reward is never hidden behind a ladder bonus.

## 7. Algorithms

**Q-Learning** (off-policy TD control). Keeps a table `Q[s, a]` and updates
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`.
The target uses the best next action even if the epsilon-greedy behaviour policy explores.

**SARSA** (on-policy TD control). Update
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))`
where `a'` is the action the epsilon-greedy policy really takes next. Exploration risk (snakes) is therefore reflected in its values.

Both tabular agents use identical hyper-parameters: alpha = 0.1, gamma = 0.99, epsilon from 1.0 decaying per episode to 0.05 (decay chosen so the floor is reached after ~60 % of training), Q-table initialised to zero. They save/load the Q-table as `.npz`.

**DQN**. A PyTorch MLP (one-hot position -> 128 -> 128 -> 6 Q-values) trained with: experience replay (50 000 transitions, batches of 64), a separate **target network** (hard update every 250 gradient steps), epsilon-greedy exploration, Huber loss, Adam (lr 1e-3), gradient clipping. Rewards are multiplied by 0.05 inside the loss for numerical stability. Weights are saved as `.pt` (loaded with `weights_only=True`). All hyper-parameters are in `configs/hyperparameters.py`.

None of the agents use BFS or any precomputed path; they only see states, rewards and termination signals from the environment.

## 8. Training, evaluation and metrics

**Training** records per episode: reward, moves, success (goal reached), epsilon and elapsed time -> `results/csv/training_<algo>_exp<k>.csv`.

**Evaluation** (`evaluation/evaluate.py`) uses separate episodes (seeds that training never uses), greedy actions, 100 episodes by default, and reports success rate, average moves, average cumulative reward. A **random agent** is evaluated as reference. Results: `evaluation_summary.csv`, `evaluation_episodes.csv`.

Important note: the core game is **deterministic**, so a greedy agent plays the identical game every time (standard deviation 0). To also get a spread, every evaluation is repeated with a *noisy dice* (`slip_prob = 0.1`: with 10 % probability the dice shows a random value instead of the chosen one). The column `slip_prob` tells the two settings apart. The agents were trained without noise, so the noisy numbers measure robustness.

**Convergence criterion** (`evaluation/metrics.py`): `plateau` = mean training reward over the last 10 % of episodes; `threshold = plateau - 10 % * |plateau|`; the convergence episode is the first episode at which the 100-episode moving average of the reward reaches the threshold **and stays above it for the next 50 episodes**. `NaN` means "did not converge". Note that DQN is trained for fewer episodes than the tabular methods (see `EPISODES`), and its episodes are not directly comparable in cost (one DQN episode is far more expensive in time) - compare the training-time column as well.

**Comparison** (`compare_models.py`): `comparison_table.csv` joins evaluation, convergence, training time and the BFS minimum; `convergence_summary.csv` holds the convergence data.

## 9. BFS baseline (not RL)

`evaluation/bfs_baseline.py` runs breadth-first search over the known board and returns the minimum number of moves (and the dice sequence) -> `results/csv/bfs_baseline.csv`. It is only a reference. **The RL agents never call it.**

* BFS minimises the *number of moves*; the RL agents maximise *reward*. These can differ: e.g. a slightly longer path through a ladder can yield a higher total reward than the shortest path, so an agent may use more moves than BFS and still earn more reward.
* Limitations: BFS needs the complete model of the world in advance; it assumes deterministic transitions (with random or noisy dice there is no single path, so you would need expectations/value iteration or learning); it ignores the reward structure; and it needs an enumerable state space. RL only needs interaction, and still works when the dynamics are stochastic or change between games.

## 10. Plots (`results/plots/`)

* `training_<algo>_exp<k>.png` - reward, moves and success rate vs. training episode (faint = raw, solid = moving average) for each algorithm and board.
* `qlearning_vs_sarsa_exp<k>.png` - Q-Learning vs SARSA.
* `all_algorithms_exp<k>.png` - all three algorithms.
* `board_size_comparison.png` - success rate, moves, reward and convergence episode across the three board sizes (with the BFS minimum).

## 11. Tests

`python -m pytest -q` (40 tests, including `tests/test_extensions.py`) covers: normal movement, snakes, ladders, invalid moves, winning, maximum episode length, board-generation rules and reproducibility, BFS, the Gymnasium API (including Gymnasium's `check_env`), training smoke tests for all three algorithms, model saving/loading round trips, missing-model errors, and a check that Q-Learning really learns the 30-square board.

## 14. Extensions (optional extras - the core project is unchanged)

The three extensions live in `extensions/` (plus `game/race.py`). The core rules, rewards, boards, models and results are untouched; the only core edits are two additive helpers (`GameEngine.hold()`, used only by the random-dice extension, and shared animation helpers moved into `visualization/game_renderer.py`). Run all of them with `python run_project.py extensions --which all` (add `--quick` for a smoke test). Outputs are prefixed `extension_` / `ext_` in `results/` and `models/`.

### 14.1 Random dice (stochastic transitions)
The environment rolls the dice, so "choose a dice value" is no longer an action. Reformulated MDP: **state** = (position, rolled value) -> `Discrete(N*6)`; **action** = `MOVE` (play the roll) or `HOLD` (skip, reward -1, new roll next turn). Same board and rewards. Tabular Q-Learning and SARSA are trained (`RandomDiceEnv`), then compared with the classic game (always move) and random decisions. Files: `extension_random_dice_exp2.csv`, `extension_random_dice_exp2.png`.
Observed in the included run (50 squares): the learned agents earn about 2.7x the reward of always-moving (about 111 vs 41) by holding when a roll would hit a snake or overshoot - but they hold often, so within the move limit their success rate is *lower* (about 0.74 vs 0.92). They optimise reward, not success; a longer move limit would change this trade-off.

### 14.2 Dynamic board (snakes and ladders change every episode)
A position-only agent memorises one board and cannot adapt. Reformulation: the **observation** describes what each dice value would do from the current square (normal / snake / ladder / goal / invalid + distance moved, plus position and distance to goal; 38 numbers, independent of the board); **actions** are the same 6 dice values. `DynamicBoardEnv` draws a new board each reset. A DQN (`VectorDQNAgent`: same replay buffer + target network, vector input) is trained on random boards and tested on 100 **held-out boards** (board seeds the training never uses), against: random, the core position-only Q-Learning agent, and the BFS optimum per board. Files: `extension_dynamic_board_exp2.csv`, `.png`, `extension_dynamic_training_exp2.csv`.
Observed (50 squares, 100 held-out boards): dynamic DQN average reward about 161 and 8.0 moves, vs random about 47 (17.7 moves), the core position-only agent about -147 (success only 0.26) and BFS about 170 (5.5 moves). The core agent is perfect on its own fixed board (1.00 success), which shows that the failure is lack of adaptation. The adaptive agent is not perfect: it only sees 6 squares ahead, and its success rate (about 0.92) is the same as random because the move limit is generous for random play on some boards - compare reward and moves, not success alone.

### 14.3 Multiple players (race)
Two players alternate turns on the same board, each with their own token and `GameEngine` (same rules/rewards); the first to reach the goal wins; tokens do not interact; no winner within the move limit = draw.
* **GUI**: menu > Mode > **Race vs AI** (human = red token, agent = blue). The human starts and, as in Human mode, can pick the dice or roll a random one. The agent loads its trained model and picks the dice itself; missing models show the "training required" message.
* **Agent vs agent**: `python run_project.py extensions --which multiplayer` plays a round robin (Random, Q-Learning, SARSA, DQN) with alternating starting player -> `extension_multiplayer.csv`.
Note: the core game is deterministic, so equally good trained agents need the same number of moves and the player who starts wins; hence the starter alternates and trained-vs-trained results are 50/50. Trained agents beat the random agent almost always.

Not included: the random-dice and dynamic-board extensions have no GUI mode (they are evaluated through CSV/plots); only the race is playable.

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python and tick "Add Python to PATH", or use `py -3.11 -m venv .venv`. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `ModuleNotFoundError: environment` (or `configs`...) | Run commands from the project root (the folder containing `run_project.py`). |
| Game says "MODEL MISSING - TRAINING REQUIRED" | Run `python run_project.py train --algo <algo> --experiment <k>` (or `python run_project.py all`). |
| DQN needs PyTorch | `pip install torch` (CPU build is fine; CUDA is never required). |
| `pip install pygame` fails | Use Python 3.10 - 3.12 and `python -m pip install --upgrade pip` first. |
| Plots empty / "No training CSV files" | Train first, then `python run_project.py plots`. |
| Game window too large for the screen | Lower Windows display scaling or edit `WINDOW` in `game/main.py`. |

## 13. What was verified

This project was generated and executed in a Linux sandbox with Python 3.12: all 40 unit tests pass, all three algorithms were trained on all three boards, evaluation / comparison tables / plots were generated, and the Pygame game was run in headless mode (screenshots checked, menu, human/AI/before-after modes and the missing-model message exercised). It has **not** been run on a real Windows 11 machine or with a physical display and mouse; paths use `pathlib` and nothing is OS specific, but if something fails on your machine please check the Troubleshooting table. The numbers in `results/` come from the real training run included in this ZIP (no hard-coded metrics); re-running training on your machine regenerates them.
