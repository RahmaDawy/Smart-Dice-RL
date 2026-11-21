"""The playable Smart Dice game.

    python -m game.main
    python -m game.main --board 50 --agent q_learning --mode ai      # skip the menu

The game uses the SAME GameEngine and board configs as the training code.
"""
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame  # noqa: E402

from algorithms.registry import DISPLAY_NAMES, ModelMissingError, load_agent, make_untrained_agent  # noqa: E402
from configs.board_configs import BOARD_CONFIGS, SIZE_TO_EXPERIMENT, get_board_config  # noqa: E402
from environment.board_generator import build_board  # noqa: E402
from environment.game_engine import GameEngine, StepResult  # noqa: E402
from game.menu import MainMenu, make_selectors  # noqa: E402
from game.race import RaceScene  # noqa: E402
from visualization.game_renderer import (  # noqa: E402
    BAD, BG, GOOD, MUTED, PANEL_BG, TEXT, WARN, BoardRenderer, Button, TokenAnimator, draw_dice, draw_text, load_fonts,
    walk_path, wrap_text,
)

WINDOW = (1180, 760)
BOARD_AREA = pygame.Rect(20, 20, 700, 720)
PANEL_X, PANEL_W = 745, 415
ROLL_TIME = 0.55          # seconds of dice-roll animation
STEP_TIME = 0.17          # seconds per square while walking
JUMP_TIME = 0.65          # seconds for a snake slide / ladder climb
AI_DELAY = 0.45           # pause between AI moves
PHASE_PAUSE = 1.6         # pause between "before" and "after" episodes


class GameApp:
    def __init__(self, headless_frames: Optional[int] = None):
        pygame.init()
        pygame.display.set_caption("Smart Dice RL")
        self.screen = pygame.display.set_mode(WINDOW)
        self.clock = pygame.time.Clock()
        self.fonts = load_fonts()
        self.menu = MainMenu(WINDOW)
        self.scene = "menu"
        self.running = True
        self.selectors = make_selectors(PANEL_X, 474, PANEL_W, spacing=60)
        self.restart_btn = Button((PANEL_X, 690, 200, 44), "Restart", color=(60, 150, 100))
        self.menu_btn = Button((PANEL_X + 215, 690, 200, 44), "Main menu", color=(110, 70, 80))
        self.dice_btns = [Button((PANEL_X + i * 54, 398, 48, 46), str(i + 1)) for i in range(6)]
        self.random_btn = Button((PANEL_X + 330, 398, 85, 46), "Random", color=(150, 90, 200))
        self.pause_btn = Button((PANEL_X, 398, 200, 46), "Pause", color=(150, 120, 50))
        self.step_btn = Button((PANEL_X + 215, 398, 200, 46), "Step", color=(70, 110, 170))

    # ------------------------------------------------------------------ session
    def start_session(self, exp_id: int, agent_key: str, mode: str) -> None:
        self.exp_id, self.agent_key, self.mode = exp_id, agent_key, mode
        self.config = get_board_config(exp_id)
        self.board = build_board(self.config)
        self.engine = GameEngine(self.board, self.config.max_steps)
        self.engine.reset()
        self.renderer = BoardRenderer(self.board, BOARD_AREA)
        self.token = TokenAnimator(self.renderer.square_center(1))
        for key, value in (("board", exp_id), ("agent", agent_key), ("mode", mode)):
            self.selectors[key].set_value(value)
        self.dice_value = None
        self.rolling = 0.0
        self.pending: Optional[StepResult] = None
        self.shown = {"position": 1, "moves": 0, "last_reward": None, "total": 0.0}
        self.message = "Choose a dice value." if mode == "human" else "Starting..."
        self.status, self.status_color = "Ready", MUTED
        self.error: Optional[str] = None
        self.auto, self.ai_timer = True, 0.0
        self.phase, self.phase_timer = "after", 0.0
        self.compare = {}
        self.agent, self.greedy = None, True
        self._prepare_agent()

    def _prepare_agent(self) -> None:
        """Load the agent needed for the current mode; show a clear message if it is missing."""
        self.error = None
        if self.mode == "human":
            return
        try:
            if self.mode == "compare":
                if self.agent_key == "random":
                    raise ModelMissingError("The Random agent has nothing to train. Pick Q-Learning, SARSA or DQN.")
                self.phase = "before"
                self.trained_agent = load_agent(self.agent_key, self.exp_id)
                self.agent, self.greedy = make_untrained_agent(self.agent_key, self.exp_id, seed=1), False
            else:
                self.agent, self.greedy = load_agent(self.agent_key, self.exp_id), True
        except (ModelMissingError, FileNotFoundError, ValueError) as err:
            self.error, self.agent = str(err), None
        except ImportError as err:
            self.error = f"{DISPLAY_NAMES[self.agent_key]} needs PyTorch ({err}). Run: pip install -r requirements.txt"
            self.agent = None

    # ---------------------------------------------------------------- game flow
    @property
    def busy(self) -> bool:
        return self.rolling > 0 or self.token.busy or self.pending is not None

    def perform_action(self, dice: int) -> None:
        if self.busy or self.engine.done:
            return
        self.pending = self.engine.step(dice)
        self.dice_value, self.rolling = dice, ROLL_TIME
        self.status, self.status_color = "Rolling...", WARN

    def _start_walk(self, res: StepResult) -> None:
        points, durations = walk_path(self.renderer, res, STEP_TIME, JUMP_TIME)
        self.token.play(points, durations)

    def _finish_move(self) -> None:
        res, self.pending = self.pending, None
        self.shown = {"position": res.position, "moves": res.moves, "last_reward": res.reward, "total": res.total_reward}
        texts = {
            "normal": f"Moved to square {res.position}.",
            "snake": f"SNAKE! Slid down from {res.previous_position + res.dice} to {res.position}.",
            "ladder": f"LADDER! Climbed from {res.previous_position + res.dice} to {res.position}.",
            "goal": "You reached the goal!" if self.mode == "human" else "The agent reached the goal!",
            "invalid": f"Invalid move: {res.dice} overshoots the goal. Turn lost.",
        }
        self.message = texts[res.event]
        if res.terminated:
            self.status, self.status_color = "GOAL REACHED!", GOOD
        elif res.truncated:
            self.status, self.status_color = "Out of moves (max episode length)", BAD
            self.message += f" The limit of {self.config.max_steps} moves was reached."
        else:
            self.status, self.status_color = "Playing", TEXT
        if res.done and self.mode == "compare":
            self._record_compare_result(res)

    def _record_compare_result(self, res: StepResult) -> None:
        self.compare[self.phase] = {"moves": res.moves, "reward": res.total_reward, "won": res.terminated}
        if self.phase == "before":
            self.phase_timer = PHASE_PAUSE

    def _begin_after_phase(self) -> None:
        self.phase = "after"
        self.agent, self.greedy = self.trained_agent, True
        self._reset_board_only()

    def _reset_board_only(self) -> None:
        self.engine.reset()
        self.token.place(self.renderer.square_center(1))
        self.dice_value, self.pending, self.rolling = None, None, 0.0
        self.shown = {"position": 1, "moves": 0, "last_reward": None, "total": 0.0}
        self.status, self.status_color = "Playing", TEXT
        self.message = "Trained agent (greedy policy)."

    def update(self, dt: float) -> None:
        if self.rolling > 0:
            self.rolling -= dt
            if self.rolling <= 0 and self.pending is not None:
                self._start_walk(self.pending)
                if not self.token.busy:
                    self._finish_move()
        elif self.token.busy:
            if self.token.update(dt):
                self._finish_move()
        elif self.pending is not None:
            self._finish_move()
        # automatic AI play
        if self.mode != "human" and self.agent is not None and not self.busy:
            if self.engine.done:
                if self.mode == "compare" and self.phase == "before":
                    self.phase_timer -= dt
                    if self.phase_timer <= 0:
                        self._begin_after_phase()
                return
            if self.auto:
                self.ai_timer += dt
                if self.ai_timer >= AI_DELAY:
                    self._ai_move()

    def _ai_move(self) -> None:
        self.ai_timer = 0.0
        action = self.agent.act(self.engine.position - 1, greedy=self.greedy)  # the AGENT picks the dice
        self.perform_action(int(action) + 1)

    def restart(self) -> None:
        self.launch(self.exp_id, self.agent_key, self.mode)

    def launch(self, exp_id: int, agent_key: str, mode: str) -> None:
        """Start a single-player session, or the two-player race extension."""
        if mode == "race":
            self.race = RaceScene(exp_id, agent_key)
            self.scene = "race"
        else:
            self.start_session(exp_id, agent_key, mode)
            self.scene = "game"

    # ------------------------------------------------------------------ events
    def handle_game_event(self, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.scene = "menu"
            elif event.key == pygame.K_r:
                self.restart()
            elif self.mode == "human" and pygame.K_1 <= event.key <= pygame.K_6:
                self.perform_action(event.key - pygame.K_0)
            elif self.mode == "human" and event.key == pygame.K_SPACE:
                self.perform_action(random.randint(1, 6))        # random dice roll
            elif self.mode != "human" and event.key == pygame.K_SPACE:
                self.auto = not self.auto
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        pos = event.pos
        if self.restart_btn.hit(pos):
            self.restart()
        elif self.menu_btn.hit(pos):
            self.scene = "menu"
        else:
            changed = False
            for sel in self.selectors.values():
                changed = sel.click(pos) or changed
            if changed:
                self.launch(self.selectors["board"].value, self.selectors["agent"].value,
                            self.selectors["mode"].value)
                return
        if self.mode == "human":
            for i, btn in enumerate(self.dice_btns):
                if btn.hit(pos):
                    self.perform_action(i + 1)
            if self.random_btn.hit(pos):
                self.perform_action(random.randint(1, 6))        # random dice roll
        elif self.agent is not None:
            if self.pause_btn.hit(pos):
                self.auto = not self.auto
            elif self.step_btn.hit(pos) and not self.auto and not self.busy and not self.engine.done:
                self._ai_move()

    # ------------------------------------------------------------------ drawing
    def draw_game(self) -> None:
        f, mouse = self.fonts, pygame.mouse.get_pos()
        self.screen.fill(BG)
        self.renderer.draw(self.screen, f)
        self.renderer.draw_token(self.screen, self.token.pos)
        pygame.draw.rect(self.screen, PANEL_BG, (PANEL_X - 15, 10, PANEL_W + 30, 740), border_radius=14)

        draw_text(self.screen, "Smart Dice RL", f["big"], TEXT, (PANEL_X, 20))
        banner = pygame.Rect(PANEL_X, 62, PANEL_W, 40)
        pygame.draw.rect(self.screen, (24, 28, 38), banner, border_radius=8)
        if self.error:
            draw_text(self.screen, "MODEL MISSING - TRAINING REQUIRED", f["normal"], BAD, banner.center, "center")
        else:
            draw_text(self.screen, self.status, f["normal"], self.status_color, banner.center, "center")

        agent_label = "You" if self.mode == "human" else DISPLAY_NAMES[self.agent_key]
        mode_label = {"human": "Human", "ai": "AI Demo", "compare": "Before / After"}[self.mode]
        last = self.shown["last_reward"]
        info = [
            ("Mode", mode_label), ("Player", agent_label),
            ("Board", f"{self.board.size} squares ({len(self.board.snakes)} snakes, {len(self.board.ladders)} ladders)"),
            ("Position", f"{self.shown['position']} / {self.board.goal}"),
            ("Moves", f"{self.shown['moves']} / {self.config.max_steps}"),
            ("Last reward", "-" if last is None else f"{last:+.0f}"),
            ("Total reward", f"{self.shown['total']:+.0f}"),
        ]
        y = 112
        for label, value in info:
            draw_text(self.screen, label, f["small"], MUTED, (PANEL_X, y + 2))
            color = TEXT
            if label == "Last reward" and last is not None:
                color = GOOD if last > 0 else BAD
            draw_text(self.screen, value, f["normal"], color, (PANEL_X + 130, y))
            y += 27

        # dice + message area
        draw_dice(self.screen, pygame.Rect(PANEL_X, 312, 76, 76), self.dice_value if self.rolling <= 0 else
                  1 + int(pygame.time.get_ticks() / 60) % 6, f, rolling=self.rolling > 0)
        text = self.error if self.error else self.message
        lines = wrap_text(text, f["small"], PANEL_W - 100)
        for i, line in enumerate(lines[:4]):
            draw_text(self.screen, line, f["small"], BAD if self.error else TEXT, (PANEL_X + 92, 314 + i * 18))
        if self.mode == "compare":
            if self.compare:
                for i, key in enumerate(("before", "after")):
                    if key in self.compare:
                        c = self.compare[key]
                        t = f"{key.upper()}: {c['moves']} moves, reward {c['reward']:+.0f}, {'GOAL' if c['won'] else 'FAILED'}"
                        draw_text(self.screen, t, f["tiny"], GOOD if c["won"] else BAD, (PANEL_X + 92, 360 + i * 15))
            else:
                draw_text(self.screen, "BEFORE training (untrained agent, epsilon=1.0)", f["tiny"], WARN,
                          (PANEL_X + 92, 360))
            if self.phase == "after" and "after" not in self.compare:
                draw_text(self.screen, "AFTER training (greedy policy)", f["tiny"], GOOD, (PANEL_X + 92, 375))

        # controls
        if self.mode == "human":
            for btn in self.dice_btns + [self.random_btn]:
                btn.enabled = not self.busy and not self.engine.done
                btn.draw(self.screen, f, mouse, "big" if btn is not self.random_btn else "normal")
            draw_text(self.screen, "Choose your dice (1-6 / keys 1-6) or roll a random one (Random / Space).", f["tiny"], MUTED,
                      (PANEL_X, 450))
        else:
            self.pause_btn.text = "Resume (Space)" if not self.auto else "Pause (Space)"
            self.pause_btn.enabled = self.agent is not None
            self.step_btn.enabled = self.agent is not None and not self.auto and not self.busy and not self.engine.done
            self.pause_btn.draw(self.screen, f, mouse)
            self.step_btn.draw(self.screen, f, mouse)
        for sel in self.selectors.values():
            sel.draw(self.screen, f, mouse)
        self.restart_btn.draw(self.screen, f, mouse)
        self.menu_btn.draw(self.screen, f, mouse)

    # --------------------------------------------------------------------- loop
    def run(self, auto_start: Optional[dict] = None) -> None:
        if auto_start:
            self.launch(auto_start["board"], auto_start["agent"], auto_start["mode"])
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif self.scene == "menu":
                    action = self.menu.handle_event(event)
                    if action == "quit":
                        self.running = False
                    elif action == "start":
                        sel = self.menu.selection()
                        self.launch(sel["board"], sel["agent"], sel["mode"])
                elif self.scene == "race":
                    if self.race.handle_event(event) == "menu":
                        self.scene = "menu"
                else:
                    self.handle_game_event(event)
            if self.scene == "menu":
                self.menu.draw(self.screen, self.fonts)
            elif self.scene == "race":
                self.race.update(dt)
                self.race.draw(self.screen, self.fonts)
            else:
                self.update(dt)
                self.draw_game()
            pygame.display.flip()
        pygame.quit()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Smart Dice RL - playable game")
    parser.add_argument("--board", type=int, choices=sorted(SIZE_TO_EXPERIMENT), help="board size (skip the menu)")
    parser.add_argument("--agent", choices=["random", "q_learning", "sarsa", "dqn"], default="random")
    parser.add_argument("--mode", choices=["human", "ai", "compare", "race"], default="human")
    args = parser.parse_args(argv)
    start = None
    if args.board:
        start = {"board": SIZE_TO_EXPERIMENT[args.board], "agent": args.agent, "mode": args.mode}
    GameApp().run(auto_start=start)


if __name__ == "__main__":
    main()
