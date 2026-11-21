"""Extension: two-player RACE scene (human vs AI) for the Pygame game.

Both players use their own GameEngine on the SAME board (same rules and
rewards as everywhere else). Players alternate turns; first to the goal wins.
"""
import random
from typing import Optional

import pygame

from algorithms.registry import DISPLAY_NAMES, ModelMissingError, load_agent
from configs.board_configs import get_board_config
from environment.board_generator import build_board
from environment.game_engine import GameEngine
from visualization.game_renderer import (
    BAD, BG, GOOD, MUTED, PANEL_BG, TEXT, WARN, BoardRenderer, Button, TokenAnimator, draw_dice, draw_text,
    walk_path, wrap_text,
)

BOARD_AREA = pygame.Rect(20, 20, 700, 720)
PANEL_X, PANEL_W = 745, 415
ROLL_TIME, AI_DELAY = 0.5, 0.6
COLORS = [(225, 55, 65), (60, 120, 235)]   # human = red, AI = blue


class RaceScene:
    def __init__(self, exp_id: int, agent_key: str):
        self.exp_id, self.agent_key = exp_id, agent_key
        self.config = get_board_config(exp_id)
        self.board = build_board(self.config)
        self.renderer = BoardRenderer(self.board, BOARD_AREA)
        self.names = ["You", DISPLAY_NAMES[agent_key]]
        self.error: Optional[str] = None
        self.agent = None
        try:
            self.agent = load_agent(agent_key, exp_id)
        except (ModelMissingError, FileNotFoundError, ValueError) as err:
            self.error = str(err)
        except ImportError as err:
            self.error = f"{DISPLAY_NAMES[agent_key]} needs PyTorch ({err}). Run: pip install -r requirements.txt"
        self.dice_btns = [Button((PANEL_X + i * 54, 470, 48, 46), str(i + 1)) for i in range(6)]
        self.random_btn = Button((PANEL_X + 330, 470, 85, 46), "Random", color=(150, 90, 200))
        self.restart_btn = Button((PANEL_X, 690, 200, 44), "Restart", color=(60, 150, 100))
        self.menu_btn = Button((PANEL_X + 215, 690, 200, 44), "Main menu", color=(110, 70, 80))
        self.restart()

    def restart(self) -> None:
        self.engines = [GameEngine(self.board, self.config.max_steps) for _ in range(2)]
        for e in self.engines:
            e.reset()
        start = self.renderer.square_center(1)
        self.tokens = [TokenAnimator(start), TokenAnimator(start)]
        self.turn, self.mover = 0, 0
        self.pending = None
        self.rolling, self.dice_value, self.ai_timer = 0.0, None, 0.0
        self.over, self.winner = False, None
        self.message = "Your turn: choose a dice value (1-6)."

    # -- flow ------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.pending is not None

    def perform(self, player: int, dice: int) -> None:
        if self.busy or self.over or self.error or self.engines[player].done:
            return
        self.pending = self.engines[player].step(dice)
        self.mover, self.dice_value, self.rolling = player, dice, ROLL_TIME

    def _finish(self) -> None:
        res, mover = self.pending, self.mover
        self.pending = None
        who = self.names[mover]
        texts = {
            "normal": f"{who} moved to {res.position}.",
            "snake": f"{who} hit a SNAKE and slid to {res.position}.",
            "ladder": f"{who} climbed a LADDER to {res.position}.",
            "goal": f"{who} reached the goal!",
            "invalid": f"{who}: {res.dice} overshoots the goal - turn lost.",
        }
        self.message = texts[res.event]
        if res.terminated:
            self.winner, self.over = mover, True
            return
        other = 1 - mover
        if not self.engines[other].done:
            self.turn = other
        elif not self.engines[mover].done:
            self.turn = mover
        else:
            self.over = True
            self.message = "Nobody reached the goal within the move limit: draw."
            return
        if self.turn == 0:
            self.message += " Your turn."

    def update(self, dt: float) -> None:
        if self.pending is not None:
            token = self.tokens[self.mover]
            if self.rolling > 0:
                self.rolling -= dt
                if self.rolling <= 0:
                    points, durations = walk_path(self.renderer, self.pending)
                    token.play(points, durations)
            elif token.update(dt):
                self._finish()
        elif not self.over and not self.error and self.turn == 1 and self.agent is not None:
            self.ai_timer += dt
            if self.ai_timer >= AI_DELAY:
                self.ai_timer = 0.0
                action = self.agent.act(self.engines[1].position - 1, greedy=True)   # the AGENT picks the dice
                self.perform(1, int(action) + 1)

    # -- events ----------------------------------------------------------------------
    def handle_event(self, event) -> Optional[str]:
        """Returns 'menu' when the user wants to leave the race."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "menu"
            if event.key == pygame.K_r:
                self.restart()
            elif pygame.K_1 <= event.key <= pygame.K_6 and self.turn == 0:
                self.perform(0, event.key - pygame.K_0)
            elif event.key == pygame.K_SPACE and self.turn == 0:
                self.perform(0, random.randint(1, 6))            # random dice roll
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.menu_btn.hit(event.pos):
                return "menu"
            if self.restart_btn.hit(event.pos):
                self.restart()
            elif self.turn == 0:
                for i, btn in enumerate(self.dice_btns):
                    if btn.hit(event.pos):
                        self.perform(0, i + 1)
                if self.random_btn.hit(event.pos):
                    self.perform(0, random.randint(1, 6))        # random dice roll
        return None

    # -- drawing ---------------------------------------------------------------------
    def draw(self, screen, fonts) -> None:
        mouse = pygame.mouse.get_pos()
        screen.fill(BG)
        self.renderer.draw(screen, fonts)
        for i in (1, 0):   # human token drawn on top
            x, y = self.tokens[i].pos
            off = -7 if i == 0 else 7
            self.renderer.draw_token(screen, (x + off, y + off), COLORS[i])
        pygame.draw.rect(screen, PANEL_BG, (PANEL_X - 15, 10, PANEL_W + 30, 740), border_radius=14)
        draw_text(screen, f"Race: You vs {self.names[1]}", fonts["big"], TEXT, (PANEL_X, 20))

        banner = pygame.Rect(PANEL_X, 62, PANEL_W, 40)
        pygame.draw.rect(screen, (24, 28, 38), banner, border_radius=8)
        if self.error:
            text, color = "MODEL MISSING - TRAINING REQUIRED", BAD
        elif self.over:
            text, color = (("YOU WIN!", GOOD) if self.winner == 0 else
                           ((f"{self.names[1].upper()} WINS", BAD) if self.winner == 1 else ("DRAW", WARN)))
        else:
            text, color = (("Your turn", GOOD) if self.turn == 0 else (f"{self.names[1]} is playing...", WARN))
        draw_text(screen, text, fonts["normal"], color, banner.center, "center")

        y = 118
        for i in range(2):
            e = self.engines[i]
            pygame.draw.circle(screen, COLORS[i], (PANEL_X + 10, y + 12), 8)
            draw_text(screen, self.names[i], fonts["normal"], TEXT, (PANEL_X + 28, y))
            draw_text(screen, f"square {e.position}/{self.board.goal}   moves {e.moves}/{self.config.max_steps}"
                              f"   reward {e.total_reward:+.0f}", fonts["small"], MUTED, (PANEL_X + 28, y + 26))
            y += 62
        draw_text(screen, f"{self.board.size} squares: first token to the goal wins", fonts["tiny"], MUTED,
                  (PANEL_X, y))

        shown = self.dice_value if self.rolling <= 0 else 1 + int(pygame.time.get_ticks() / 60) % 6
        draw_dice(screen, pygame.Rect(PANEL_X, 300, 76, 76), shown, fonts, rolling=self.rolling > 0)
        text = self.error if self.error else self.message
        for i, line in enumerate(wrap_text(text, fonts["small"], PANEL_W - 100)[:5]):
            draw_text(screen, line, fonts["small"], BAD if self.error else TEXT, (PANEL_X + 92, 304 + i * 18))
        for btn in self.dice_btns + [self.random_btn]:
            btn.enabled = self.turn == 0 and not self.busy and not self.over and not self.error
            btn.draw(screen, fonts, mouse, "big" if btn is not self.random_btn else "normal")
        draw_text(screen, "Choose your dice (1-6) or roll a random one (Random / Space).", fonts["tiny"], MUTED,
                  (PANEL_X, 524))
        self.restart_btn.draw(screen, fonts, mouse)
        self.menu_btn.draw(screen, fonts, mouse)
