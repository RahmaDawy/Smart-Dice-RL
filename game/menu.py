"""Start menu: choose board size, agent and mode, then press Start."""
from typing import Dict, Optional

import pygame

from algorithms.registry import DISPLAY_NAMES, model_exists
from configs.board_configs import BOARD_CONFIGS
from visualization.game_renderer import BG, MUTED, TEXT, WARN, GOOD, Button, Selector, draw_text

BOARD_OPTIONS = [(f"{cfg.size} squares", exp) for exp, cfg in BOARD_CONFIGS.items()]
AGENT_OPTIONS = [("Random", "random"), ("Q-Learning", "q_learning"), ("SARSA", "sarsa"), ("DQN", "dqn")]
MODE_OPTIONS = [("Human", "human"), ("AI Demo", "ai"), ("Before/After", "compare"), ("Race vs AI", "race")]


def make_selectors(x: int, y: int, width: int, spacing: int = 66):
    """The three selector rows shared by the menu and the in-game panel."""
    return {
        "board": Selector("Board size", BOARD_OPTIONS, x, y, width),
        "agent": Selector("Agent", AGENT_OPTIONS, x, y + spacing, width),
        "mode": Selector("Mode", MODE_OPTIONS, x, y + 2 * spacing, width),
    }


class MainMenu:
    def __init__(self, size):
        self.size = size
        w, h = size
        self.selectors = make_selectors(w // 2 - 280, 250, 560, spacing=86)
        self.start_button = Button((w // 2 - 130, 560, 260, 56), "START GAME", color=(60, 160, 100))
        self.quit_button = Button((w // 2 - 130, 630, 260, 44), "Quit", color=(110, 60, 70))

    def selection(self) -> Dict:
        return {k: s.value for k, s in self.selectors.items()}

    def handle_event(self, event) -> Optional[str]:
        """Returns 'start', 'quit' or None."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for selector in self.selectors.values():
                selector.click(event.pos)
            if self.start_button.hit(event.pos):
                return "start"
            if self.quit_button.hit(event.pos):
                return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return "start"
            if event.key == pygame.K_ESCAPE:
                return "quit"
        return None

    def draw(self, surface, fonts):
        w, _ = self.size
        mouse = pygame.mouse.get_pos()
        surface.fill(BG)
        draw_text(surface, "Smart Dice RL", fonts["title"], TEXT, (w // 2, 90), "center")
        draw_text(surface, "Snakes & Ladders where the agent chooses the dice", fonts["normal"], MUTED,
                  (w // 2, 135), "center")
        draw_text(surface, "Reinforcement Learning course project", fonts["small"], MUTED, (w // 2, 165), "center")
        for selector in self.selectors.values():
            selector.draw(surface, fonts, mouse)

        sel = self.selection()
        mode_help = {
            "human": "You choose the dice value (1-6) each turn.",
            "ai": "The selected agent plays automatically, choosing the dice itself.",
            "compare": "Watch the UNTRAINED agent, then the TRAINED agent, on the same board.",
            "race": "Extension: you and the selected agent race on the same board, taking turns.",
        }[sel["mode"]]
        draw_text(surface, mode_help, fonts["small"], TEXT, (w // 2, 490), "center")
        if sel["agent"] != "random" and sel["mode"] != "human":
            ok = model_exists(sel["agent"], sel["board"])
            msg = (f"Trained {DISPLAY_NAMES[sel['agent']]} model found." if ok else
                   f"No trained {DISPLAY_NAMES[sel['agent']]} model for this board yet - training is required.")
            draw_text(surface, msg, fonts["small"], GOOD if ok else WARN, (w // 2, 520), "center")
        self.start_button.draw(surface, fonts, mouse, "big")
        self.quit_button.draw(surface, fonts, mouse)
