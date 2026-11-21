"""Pygame drawing helpers: board, snakes, ladders, token, dice and UI widgets.

This module only DRAWS. All game rules live in environment/game_engine.py.
"""
import math
from typing import Dict, List, Sequence, Tuple

import pygame

from environment.board_generator import Board

# --- palette -------------------------------------------------------------------
BG = (24, 28, 38)
PANEL_BG = (34, 40, 54)
TEXT = (235, 238, 245)
MUTED = (150, 160, 180)
GOOD = (90, 200, 120)
BAD = (235, 90, 90)
WARN = (240, 190, 70)
ACCENT = (80, 140, 235)
SQUARE_A = (238, 226, 196)
SQUARE_B = (222, 205, 168)
START_COLOR = (150, 215, 160)
GOAL_COLOR = (250, 215, 100)
SNAKE_COLORS = [(46, 160, 90), (150, 70, 190), (230, 120, 40), (40, 150, 200), (200, 60, 90),
                (120, 160, 40), (60, 90, 200), (170, 110, 60)]
LADDER_COLOR = (140, 90, 45)
TOKEN_COLOR = (225, 55, 65)


def load_fonts() -> Dict[str, pygame.font.Font]:
    """Fonts that exist on Windows 11 (Segoe UI) with safe fallbacks."""
    names = "segoeui,arial,helvetica,dejavusans"
    return {
        "title": pygame.font.SysFont(names, 34, bold=True),
        "big": pygame.font.SysFont(names, 26, bold=True),
        "normal": pygame.font.SysFont(names, 19),
        "small": pygame.font.SysFont(names, 15),
        "tiny": pygame.font.SysFont(names, 13),
    }


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> List[str]:
    """Split ``text`` into lines that fit ``max_width`` pixels."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(surface, text, font, color, pos, anchor="topleft"):
    img = font.render(str(text), True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


# --- UI widgets ----------------------------------------------------------------
class Button:
    def __init__(self, rect: Sequence[int], text: str, enabled: bool = True, color=ACCENT):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.enabled = enabled
        self.color = color

    def draw(self, surface, fonts, mouse_pos, font_key="normal"):
        hover = self.enabled and self.rect.collidepoint(mouse_pos)
        base = self.color if self.enabled else (70, 74, 88)
        color = tuple(min(255, c + 25) for c in base) if hover else base
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, (255, 255, 255, 40), self.rect, width=1, border_radius=8)
        draw_text(surface, self.text, fonts[font_key], TEXT if self.enabled else MUTED, self.rect.center, "center")

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class Selector:
    """A labelled row of toggle buttons (one option selected at a time)."""

    def __init__(self, label: str, options: List[Tuple[str, object]], x: int, y: int, width: int,
                 height: int = 34, selected: int = 0):
        self.label = label
        self.options = options
        self.x, self.y, self.width, self.height = x, y, width, height
        self.selected = selected

    @property
    def value(self):
        return self.options[self.selected][1]

    def _rects(self) -> List[pygame.Rect]:
        gap = 6
        w = (self.width - gap * (len(self.options) - 1)) / len(self.options)
        return [pygame.Rect(int(self.x + i * (w + gap)), self.y + 22, int(w), self.height)
                for i in range(len(self.options))]

    def set_value(self, value) -> None:
        for i, (_, v) in enumerate(self.options):
            if v == value:
                self.selected = i

    def draw(self, surface, fonts, mouse_pos):
        draw_text(surface, self.label, fonts["small"], MUTED, (self.x, self.y))
        for i, (rect, (text, _)) in enumerate(zip(self._rects(), self.options)):
            chosen = i == self.selected
            hover = rect.collidepoint(mouse_pos)
            color = ACCENT if chosen else ((70, 78, 98) if hover else (52, 60, 78))
            pygame.draw.rect(surface, color, rect, border_radius=8)
            draw_text(surface, text, fonts["small"], TEXT, rect.center, "center")

    def click(self, pos) -> bool:
        """Returns True if the click changed the selection."""
        for i, rect in enumerate(self._rects()):
            if rect.collidepoint(pos) and i != self.selected:
                self.selected = i
                return True
        return False


# --- board ---------------------------------------------------------------------
def board_columns(size: int) -> int:
    return 6 if size <= 30 else 10


class BoardRenderer:
    """Draws a Board inside a pygame.Rect and converts squares to pixel positions."""

    def __init__(self, board: Board, area: pygame.Rect):
        self.board = board
        self.cols = board_columns(board.size)
        self.rows = math.ceil(board.size / self.cols)
        self.cell = int(min(area.width / self.cols, area.height / self.rows))
        self.origin_x = area.x + (area.width - self.cell * self.cols) // 2
        self.origin_y = area.y + (area.height - self.cell * self.rows) // 2
        self.snake_color = {head: SNAKE_COLORS[i % len(SNAKE_COLORS)] for i, head in enumerate(sorted(board.snakes))}
        self._static = None

    def square_center(self, square: int) -> Tuple[float, float]:
        """Pixel centre of a square (1 is bottom-left, rows zig-zag upwards)."""
        index = square - 1
        row, col = divmod(index, self.cols)
        if row % 2 == 1:
            col = self.cols - 1 - col
        x = self.origin_x + col * self.cell + self.cell / 2
        y = self.origin_y + (self.rows - 1 - row) * self.cell + self.cell / 2
        return x, y

    # -- static layer (squares, numbers, snakes, ladders) is cached ------------
    def draw(self, surface, fonts):
        if self._static is None:
            self._static = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            self._render_static(self._static, fonts)
        surface.blit(self._static, (0, 0))

    def _render_static(self, surf, fonts):
        cell = self.cell
        num_font = fonts["small"] if cell < 90 else fonts["normal"]
        for square in range(1, self.board.size + 1):
            cx, cy = self.square_center(square)
            rect = pygame.Rect(int(cx - cell / 2), int(cy - cell / 2), cell, cell)
            row, col = divmod(square - 1, self.cols)
            color = SQUARE_A if (row + col) % 2 == 0 else SQUARE_B
            if square == 1:
                color = START_COLOR
            elif square == self.board.size:
                color = GOAL_COLOR
            pygame.draw.rect(surf, color, rect)
            pygame.draw.rect(surf, (120, 105, 80), rect, width=1)
            draw_text(surf, square, num_font, (70, 60, 45), (rect.x + 5, rect.y + 3))
        sx, sy = self.square_center(1)
        draw_text(surf, "START", fonts["tiny"], (30, 90, 50), (sx, sy + cell * 0.28), "center")
        gx, gy = self.square_center(self.board.size)
        draw_text(surf, "GOAL", fonts["tiny"], (120, 80, 0), (gx, gy + cell * 0.28), "center")
        for bottom, top in self.board.ladders.items():
            self._draw_ladder(surf, bottom, top)
        for head, tail in self.board.snakes.items():
            self._draw_snake(surf, head, tail)

    def _draw_ladder(self, surf, bottom, top):
        x1, y1 = self.square_center(bottom)
        x2, y2 = self.square_center(top)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length          # unit normal
        off = self.cell * 0.14
        width = max(3, int(self.cell * 0.07))
        for sign in (-1, 1):
            pygame.draw.line(surf, LADDER_COLOR, (x1 + sign * nx * off, y1 + sign * ny * off),
                             (x2 + sign * nx * off, y2 + sign * ny * off), width)
        rungs = max(3, int(length / (self.cell * 0.35)))
        for i in range(1, rungs):
            t = i / rungs
            cx, cy = x1 + dx * t, y1 + dy * t
            pygame.draw.line(surf, (190, 140, 80), (cx - nx * off, cy - ny * off), (cx + nx * off, cy + ny * off),
                             max(2, width - 1))

    def _draw_snake(self, surf, head, tail):
        color = self.snake_color[head]
        x1, y1 = self.square_center(head)
        x2, y2 = self.square_center(tail)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        waves = max(2, int(length / (self.cell * 1.3)))
        amp = self.cell * 0.22
        pts = []
        steps = 40
        for i in range(steps + 1):
            t = i / steps
            sway = math.sin(t * math.pi * 2 * waves) * amp * math.sin(t * math.pi) ** 0.5
            pts.append((x1 + dx * t + nx * sway, y1 + dy * t + ny * sway))
        for i in range(steps):
            thickness = max(2, int(self.cell * (0.22 - 0.15 * i / steps)))
            pygame.draw.line(surf, color, pts[i], pts[i + 1], thickness)
            pygame.draw.circle(surf, color, (int(pts[i][0]), int(pts[i][1])), thickness // 2)
        pygame.draw.circle(surf, tuple(min(255, c + 60) for c in color), (int(pts[0][0]), int(pts[0][1])),
                           int(self.cell * 0.16))
        hx, hy = pts[0]
        eye = max(2, int(self.cell * 0.04))
        for s in (-1, 1):
            pygame.draw.circle(surf, (255, 255, 255), (int(hx + nx * s * self.cell * 0.07), int(hy + ny * s * self.cell * 0.07)), eye + 1)
            pygame.draw.circle(surf, (0, 0, 0), (int(hx + nx * s * self.cell * 0.07), int(hy + ny * s * self.cell * 0.07)), max(1, eye - 1))
        pygame.draw.circle(surf, (20, 20, 20), (int(x2), int(y2)), max(2, int(self.cell * 0.04)))

    def draw_token(self, surface, pos: Tuple[float, float], color=TOKEN_COLOR):
        radius = max(8, int(self.cell * 0.26))
        x, y = int(pos[0]), int(pos[1])
        pygame.draw.circle(surface, (0, 0, 0), (x + 2, y + 3), radius)       # shadow
        pygame.draw.circle(surface, color, (x, y), radius)
        pygame.draw.circle(surface, (255, 255, 255), (x, y), radius, 2)
        pygame.draw.circle(surface, (255, 150, 155), (x - radius // 3, y - radius // 3), max(2, radius // 4))


# --- dice ----------------------------------------------------------------------
_PIPS = {
    1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
    6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
}


def draw_dice(surface, rect: pygame.Rect, value, fonts, rolling: bool = False):
    pygame.draw.rect(surface, (250, 250, 250), rect, border_radius=14)
    pygame.draw.rect(surface, (90, 90, 100), rect, width=3, border_radius=14)
    if value is None:
        draw_text(surface, "?", fonts["title"], (120, 120, 130), rect.center, "center")
        return
    step = rect.width * 0.26
    radius = max(4, int(rect.width * 0.08))
    for px, py in _PIPS[int(value)]:
        pygame.draw.circle(surface, (200, 50, 60) if rolling else (30, 30, 40),
                           (int(rect.centerx + px * step), int(rect.centery + py * step)), radius)


# --- token animation (purely visual) ---------------------------------------------
class TokenAnimator:
    """Moves a token along waypoints."""

    def __init__(self, pos):
        self.pos = pos
        self.segments = []   # list of (start, end, duration)
        self.t = 0.0

    @property
    def busy(self) -> bool:
        return bool(self.segments)

    def place(self, pos):
        self.pos, self.segments, self.t = pos, [], 0.0

    def play(self, waypoints, durations):
        self.segments = [(waypoints[i], waypoints[i + 1], durations[i]) for i in range(len(durations))]
        self.t = 0.0

    def update(self, dt: float) -> bool:
        """Advance the animation; returns True when it just finished."""
        if not self.segments:
            return False
        self.t += dt
        while self.segments and self.t >= self.segments[0][2]:
            self.t -= self.segments[0][2]
            self.pos = self.segments[0][1]
            self.segments.pop(0)
        if self.segments:
            a, b, d = self.segments[0]
            k = self.t / d
            k = k * k * (3 - 2 * k)  # smoothstep
            self.pos = (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
            return False
        self.t = 0.0
        return True


def walk_path(renderer: "BoardRenderer", res, step_time: float = 0.17, jump_time: float = 0.65):
    """Waypoints and durations that animate one StepResult (walk, then snake/ladder jump)."""
    if res.event == "invalid":
        x, y = renderer.square_center(res.previous_position)
        return [(x, y), (x + 8, y), (x - 8, y), (x, y)], [0.07, 0.12, 0.07]
    points = [renderer.square_center(res.previous_position)]
    durations = []
    for sq in range(res.previous_position + 1, res.previous_position + res.dice + 1):
        points.append(renderer.square_center(sq))
        durations.append(step_time)
    if res.event in ("snake", "ladder"):
        points.append(renderer.square_center(res.position))
        durations.append(jump_time)
    return points, durations
