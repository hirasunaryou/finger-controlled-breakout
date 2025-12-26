"""Pygame Breakout implementation with clear separation from vision code.

This module contains all game state and rendering logic. Input is provided to
the ``Game`` class as a normalized x-position (range ``[0.0, 1.0]``), allowing
the vision module to remain completely decoupled. Keyboard controls are kept
as a fallback for development and troubleshooting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import pygame


# Basic configuration values that make the game feel responsive yet accessible.
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 720
BRICK_ROWS = 6
BRICK_COLUMNS = 10
BRICK_WIDTH = SCREEN_WIDTH // BRICK_COLUMNS
BRICK_HEIGHT = 28
PADDLE_WIDTH = 120
PADDLE_HEIGHT = 18
BALL_RADIUS = 10
PADDLE_Y_OFFSET = 60
BALL_SPEED = 440  # Pixels per second.
PADDLE_SPEED = 540  # Keyboard fallback speed in pixels per second.
LIVES = 3


@dataclass
class Brick:
    """Lightweight brick definition used for collision detection and rendering."""

    rect: pygame.Rect
    color: Tuple[int, int, int]


class Paddle:
    """Player paddle that responds to normalized input and keyboard fallback."""

    def __init__(self) -> None:
        x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        y = SCREEN_HEIGHT - PADDLE_Y_OFFSET
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move_to_normalized(self, normalized_x: float) -> None:
        """Map normalized x (0-1) to the screen and clamp to boundaries."""

        clamped = max(0.0, min(1.0, normalized_x))
        target_center = int(clamped * SCREEN_WIDTH)
        self.rect.centerx = target_center
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

    def move_with_keyboard(self, direction: float, dt: float) -> None:
        """Move left/right with arrow keys as a reliable fallback."""

        displacement = direction * PADDLE_SPEED * dt
        self.rect.x += int(displacement)
        self.rect.clamp_ip(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))


class Ball:
    """Ball with position and velocity that can be reset after losing a life."""

    def __init__(self) -> None:
        self.position = pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        # Slight angle so the ball is not perfectly vertical at spawn.
        self.velocity = pygame.Vector2(BALL_SPEED * math.cos(math.pi / 4), -BALL_SPEED)

    @property
    def rect(self) -> pygame.Rect:
        """Convenience property to get the Rect used for collisions."""

        return pygame.Rect(
            int(self.position.x - BALL_RADIUS),
            int(self.position.y - BALL_RADIUS),
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

    def update(self, dt: float) -> None:
        """Advance position using the configured velocity."""

        self.position += self.velocity * dt

    def reset(self, paddle_rect: pygame.Rect) -> None:
        """Center the ball above the paddle after a lost life."""

        self.position = pygame.Vector2(paddle_rect.centerx, paddle_rect.top - BALL_RADIUS - 2)
        self.velocity = pygame.Vector2(BALL_SPEED * math.cos(math.pi / 4), -BALL_SPEED)


class Game:
    """Encapsulates the Breakout game loop and rendering."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Finger-Controlled Breakout")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)

        self.paddle = Paddle()
        self.ball = Ball()
        self.bricks: List[Brick] = self._create_bricks()
        self.score = 0
        self.lives = LIVES

    @staticmethod
    def _create_bricks() -> List[Brick]:
        """Generate a colorful wall of bricks."""

        bricks: List[Brick] = []
        palette: Sequence[Tuple[int, int, int]] = [
            (255, 99, 72),
            (255, 159, 67),
            (255, 205, 86),
            (75, 192, 192),
            (54, 162, 235),
            (153, 102, 255),
        ]

        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLUMNS):
                x = col * BRICK_WIDTH
                y = row * BRICK_HEIGHT + 60
                rect = pygame.Rect(x + 4, y + 4, BRICK_WIDTH - 8, BRICK_HEIGHT - 8)
                color = palette[row % len(palette)]
                bricks.append(Brick(rect, color))
        return bricks

    def _handle_wall_collisions(self) -> None:
        """Bounce off screen edges and detect when the ball is lost."""

        if self.ball.rect.left <= 0 or self.ball.rect.right >= SCREEN_WIDTH:
            self.ball.velocity.x *= -1
        if self.ball.rect.top <= 0:
            self.ball.velocity.y *= -1
        if self.ball.rect.bottom >= SCREEN_HEIGHT:
            self.lives -= 1
            self.ball.reset(self.paddle.rect)

    def _handle_paddle_collision(self) -> None:
        """Reflect the ball with slight angle adjustments based on impact point."""

        if self.ball.rect.colliderect(self.paddle.rect) and self.ball.velocity.y > 0:
            overlap_center = self.ball.position.x - self.paddle.rect.centerx
            angle = max(-math.pi / 3, min(math.pi / 3, overlap_center / (PADDLE_WIDTH / 2) * (math.pi / 3)))
            speed = self.ball.velocity.length() or BALL_SPEED
            self.ball.velocity = pygame.Vector2(speed * math.sin(angle), -abs(speed * math.cos(angle)))

    def _handle_brick_collisions(self) -> None:
        """Remove bricks on impact and bounce the ball."""

        for brick in list(self.bricks):
            if self.ball.rect.colliderect(brick.rect):
                self.bricks.remove(brick)
                self.score += 10
                self.ball.velocity.y *= -1
                break

    def _draw_hud(self) -> None:
        """Render score and lives in the top bar."""

        score_text = self.font.render(f"Score: {self.score}", True, (240, 240, 240))
        lives_text = self.font.render(f"Lives: {self.lives}", True, (240, 240, 240))
        self.screen.blit(score_text, (20, 12))
        self.screen.blit(lives_text, (SCREEN_WIDTH - 120, 12))

    def _draw_entities(self) -> None:
        """Clear the screen and draw bricks, paddle, and ball."""

        self.screen.fill((15, 15, 25))
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick.color, brick.rect, border_radius=4)
        pygame.draw.rect(self.screen, (200, 200, 200), self.paddle.rect, border_radius=4)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball.position.x), int(self.ball.position.y)), BALL_RADIUS)
        self._draw_hud()

    def run(self, control_source: Optional[Callable[[], Optional[float]]] = None) -> None:
        """Main game loop.

        Args:
            control_source: Callable that returns a normalized x-position (``0..1``)
                for the paddle each frame. If ``None`` or if the callable returns
                ``None`` we keep the paddle still unless keyboard input is provided.
        """

        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if control_source:
                try:
                    normalized = control_source()
                except Exception:
                    # Keep the game safe even if the camera code experiences issues.
                    normalized = None
                if normalized is not None:
                    self.paddle.move_to_normalized(normalized)

            # Keyboard fallback remains active for testing without a camera.
            keys = pygame.key.get_pressed()
            direction = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                direction -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                direction += 1.0
            if direction != 0.0:
                self.paddle.move_with_keyboard(direction, dt)

            self.ball.update(dt)
            self._handle_wall_collisions()
            self._handle_paddle_collision()
            self._handle_brick_collisions()

            if self.lives <= 0 or not self.bricks:
                running = False

            self._draw_entities()
            pygame.display.flip()

        self._show_end_screen()
        pygame.quit()

    def _show_end_screen(self) -> None:
        """Display a simple end message when the loop concludes."""

        message = "You Win! Press any key to exit." if not self.bricks else "Game Over. Press any key to exit."
        prompt = self.font.render(message, True, (255, 255, 255))
        rect = prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False

            self.screen.fill((0, 0, 0))
            self.screen.blit(prompt, rect)
            pygame.display.flip()

