"""Pygame Breakout implementation with clear separation from vision code.

This module contains all game state and rendering logic. Input is provided to
the ``Game`` class as a normalized x-position (range ``[0.0, 1.0]``), allowing
the vision module to remain completely decoupled. Keyboard controls are kept
as a fallback for development and troubleshooting.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from src.calibration import PersistedState, persist_state
from src.control_types import ControlSource, ControlState


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
TRAIL_LENGTH = 14
FLASH_DURATION = 0.18
SHAKE_DECAY = 6.0
PARTICLE_LIFETIME = 0.5
PARTICLE_COUNT = 12
PARTICLE_SPEED = 260
STAR_COUNT = 140


@dataclass
class Brick:
    """Lightweight brick definition used for collision detection and rendering."""

    rect: pygame.Rect
    color: Tuple[int, int, int]


@dataclass
class Particle:
    """Tiny circle used for brick-break bursts."""

    position: pygame.Vector2
    velocity: pygame.Vector2
    lifetime: float


class Paddle:
    """Player paddle that responds to normalized input and keyboard fallback."""

    def __init__(self) -> None:
        x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        y = SCREEN_HEIGHT - PADDLE_Y_OFFSET
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move_to_normalized(self, normalized_x: float) -> None:
        """Map normalized x (0-1) to the screen and clamp to boundaries."""

        clamped = max(0.0, min(1.0, normalized_x))
        # Map to the paddle's reachable center range so 0.0 and 1.0 hit the edges
        # without flattening the response near the walls.
        min_cx = PADDLE_WIDTH // 2
        max_cx = SCREEN_WIDTH - PADDLE_WIDTH // 2
        self.rect.centerx = int(min_cx + clamped * (max_cx - min_cx))
        # clamp_ip remains as a safety net for any rounding edge cases.
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
        direction = pygame.Vector2(1, -1).normalize()
        self.velocity = direction * BALL_SPEED

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
        direction = pygame.Vector2(1, -1).normalize()
        self.velocity = direction * BALL_SPEED


class Game:
    """Encapsulates the Breakout game loop and rendering."""

    def __init__(self, persisted_state: Optional[PersistedState] = None) -> None:
        pygame.init()
        pygame.display.set_caption("Finger-Controlled Breakout")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        # Slightly more playful fonts improve readability and fit the arcade feel.
        self.font = pygame.font.SysFont("montserrat", 22)
        self.big_font = pygame.font.SysFont("montserrat", 36, bold=True)
        self.hud_font = pygame.font.SysFont("montserrat", 20, bold=True)

        self.paddle = Paddle()
        self.ball = Ball()
        self.bricks: List[Brick] = self._create_bricks()
        self.score = 0
        self.lives = LIVES
        # Ball starts stuck to the paddle until the player launches it.
        self.ball_stuck = True
        self.ball_caught = False
        self.caught_offset = 0.0
        self.ball_trail: Deque[pygame.Vector2] = deque(maxlen=TRAIL_LENGTH)
        self.hit_flash_timer = 0.0
        self.hit_flash_position: Optional[pygame.Vector2] = None
        self.particles: List[Particle] = []
        self.shake_offset = pygame.Vector2(0, 0)
        self.shake_timer = 0.0
        # Pre-rendered background with a light gradient and tiny stars keeps
        # per-frame work low while making the scene feel less flat.
        self.background = self._build_background()
        self.fps_display = 0.0

        # Persistence
        self.persisted_state = persisted_state or PersistedState()
        self.best_score = self.persisted_state.best_score
        self.last_score = self.persisted_state.last_score

        # Lightweight generated sound effects (simple sine beeps).
        self.sounds = {}
        try:
            pygame.mixer.init()
            self.sounds = {
                "paddle": self._generate_tone(660, 0.08),
                "brick": self._generate_tone(880, 0.09),
                "wall": self._generate_tone(520, 0.06),
                "life_lost": self._generate_tone(180, 0.18),
                "catch": self._generate_tone(320, 0.1),
            }
        except pygame.error:
            # Audio can fail in some headless environments; gameplay continues without sound.
            self.sounds = {}

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

    def _generate_tone(self, frequency: float, duration: float) -> pygame.mixer.Sound:
        """Create a short sine wave beep without external assets."""

        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = 0.4 * np.sin(2 * math.pi * frequency * t)
        # Simple fade-out to prevent clicks.
        tone *= np.linspace(1, 0.2, tone.size)
        audio = np.int16(tone * 32767)
        return pygame.mixer.Sound(audio)

    def _build_background(self) -> pygame.Surface:
        """Generate a single gradient + star field surface for reuse each frame."""

        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        top = np.array([18, 24, 48], dtype=float)
        bottom = np.array([8, 8, 16], dtype=float)
        for y in range(SCREEN_HEIGHT):
            t = y / max(1, SCREEN_HEIGHT - 1)
            color = (top * (1 - t) + bottom * t).astype(int)
            pygame.draw.line(surface, color.tolist(), (0, y), (SCREEN_WIDTH, y))

        # Sprinkle a handful of stars; tiny rectangles are faster than circles.
        for _ in range(STAR_COUNT):
            x = random.randint(0, SCREEN_WIDTH - 1)
            y = random.randint(0, SCREEN_HEIGHT - 1)
            size = random.choice([1, 1, 2])
            brightness = random.randint(180, 255)
            pygame.draw.rect(surface, (brightness, brightness, 255), pygame.Rect(x, y, size, size))
        return surface

    def _lighten(self, color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
        """Lift a color toward white while clamping to valid ranges."""

        r, g, b = color
        return (min(255, r + amount), min(255, g + amount), min(255, b + amount))

    def _handle_wall_collisions(self) -> None:
        """Bounce off screen edges and detect when the ball is lost."""

        if self.ball.rect.left <= 0 or self.ball.rect.right >= SCREEN_WIDTH:
            self.ball.velocity.x *= -1
            self._play_sound("wall")
        if self.ball.rect.top <= 0:
            self.ball.velocity.y *= -1
            self._play_sound("wall")
        if self.ball.rect.bottom >= SCREEN_HEIGHT:
            self.lives -= 1
            self.ball.reset(self.paddle.rect)
            self.ball_stuck = True
            self.ball_caught = False
            self.ball.velocity.update(0, 0)
            self.ball_trail.clear()
            self._trigger_shake(intensity=10.0)
            self._play_sound("life_lost")

    def _handle_paddle_collision(self, catch_active: bool) -> None:
        """Reflect the ball or catch it depending on the current input."""

        if self.ball.rect.colliderect(self.paddle.rect) and self.ball.velocity.y > 0:
            if catch_active:
                # Stick the ball to the paddle until the player releases pinch/space.
                self.ball_caught = True
                self.caught_offset = self.ball.position.x - self.paddle.rect.centerx
                self.ball.velocity.update(0, 0)
                self._play_sound("catch")
                return

            overlap_center = self.ball.position.x - self.paddle.rect.centerx
            angle = max(-math.pi / 3, min(math.pi / 3, overlap_center / (PADDLE_WIDTH / 2) * (math.pi / 3)))
            speed = self.ball.velocity.length() or BALL_SPEED
            self.ball.velocity = pygame.Vector2(speed * math.sin(angle), -abs(speed * math.cos(angle)))
            self.hit_flash_position = pygame.Vector2(self.ball.position)
            self.hit_flash_timer = FLASH_DURATION
            self._play_sound("paddle")

    def _handle_brick_collisions(self) -> None:
        """Remove bricks on impact and bounce the ball."""

        for brick in list(self.bricks):
            if self.ball.rect.colliderect(brick.rect):
                self.bricks.remove(brick)
                self.score += 10
                self.ball.velocity.y *= -1
                self.hit_flash_position = pygame.Vector2(brick.rect.center)
                self.hit_flash_timer = FLASH_DURATION
                self._spawn_particles(pygame.Vector2(brick.rect.center))
                self._play_sound("brick")
                self._trigger_shake(intensity=6.0)
                break

    def _spawn_particles(self, position: pygame.Vector2) -> None:
        """Emit a handful of tiny particles when a brick breaks."""

        for _ in range(PARTICLE_COUNT):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(PARTICLE_SPEED * 0.4, PARTICLE_SPEED)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(position=position.copy(), velocity=velocity, lifetime=PARTICLE_LIFETIME))

    def _update_particles(self, dt: float) -> None:
        """Advance and decay brick-break particles."""

        alive: List[Particle] = []
        for particle in self.particles:
            particle.lifetime -= dt
            if particle.lifetime <= 0:
                continue
            particle.position += particle.velocity * dt
            particle.velocity *= 0.92
            alive.append(particle)
        self.particles = alive

    def _draw_hud(self, surface: pygame.Surface) -> None:
        """Render score, lives, and FPS with a soft shadow for readability."""

        def draw_text(text: str, pos: Tuple[int, int]) -> None:
            shadow = self.hud_font.render(text, True, (0, 0, 0))
            main = self.hud_font.render(text, True, (240, 240, 255))
            surface.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface.blit(main, pos)

        fps_text = f"{self.fps_display:5.1f} FPS"
        draw_text(f"Score: {self.score}", (20, 12))
        draw_text(f"Lives: {self.lives}", (SCREEN_WIDTH // 2 - 40, 12))
        draw_text(fps_text, (SCREEN_WIDTH - 140, 12))

    def _draw_trail(self, surface: pygame.Surface) -> None:
        """Render a fading trail behind the moving ball for visual polish."""

        if len(self.ball_trail) < 2:
            return

        trail_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for i, pos in enumerate(reversed(self.ball_trail)):
            alpha = int(255 * (1 - (i / len(self.ball_trail))))
            radius = max(3, BALL_RADIUS - i // 2)
            pygame.draw.circle(trail_surface, (255, 255, 255, max(30, alpha // 3)), (int(pos.x), int(pos.y)), radius)
        surface.blit(trail_surface, (0, 0))

    def _draw_hit_flash(self, dt: float, surface: pygame.Surface) -> None:
        """Draw a brief ring when the ball hits the paddle or a brick."""

        if self.hit_flash_timer <= 0.0 or self.hit_flash_position is None:
            return

        t = self.hit_flash_timer / FLASH_DURATION
        radius = int(BALL_RADIUS * (1 + (1 - t) * 4))
        alpha = int(255 * t)
        flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(
            flash_surface,
            (255, 220, 120, alpha),
            (int(self.hit_flash_position.x), int(self.hit_flash_position.y)),
            radius,
            width=3,
        )
        surface.blit(flash_surface, (0, 0))
        self.hit_flash_timer = max(0.0, self.hit_flash_timer - dt)

    def _draw_brick(self, canvas: pygame.Surface, brick: Brick) -> None:
        """Render a brick with a subtle top highlight for depth."""

        pygame.draw.rect(canvas, brick.color, brick.rect, border_radius=5)
        highlight = self._lighten(brick.color, 40)
        highlight_rect = pygame.Rect(brick.rect.x, brick.rect.y, brick.rect.width, max(4, brick.rect.height // 5))
        pygame.draw.rect(canvas, highlight, highlight_rect, border_radius=4)

    def _draw_entities(self, dt: float) -> None:
        """Clear the screen and draw bricks, paddle, and ball."""

        # Start from the cached background to avoid re-drawing gradients each frame.
        canvas = self.background.copy()
        for brick in self.bricks:
            self._draw_brick(canvas, brick)
        for particle in self.particles:
            alpha = int(255 * (particle.lifetime / PARTICLE_LIFETIME))
            pygame.draw.circle(
                canvas,
                (255, 255, 200, alpha),
                (int(particle.position.x), int(particle.position.y)),
                4,
            )
        pygame.draw.rect(canvas, (200, 200, 200), self.paddle.rect, border_radius=4)
        self._draw_trail(canvas)
        pygame.draw.circle(canvas, (255, 255, 255), (int(self.ball.position.x), int(self.ball.position.y)), BALL_RADIUS)
        self._draw_hit_flash(dt, canvas)
        self._draw_hud(canvas)

        offset = (int(self.shake_offset.x), int(self.shake_offset.y))
        self.screen.fill((0, 0, 0))
        self.screen.blit(canvas, offset)

    def _play_sound(self, key: str) -> None:
        sound = self.sounds.get(key)
        if sound:
            sound.play()

    def _trigger_shake(self, intensity: float) -> None:
        """Start a brief camera shake by animating an offset."""

        self.shake_timer = 0.2
        angle = random.uniform(0, 2 * math.pi)
        self.shake_offset = pygame.Vector2(math.cos(angle), math.sin(angle)) * intensity

    def _update_shake(self, dt: float) -> None:
        if self.shake_timer <= 0:
            self.shake_offset.update(0, 0)
            return
        self.shake_timer = max(0.0, self.shake_timer - dt)
        self.shake_offset *= math.exp(-SHAKE_DECAY * dt)

    def run(self, control_source: Optional[ControlSource] = None) -> None:
        """Main game loop that consumes ``ControlState`` frames."""

        running = self._show_start_screen(control_source)
        while running:
            dt = self.clock.tick(60) / 1000.0
            # Smooth FPS readout to avoid flicker in the HUD.
            instantaneous_fps = self.clock.get_fps() or 0.0
            self.fps_display = 0.9 * self.fps_display + 0.1 * instantaneous_fps
            self._update_shake(dt)
            self._update_particles(dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c and control_source:
                    try:
                        control_source.toggle_calibration()
                    except Exception:
                        pass
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and control_source:
                    try:
                        control_source.reset_calibration()
                    except Exception:
                        pass

            control_state = ControlState(x=None, pinch=False, pinch_pressed=False, pinch_released=False)
            if control_source:
                try:
                    control_state = control_source.read()
                except Exception:
                    # Keep the game safe even if the camera code experiences issues.
                    control_state = ControlState(x=None, pinch=False, pinch_pressed=False, pinch_released=False)

            if control_state.x is not None:
                self.paddle.move_to_normalized(control_state.x)

            # Keyboard fallback remains active for testing without a camera.
            keys = pygame.key.get_pressed()
            direction = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                direction -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                direction += 1.0
            if direction != 0.0:
                self.paddle.move_with_keyboard(direction, dt)

            catch_hold = control_state.pinch or keys[pygame.K_SPACE] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            # While stuck, keep the ball attached to the paddle and watch for launch gestures.
            launch_requested = control_state.pinch_pressed or keys[pygame.K_SPACE]
            if self.ball_stuck:
                self.ball.position = pygame.Vector2(self.paddle.rect.centerx, self.paddle.rect.top - BALL_RADIUS - 2)
                if launch_requested:
                    self.ball_stuck = False
                    self.ball_caught = False
                    # Launch at a 45-degree angle while preserving configured speed.
                    launch_direction = pygame.Vector2(1, -1).normalize()
                    self.ball.velocity = launch_direction * BALL_SPEED
                    self._play_sound("paddle")
                self.ball_trail.clear()
            elif self.ball_caught:
                # Keep the ball locked to the paddle at the capture offset.
                self.ball.position = pygame.Vector2(
                    max(self.paddle.rect.left + BALL_RADIUS, min(self.paddle.rect.right - BALL_RADIUS, self.paddle.rect.centerx + self.caught_offset)),
                    self.paddle.rect.top - BALL_RADIUS - 2,
                )
                if not catch_hold or control_state.pinch_released:
                    # Launch with angle based on where the ball was caught.
                    overlap_center = self.ball.position.x - self.paddle.rect.centerx
                    angle = max(-math.pi / 3, min(math.pi / 3, overlap_center / (PADDLE_WIDTH / 2) * (math.pi / 3)))
                    speed = BALL_SPEED
                    self.ball.velocity = pygame.Vector2(speed * math.sin(angle), -abs(speed * math.cos(angle)))
                    self.ball_caught = False
                    self.ball_trail.clear()
                    self._play_sound("paddle")
            else:
                self.ball.update(dt)

            if not self.ball_stuck:
                if not self.ball_caught:
                    self._handle_wall_collisions()
                    self._handle_paddle_collision(catch_hold)
                    self._handle_brick_collisions()
                    self.ball_trail.append(pygame.Vector2(self.ball.position))

            if self.lives <= 0 or not self.bricks:
                running = False

            self._draw_entities(dt)
            pygame.display.flip()

        self._finalize_scores()
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

    def _show_start_screen(self, control_source: Optional[ControlSource]) -> bool:
        """Wait for pinch/space to start and show score history."""

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c and control_source:
                    try:
                        control_source.toggle_calibration()
                    except Exception:
                        pass

            control_state = ControlState(x=None, pinch=False, pinch_pressed=False, pinch_released=False)
            if control_source:
                try:
                    control_state = control_source.read()
                except Exception:
                    control_state = ControlState(x=None, pinch=False, pinch_pressed=False, pinch_released=False)
            keys = pygame.key.get_pressed()
            if control_state.x is not None:
                self.paddle.move_to_normalized(control_state.x)
            direction = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                direction -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                direction += 1.0
            if direction != 0.0:
                self.paddle.move_with_keyboard(direction, 1 / 60)

            start_requested = control_state.pinch_pressed or keys[pygame.K_SPACE]
            if start_requested:
                waiting = False

            self.screen.blit(self.background, (0, 0))
            title = self.big_font.render("Pinch or Space to Serve!", True, (255, 255, 255))
            best = self.font.render(f"Best Score: {self.best_score}", True, (210, 210, 210))
            last = self.font.render(f"Last Score: {self.last_score}", True, (180, 180, 180))
            hint = self.font.render("Press C to calibrate (camera mode)", True, (180, 220, 255))
            tip = self.font.render("Tip: Pinch catches/launches the ball.", True, (200, 210, 255))
            self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40)))
            self.screen.blit(best, best.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 8)))
            self.screen.blit(last, last.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40)))
            self.screen.blit(hint, hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 90)))
            self.screen.blit(tip, tip.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120)))
            pygame.display.flip()
        return True

    def _finalize_scores(self) -> None:
        """Update persisted score values once per session."""

        self.best_score = max(self.best_score, self.score)
        self.last_score = self.score
        latest = persist_state(best_score=self.best_score, last_score=self.last_score)
        self.persisted_state = latest
