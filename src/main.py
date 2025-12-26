"""Entry point wiring together the game and the vision module."""

from __future__ import annotations

import argparse
from typing import Optional

import pygame

from src.control_types import ControlSource, ControlState
from src.calibration import load_state, persist_state
from src.game import Game
from src.vision import VisionControlSource


class KeyboardControlSource(ControlSource):
    """Keyboard-only control for debugging without a camera.

    This source focuses on pinch emulation so that the main game loop continues
    to receive consistent `ControlState` objects even in `--no-camera` mode.
    """

    def __init__(self) -> None:
        self.previous_space = False

    def read(self) -> ControlState:
        keys = pygame.key.get_pressed()
        space = bool(keys[pygame.K_SPACE])
        pinch_pressed = not self.previous_space and space
        pinch_released = self.previous_space and not space
        self.previous_space = space
        return ControlState(x=None, pinch=space, pinch_pressed=pinch_pressed, pinch_released=pinch_released)

    def close(self) -> None:
        # Nothing to clean up for keyboard-only mode.
        return

    def toggle_calibration(self) -> None:
        # Keyboard-only mode has nothing to calibrate but keeps the interface uniform.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Breakout with your hand or keyboard.")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera control and rely on keyboard input.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor for hand x-position (0-1, higher = snappier).",
    )
    parser.add_argument(
        "--smoothing-deadzone",
        type=float,
        default=0.01,
        help="Dead zone applied before smoothing to reduce micro jitter.",
    )
    parser.add_argument("--control-mode", choices=["palm", "index"], default="palm", help="Use palm center or index tip.")
    parser.add_argument("--mirror", action="store_true", help="Mirror the horizontal input (useful for some cameras).")
    parser.add_argument(
        "--pinch-on-threshold",
        type=float,
        default=0.17,
        help="Normalized pinch distance below which pinch is considered active.",
    )
    parser.add_argument(
        "--pinch-off-threshold",
        type=float,
        default=0.22,
        help="Normalized pinch distance above which pinch is released.",
    )
    parser.add_argument(
        "--show-debug-overlay",
        action="store_true",
        help="Show debug info (x, pinch state, FPS) on the camera feed.",
    )
    args = parser.parse_args()

    persisted_state = load_state()
    game = Game(persisted_state=persisted_state)
    control_source: Optional[ControlSource]
    if args.no_camera:
        control_source = KeyboardControlSource()
    else:
        control_source = VisionControlSource(
            smoothing_alpha=args.smoothing_alpha,
            mirror=args.mirror,
            control_mode=args.control_mode,
            pinch_on_threshold=args.pinch_on_threshold,
            pinch_off_threshold=args.pinch_off_threshold,
            show_debug_overlay=args.show_debug_overlay,
            smoothing_deadzone=args.smoothing_deadzone,
            persisted_state=persisted_state,
        )

    try:
        game.run(control_source)
    finally:
        if control_source:
            control_source.close()
        # Persist best score at shutdown in case the game updated it.
        persist_state(best_score=game.best_score)


if __name__ == "__main__":
    main()
