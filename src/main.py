"""Entry point wiring together the game and the vision module."""

from __future__ import annotations

import argparse
from typing import Callable, Optional

from src.game import Game
from src.vision import FingerTracker


def build_control_source(use_camera: bool, smoothing_alpha: float) -> Optional[Callable[[], Optional[float]]]:
    """Create a callable that returns the normalized x-position each frame."""

    if not use_camera:
        return None

    tracker = FingerTracker(smoothing_alpha=smoothing_alpha)

    def control() -> Optional[float]:
        return tracker.read_normalized_x()

    # Attach release hook so the camera closes once the game ends.
    control.release_tracker = tracker.release  # type: ignore[attr-defined]
    return control


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Breakout with your index finger.")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera control and rely on keyboard input.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor for fingertip x-position (0-1, higher = snappier).",
    )
    args = parser.parse_args()

    control_source = build_control_source(not args.no_camera, args.smoothing_alpha)
    game = Game()
    try:
        game.run(control_source)
    finally:
        if control_source and hasattr(control_source, "release_tracker"):
            control_source.release_tracker()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

