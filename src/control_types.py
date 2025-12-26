"""Typed control interface shared between the game loop and vision module.

This module centralizes the control data model so that both the pygame game
loop and the MediaPipe/OpenCV vision stack can evolve independently while
remaining type-safe. It also provides a tiny exponential moving average helper
to smooth noisy signals such as the horizontal hand position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class ControlState:
    """Represents the current input state for a single frame.

    This intentionally bundles both pose (normalized ``x``) and gesture
    information so the vision layer can grow without forcing the game loop to
    learn MediaPipe-specific details. ``confidence`` is kept optional to avoid
    punishing keyboard-only fallback and tests.

    Attributes:
        x: Normalized horizontal position ``[0, 1]`` for the paddle. ``None``
            indicates that no reliable position was detected this frame.
        pinch: Whether the pinch gesture is currently considered active.
        pinch_pressed: Rising edge for the pinch gesture (``False`` -> ``True``).
        pinch_released: Falling edge for the pinch gesture (``True`` -> ``False``).
        confidence: Optional per-frame tracking confidence. ``None`` when the
            source cannot provide one (keyboard mode, failed detection, etc.).
    """

    x: Optional[float]
    pinch: bool
    pinch_pressed: bool
    pinch_released: bool
    confidence: Optional[float] = None


class ControlSource(Protocol):
    """Interface implemented by control providers (vision, keyboard, etc.)."""

    def read(self) -> ControlState:  # pragma: no cover - protocol definition
        ...

    def close(self) -> None:  # pragma: no cover - protocol definition
        ...

    def toggle_calibration(self) -> None:  # pragma: no cover - optional extension point
        ...

    def reset_calibration(self) -> None:  # pragma: no cover - optional extension point
        ...


@dataclass
class ExponentialSmoother:
    """Reusable exponential moving average for scalar values.

    This helper keeps the last smoothed value so callers can continuously feed
    noisy measurements and receive a stable output. It intentionally accepts
    ``None`` samples to keep the previous value untouched when a measurement is
    missing (e.g., hand temporarily not detected). A small configurable dead
    zone ignores micro-jitter while remaining responsive to real motion.
    """

    alpha: float
    value: Optional[float] = None
    dead_zone: float = 0.0

    def update(self, sample: Optional[float]) -> Optional[float]:
        """Blend ``sample`` into the EMA and return the smoothed value.

        Args:
            sample: New measurement or ``None`` when unavailable.

        Returns:
            The updated smoothed value, or ``None`` if no value has been seen
            yet and the sample was ``None``.
        """

        if sample is None:
            return self.value

        if self.value is None:
            self.value = sample
        else:
            # Ignore tiny changes so hand jitter does not shake the paddle.
            delta = sample - self.value
            if abs(delta) < self.dead_zone:
                return self.value
            self.value = (1 - self.alpha) * self.value + self.alpha * sample
        return self.value
