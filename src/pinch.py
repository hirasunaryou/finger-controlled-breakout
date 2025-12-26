"""Pure pinch-gesture tracker with hysteresis to keep tests lightweight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.control_types import ControlState


@dataclass
class PinchTracker:
    """Tracks pinch gesture state with hysteresis to reduce flicker."""

    on_threshold: float
    off_threshold: float
    active: bool = False

    def update(self, distance: Optional[float]) -> ControlState:
        """Update pinch state given the latest normalized distance."""

        pinch = self.active
        pinch_pressed = False
        pinch_released = False

        if distance is not None:
            if not self.active and distance <= self.on_threshold:
                pinch = True
                pinch_pressed = True
            elif self.active and distance >= self.off_threshold:
                pinch = False
                pinch_released = True

        self.active = pinch
        return ControlState(x=None, pinch=pinch, pinch_pressed=pinch_pressed, pinch_released=pinch_released)
