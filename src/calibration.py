"""Calibration helpers and lightweight persistence for shared game state.

This module centralizes two small pieces of data the game and the camera
pipeline need to agree on:

* Horizontal calibration (``x_left`` / ``x_right``) so raw camera coordinates
  can be mapped to ``[0, 1]`` in a user-specific way.
* The best score so the start screen can celebrate progress across sessions.

The functions are intentionally tiny and pure to keep them easy to unit test
without a camera or pygame running.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple


STATE_PATH = Path.home() / ".finger_breakout.json"
# Sentinel to allow callers to explicitly clear calibration without
# overloading ``None`` (which legitimately represents "no calibration").
_CALIBRATION_UNSET = object()


@dataclass
class Calibration:
    """User-specific calibration describing how far left/right the hand can go."""

    x_left: float
    x_right: float

    def clamp_and_map(self, raw_x: float) -> float:
        """Convert a raw normalized x value into calibrated ``[0, 1]``.

        The mapping applies the standard linear transform and clamps the
        output. A small epsilon guards against divide-by-zero if the two
        calibration points are extremely close.
        """

        span = max(1e-4, self.x_right - self.x_left)
        normalized = (raw_x - self.x_left) / span
        return max(0.0, min(1.0, normalized))


@dataclass
class PersistedState:
    """Combined on-disk settings shared by the vision and game layers."""

    calibration: Optional[Calibration] = None
    best_score: int = 0
    last_score: int = 0


def _decode_state(data: Dict[str, Any]) -> PersistedState:
    calibration_data = data.get("calibration") or {}
    calibration = None
    if "x_left" in calibration_data and "x_right" in calibration_data:
        calibration = Calibration(float(calibration_data["x_left"]), float(calibration_data["x_right"]))
    best_score = int(data.get("best_score", 0))
    last_score = int(data.get("last_score", 0))
    return PersistedState(calibration=calibration, best_score=best_score, last_score=last_score)


def _encode_state(state: PersistedState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"best_score": int(state.best_score), "last_score": int(state.last_score)}
    if state.calibration:
        payload["calibration"] = {"x_left": state.calibration.x_left, "x_right": state.calibration.x_right}
    return payload


def load_state(path: Path = STATE_PATH) -> PersistedState:
    """Load persisted values from disk; missing files fall back to defaults."""

    if not path.exists():
        return PersistedState()

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return PersistedState()

    if not isinstance(data, dict):
        return PersistedState()

    return _decode_state(data)


def persist_state(
    *,
    calibration: Optional[Calibration] | object = _CALIBRATION_UNSET,
    best_score: Optional[int] = None,
    last_score: Optional[int] = None,
    path: Path = STATE_PATH,
) -> PersistedState:
    """Merge incoming values with any existing file and write it back to disk.

    ``calibration`` accepts ``None`` to intentionally clear saved calibration, while
    the private ``_CALIBRATION_UNSET`` sentinel means "leave calibration as-is".
    This keeps the API flexible for the debug overlay shortcuts without breaking
    existing callers that only care about score persistence.
    """

    current = load_state(path)
    if calibration is not _CALIBRATION_UNSET:
        current.calibration = calibration  # may be ``None`` to clear the file
    if best_score is not None:
        current.best_score = max(current.best_score, int(best_score))
    if last_score is not None:
        current.last_score = int(last_score)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_encode_state(current), indent=2))
    return current


def apply_calibration(raw_x: float, calibration: Optional[Calibration]) -> float:
    """Helper for tests and callers that may not have calibration yet."""

    if calibration is None:
        return max(0.0, min(1.0, raw_x))
    return calibration.clamp_and_map(raw_x)


@dataclass
class AutoCalibrationWindow:
    """Snapshot of the rolling window used for auto-calibration."""

    min_x: float
    max_x: float
    span: float
    guard_active: bool


class AutoCalibrator:
    """Rolling-window calibration that continuously rescales ``raw_x``.

    The class keeps the last N seconds of raw x-values and maps the incoming
    value into ``[0, 1]`` based on the observed min/max. When the window span
    collapses (hand barely moving), the mapper falls back to the raw clamped
    value instead of amplifying noise.
    """

    def __init__(self, window_seconds: float, guard_span: float = 0.05) -> None:
        # ``deque`` keeps popping old samples O(1) as new values arrive.
        self.window_seconds = max(0.0, float(window_seconds))
        self.guard_span = guard_span
        self._samples: Deque[Tuple[float, float]] = deque()
        self.last_window: Optional[AutoCalibrationWindow] = None

    def map_value(self, raw_x: float, *, now: float) -> tuple[float, Optional[AutoCalibrationWindow]]:
        """Return the auto-calibrated x plus the latest window snapshot.

        ``now`` is injected by callers for testability and to avoid extra clock
        reads. When the observed span is too small, the method returns the
        clamped raw value and marks ``guard_active=True`` in the window.
        """

        # Push new sample and evict anything outside the rolling window.
        self._samples.append((now, raw_x))
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        xs = [sample for _, sample in self._samples]
        if not xs:
            return max(0.0, min(1.0, raw_x)), None

        min_x, max_x = min(xs), max(xs)
        span = max_x - min_x
        # When the hand barely moves, avoid stretching noise across the whole range.
        guard_active = span < self.guard_span

        if guard_active:
            mapped = max(0.0, min(1.0, raw_x))
        else:
            mapped = max(0.0, min(1.0, (raw_x - min_x) / span))

        self.last_window = AutoCalibrationWindow(min_x=min_x, max_x=max_x, span=span, guard_active=guard_active)
        return mapped, self.last_window
