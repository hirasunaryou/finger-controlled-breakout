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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


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
