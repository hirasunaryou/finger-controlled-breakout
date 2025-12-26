import math

from src.calibration import Calibration, apply_calibration
from src.control_types import ExponentialSmoother
from src.pinch import PinchTracker


def test_exponential_smoother_dead_zone() -> None:
    smoother = ExponentialSmoother(alpha=0.5, dead_zone=0.05)
    assert smoother.update(0.0) == 0.0
    # Within dead zone: value should stay pinned.
    assert smoother.update(0.02) == 0.0
    # Outside dead zone: blend toward the new value.
    assert math.isclose(smoother.update(0.2) or 0.0, 0.1, rel_tol=1e-6)


def test_pinch_hysteresis_edges() -> None:
    tracker = PinchTracker(on_threshold=0.1, off_threshold=0.2)
    state = tracker.update(0.15)
    assert state.pinch is False
    state = tracker.update(0.09)
    assert state.pinch is True and state.pinch_pressed is True
    state = tracker.update(0.18)
    assert state.pinch is True  # still active inside hysteresis band
    state = tracker.update(0.25)
    assert state.pinch is False and state.pinch_released is True


def test_calibration_mapping_and_clamp() -> None:
    calib = Calibration(x_left=0.2, x_right=0.8)
    assert apply_calibration(0.2, calib) == 0.0
    assert math.isclose(apply_calibration(0.5, calib), 0.5, rel_tol=1e-6)
    assert apply_calibration(1.0, calib) == 1.0
    # When no calibration is present, values are simply clamped.
    assert apply_calibration(-0.2, None) == 0.0
