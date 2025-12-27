import math
from pathlib import Path

from src.calibration import AutoCalibrator, Calibration, apply_calibration, load_state, persist_state
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


def test_persist_state_can_clear_and_keep_calibration(tmp_path: Path) -> None:
    """Saving should allow clearing calibration explicitly without forcing callers to edit JSON."""

    temp_file = tmp_path / "state.json"
    calib = Calibration(x_left=0.1, x_right=0.9)
    persist_state(calibration=calib, path=temp_file)
    saved = load_state(temp_file)
    assert saved.calibration is not None
    # Clearing calibration should wipe it while preserving scores.
    persist_state(calibration=None, best_score=5, path=temp_file)
    cleared = load_state(temp_file)
    assert cleared.calibration is None
    assert cleared.best_score == 5
    # Omitting calibration should keep the cleared state intact.
    persist_state(best_score=10, path=temp_file)
    kept = load_state(temp_file)
    assert kept.calibration is None
    assert kept.best_score == 10


def test_auto_calibrator_maps_and_guards() -> None:
    calibrator = AutoCalibrator(window_seconds=2.0, guard_span=0.1)

    mapped, window = calibrator.map_value(0.5, now=0.0)
    assert math.isclose(mapped, 0.5, rel_tol=1e-6)
    assert window is not None and window.guard_active is True

    mapped, window = calibrator.map_value(0.2, now=0.5)
    # With a wide enough span, the mapping should normalize to 0.0 at the minimum.
    assert math.isclose(mapped, 0.0, rel_tol=1e-6)
    assert window is not None and window.guard_active is False

    mapped, window = calibrator.map_value(0.4, now=2.4)
    # The earliest sample falls out of the 2s window, so span recomputes and maps back to 1.0 while keeping the newer sample.
    assert math.isclose(mapped, 1.0, rel_tol=1e-6)
    assert window is not None and window.guard_active is False
