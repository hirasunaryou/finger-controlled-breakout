"""Camera + MediaPipe hand tracking kept separate from gameplay logic."""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.calibration import Calibration, PersistedState, apply_calibration, load_state, persist_state
from src.control_types import ControlSource, ControlState, ExponentialSmoother
from src.pinch import PinchTracker

# Some Windows Python environments ship with a lightweight "mediapipe" package
# that does not expose ``solutions`` at the top level, so we attempt a second
# import path and keep a helpful error ready for callers.
try:
    from mediapipe import solutions as mp_solutions
except Exception:
    mp_solutions = getattr(mp, "solutions", None)
if mp_solutions is None:
    MP_IMPORT_ERROR = ImportError(
        "mediapipe.solutions could not be imported. Try reinstalling mediapipe "
        "or upgrading to 0.10.14+. (pip install --upgrade mediapipe)"
    )
else:
    MP_IMPORT_ERROR = None


class VisionControlSource(ControlSource):
    """Encapsulates OpenCV capture and MediaPipe Hands inference."""

    def __init__(
        self,
        camera_index: int = 0,
        smoothing_alpha: float = 0.25,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        mirror: bool = False,
        rotate: int = 0,
        flip_x: bool = False,
        flip_y: bool = False,
        control_mode: str = "palm",
        pinch_on_threshold: float = 0.17,
        pinch_off_threshold: float = 0.22,
        show_debug_overlay: bool = False,
        smoothing_deadzone: float = 0.01,
        persisted_state: Optional[PersistedState] = None,
    ) -> None:
        # Camera configuration is intentionally lightweight for laptop use.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if MP_IMPORT_ERROR:
            # Raise a clear, actionable error instead of the cryptic attribute error.
            raise MP_IMPORT_ERROR

        self.hands = mp_solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.smoother = ExponentialSmoother(alpha=smoothing_alpha, dead_zone=smoothing_deadzone)
        self.pinch_tracker = PinchTracker(on_threshold=pinch_on_threshold, off_threshold=pinch_off_threshold)
        self.mirror = mirror
        self.rotate = rotate
        self.flip_camera_x = flip_x
        self.flip_camera_y = flip_y
        self.control_mode = control_mode
        self.show_debug_overlay = show_debug_overlay
        self.last_time = time.time()
        self.last_fps = 0.0
        self.last_x: Optional[float] = None
        self.persisted_state = persisted_state or load_state()
        self.calibration: Optional[Calibration] = self.persisted_state.calibration
        self.calibration_mode = False
        self.calibration_progress: Tuple[Optional[float], Optional[float]] = (None, None)
        self.calibration_error: Optional[str] = None
        # Preparing a dedicated window name keeps calibration discoverable.
        self.window_name = "Control Debug"

    def toggle_calibration(self) -> None:
        """Expose calibration toggling so pygame's 'C' key can trigger it."""

        self.calibration_mode = not self.calibration_mode
        self.calibration_progress = (None, None)
        self.calibration_error = None

    def reset_calibration(self) -> None:
        """Clear calibration both in-memory and on disk for a clean slate."""

        self.calibration = None
        self.calibration_progress = (None, None)
        self.calibration_error = None
        self.persisted_state = persist_state(
            calibration=None,
            best_score=self.persisted_state.best_score,
            last_score=self.persisted_state.last_score,
        )

    def _transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate/flip the camera frame before inference and overlay.

        The transforms happen up front so MediaPipe receives the same view the
        user sees in the debug window, keeping x-values intuitive when the
        webcam feed is upside-down or mirrored by default.
        """

        transformed = frame
        if self.rotate == 90:
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            transformed = cv2.rotate(transformed, cv2.ROTATE_180)
        elif self.rotate == 270:
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.flip_camera_x:
            transformed = cv2.flip(transformed, 1)
        if self.flip_camera_y:
            transformed = cv2.flip(transformed, 0)

        return transformed

    def _compute_hand_scale(self, landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark]) -> float:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        return float(np.linalg.norm([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y])) or 1.0

    def _compute_palm_center_x(self, landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark]) -> float:
        indices = [0, 5, 9, 13, 17]
        xs = [landmarks[i].x for i in indices]
        return float(np.mean(xs))

    def _compute_index_tip_x(self, landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark]) -> float:
        return float(landmarks[8].x)

    def _compute_pinch_distance(self, landmarks: list[mp.framework.formats.landmark_pb2.NormalizedLandmark]) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        return float(np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))

    def _pick_hand(
        self,
        hands: Sequence[list[mp.framework.formats.landmark_pb2.NormalizedLandmark]],
        handedness: Sequence[mp.framework.formats.classification_pb2.ClassificationList],
    ) -> tuple[list[mp.framework.formats.landmark_pb2.NormalizedLandmark], float]:
        """Pick the most confident or nearest-to-last hand when multiple are visible."""

        best_idx = 0
        best_score = -1.0
        for idx, (landmarks, classification) in enumerate(zip(hands, handedness)):
            confidence = classification.classification[0].score if classification.classification else 0.0
            raw_x = (
                self._compute_palm_center_x(landmarks)
                if self.control_mode == "palm"
                else self._compute_index_tip_x(landmarks)
            )
            if self.mirror:
                raw_x = 1.0 - raw_x
            # Prefer confident detections; tie-break with proximity to previous x.
            distance_bonus = 0.0
            if self.last_x is not None:
                distance_bonus = max(0.0, 1.0 - abs(raw_x - self.last_x))
            score = confidence + 0.1 * distance_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen = hands[best_idx]
        confidence = handedness[best_idx].classification[0].score if handedness[best_idx].classification else 0.0
        return chosen, confidence

    def _update_calibration(self, pinch_state: ControlState, raw_x: Optional[float]) -> None:
        """Handle the guided two-step calibration when the user presses 'C'.

        The flow is intentionally explicit:
        1) Wait for a pinch on the left edge to capture ``x_left``.
        2) Wait for a pinch on the right edge to capture ``x_right``.
        3) Validate that the span is usable before persisting to disk.
        """

        if not self.calibration_mode:
            return
        if raw_x is None:
            return

        left, right = self.calibration_progress
        if left is None and pinch_state.pinch_pressed:
            left = raw_x
            self.calibration_error = None
        elif left is not None and right is None and pinch_state.pinch_pressed:
            right = raw_x

        self.calibration_progress = (left, right)

        if left is not None and right is not None:
            # Require a small span to avoid divide-by-zero and confusing mappings.
            if right <= left + 0.02:
                self.calibration_error = "Right must be to the right of left; try again."
                self.calibration_progress = (None, None)
                return

            self.calibration = Calibration(x_left=left, x_right=right)
            self.persisted_state = persist_state(
                calibration=self.calibration,
                best_score=self.persisted_state.best_score,
                last_score=self.persisted_state.last_score,
            )
            self.calibration_mode = False
            self.calibration_progress = (None, None)
            self.calibration_error = None

    def _draw_calibration_overlay(self, frame: np.ndarray) -> None:
        """Show step-by-step prompts over the camera preview."""

        if not self.calibration_mode:
            return

        prompt = (
            "Step 1: Move hand to LEFT edge and pinch"
            if self.calibration_progress[0] is None
            else "Step 2: Move hand to RIGHT edge and pinch"
        )
        cv2.putText(frame, "CALIBRATION MODE (C to exit)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, prompt, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "Aim for a wide span; press R to clear saved calibration.",
            (18, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 220, 255),
            2,
        )

    def _overlay_debug(
        self,
        frame: np.ndarray,
        raw_x: Optional[float],
        calibrated_x: Optional[float],
        smoothed_x: Optional[float],
        pinch_state: ControlState,
        detected: bool,
        landmarks: Optional[list[mp.framework.formats.landmark_pb2.NormalizedLandmark]],
    ) -> None:
        """Draw rich overlay so players can self-debug without reading code."""

        height, width, _ = frame.shape

        # 1) Stick picture: landmarks and bones to reassure that MediaPipe sees the hand.
        if landmarks is not None:
            mp_solutions.drawing_utils.draw_landmarks(
                frame,
                landmarks,
                mp_solutions.hands.HAND_CONNECTIONS,
                mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp_solutions.drawing_styles.get_default_hand_connections_style(),
            )
        else:
            cv2.putText(
                frame,
                "NO HAND",
                (max(8, width // 2 - 70), height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )

        # 2) Vertical guides for raw and calibrated x positions.
        if raw_x is not None:
            cv2.line(
                frame,
                (int(raw_x * width), 0),
                (int(raw_x * width), height),
                (255, 255, 0),
                2,
            )
        if calibrated_x is not None:
            cv2.line(
                frame,
                (int(calibrated_x * width), 0),
                (int(calibrated_x * width), height),
                (0, 255, 255),
                2,
            )
        if smoothed_x is not None:
            cv2.line(
                frame,
                (int(smoothed_x * width), 0),
                (int(smoothed_x * width), height),
                (0, 200, 120),
                1,
            )

        # 3) Textual debug heads-up display.
        lines = []
        step_label = "Idle"
        if self.calibration_mode:
            step_label = "Waiting LEFT pinch" if self.calibration_progress[0] is None else "Waiting RIGHT pinch"
        lines.append(f"Calibration: {'ON' if self.calibration_mode else 'OFF'} (C to toggle, R to reset)")
        lines.append(f"Step: {step_label}")
        lines.append(f"Raw x: {raw_x:.3f}" if raw_x is not None else "Raw x: None")
        lines.append(f"Calibrated x: {calibrated_x:.3f}" if calibrated_x is not None else "Calibrated x: None")
        lines.append(f"Smoothed x: {smoothed_x:.3f}" if smoothed_x is not None else "Smoothed x: None")
        lines.append(f"Pinch: {pinch_state.pinch}")
        lines.append(f"Pressed: {pinch_state.pinch_pressed} / Released: {pinch_state.pinch_released}")
        lines.append(
            f"Confidence: {pinch_state.confidence:.2f}" if pinch_state.confidence is not None else "Confidence: n/a"
        )

        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.last_fps = 0.9 * self.last_fps + 0.1 * (1.0 / dt)
        self.last_time = now
        lines.append(f"FPS: {self.last_fps:.1f}")

        if self.calibration_error:
            lines.append(f"Error: {self.calibration_error}")

        for idx, text in enumerate(lines):
            y = 24 + idx * 20
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def read(self) -> ControlState:
        success, frame = self.cap.read()
        if not success:
            # Keep pinch state flowing so the game loop remains consistent even
            # if the camera momentarily hiccups.
            return ControlState(
                x=None,
                pinch=self.pinch_tracker.active,
                pinch_pressed=False,
                pinch_released=False,
                confidence=None,
            )

        frame = self._transform_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        x_value: Optional[float]
        raw_x_for_calibration: Optional[float] = None
        calibrated_for_overlay: Optional[float] = None
        pinch_state = ControlState(
            x=None,
            pinch=self.pinch_tracker.active,
            pinch_pressed=False,
            pinch_released=False,
            confidence=None,
        )
        chosen_landmarks: Optional[list[mp.framework.formats.landmark_pb2.NormalizedLandmark]] = None

        if results.multi_hand_landmarks and results.multi_handedness:
            chosen, confidence = self._pick_hand(results.multi_hand_landmarks, results.multi_handedness)
            raw_x = (
                self._compute_palm_center_x(chosen) if self.control_mode == "palm" else self._compute_index_tip_x(chosen)
            )
            if self.mirror:
                raw_x = 1.0 - raw_x
            raw_x = max(0.0, min(1.0, raw_x))

            raw_x_for_calibration = raw_x
            calibrated_for_overlay = apply_calibration(raw_x, self.calibration)
            x_value = self.smoother.update(calibrated_for_overlay)

            scale = self._compute_hand_scale(chosen)
            pinch_distance = self._compute_pinch_distance(chosen) / scale
            pinch_state = self.pinch_tracker.update(pinch_distance)
            pinch_state.confidence = confidence
            chosen_landmarks = chosen

            # Keep the last usable x for multi-hand handoff scoring.
            if raw_x_for_calibration is not None:
                self.last_x = raw_x_for_calibration
        else:
            x_value = self.smoother.update(None)
            pinch_state = self.pinch_tracker.update(None)

        # Update calibration state based on pinch edges and raw x.
        self._update_calibration(pinch_state, raw_x_for_calibration)

        if self.show_debug_overlay or self.calibration_mode:
            debug_frame = frame.copy()
            self._overlay_debug(
                debug_frame,
                raw_x_for_calibration,
                calibrated_for_overlay,
                x_value,
                pinch_state,
                results.multi_hand_landmarks is not None,
                chosen_landmarks,
            )
            self._draw_calibration_overlay(debug_frame)
            cv2.imshow(self.window_name, debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return ControlState(x=None, pinch=self.pinch_tracker.active, pinch_pressed=False, pinch_released=False)
            if key == ord("c"):
                # Toggle calibration; reset progress each time to keep it simple.
                self.toggle_calibration()
            if key == ord("r"):
                self.reset_calibration()

        return ControlState(
            x=x_value,
            pinch=pinch_state.pinch,
            pinch_pressed=pinch_state.pinch_pressed,
            pinch_released=pinch_state.pinch_released,
            confidence=pinch_state.confidence,
        )

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        if self.show_debug_overlay or self.calibration_mode:
            cv2.destroyAllWindows()
