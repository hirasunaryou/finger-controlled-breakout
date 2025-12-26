"""Camera + MediaPipe hand tracking kept separate from gameplay logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.control_types import ControlSource, ControlState, ExponentialSmoother


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


class VisionControlSource(ControlSource):
    """Encapsulates OpenCV capture and MediaPipe Hands inference."""

    def __init__(
        self,
        camera_index: int = 0,
        smoothing_alpha: float = 0.25,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        mirror: bool = False,
        control_mode: str = "palm",
        pinch_on_threshold: float = 0.17,
        pinch_off_threshold: float = 0.22,
        show_debug_overlay: bool = False,
    ) -> None:
        # Camera configuration is intentionally lightweight for laptop use.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.hands = mp.solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.smoother = ExponentialSmoother(alpha=smoothing_alpha)
        self.pinch_tracker = PinchTracker(on_threshold=pinch_on_threshold, off_threshold=pinch_off_threshold)
        self.mirror = mirror
        self.control_mode = control_mode
        self.show_debug_overlay = show_debug_overlay
        self.last_time = time.time()
        self.last_fps = 0.0

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

    def _overlay_debug(
        self,
        frame: np.ndarray,
        x: Optional[float],
        pinch_state: ControlState,
        detected: bool,
    ) -> None:
        """Draw lightweight overlay for troubleshooting."""

        cv2.putText(
            frame,
            f"Detected: {detected}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"x: {x:.2f}" if x is not None else "x: None",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(frame, f"Pinch: {pinch_state.pinch}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Pressed: {pinch_state.pinch_pressed}", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Released: {pinch_state.pinch_released}", (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.last_fps = 0.9 * self.last_fps + 0.1 * (1.0 / dt)
        self.last_time = now
        cv2.putText(frame, f"FPS: {self.last_fps:.1f}", (12, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

    def read(self) -> ControlState:
        success, frame = self.cap.read()
        if not success:
            return ControlState(x=None, pinch=self.pinch_tracker.active, pinch_pressed=False, pinch_released=False)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        x_value: Optional[float]
        pinch_state = ControlState(x=None, pinch=self.pinch_tracker.active, pinch_pressed=False, pinch_released=False)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            raw_x = (
                self._compute_palm_center_x(landmarks)
                if self.control_mode == "palm"
                else self._compute_index_tip_x(landmarks)
            )
            if self.mirror:
                raw_x = 1.0 - raw_x
            raw_x = max(0.0, min(1.0, raw_x))
            x_value = self.smoother.update(raw_x)

            scale = self._compute_hand_scale(landmarks)
            pinch_distance = self._compute_pinch_distance(landmarks) / scale
            pinch_state = self.pinch_tracker.update(pinch_distance)
        else:
            x_value = self.smoother.update(None)
            pinch_state = self.pinch_tracker.update(None)

        if self.show_debug_overlay:
            debug_frame = frame.copy()
            self._overlay_debug(debug_frame, x_value, pinch_state, results.multi_hand_landmarks is not None)
            cv2.imshow("Control Debug", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return ControlState(x=None, pinch=self.pinch_tracker.active, pinch_pressed=False, pinch_released=False)

        return ControlState(
            x=x_value,
            pinch=pinch_state.pinch,
            pinch_pressed=pinch_state.pinch_pressed,
            pinch_released=pinch_state.pinch_released,
        )

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        if self.show_debug_overlay:
            cv2.destroyAllWindows()
