"""Camera + MediaPipe hand tracking kept separate from gameplay logic."""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.calibration import Calibration, PersistedState, apply_calibration, load_state, persist_state
from src.control_types import ControlSource, ControlState, ExponentialSmoother
from src.pinch import PinchTracker
from src.ui_hud import draw_lines, draw_panel

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
        no_debug_window: bool = False,
        smoothing_deadzone: float = 0.01,
        persisted_state: Optional[PersistedState] = None,
        hud_scale: float = 1.0,
        hud_alpha: int = 140,
        camera_width: int = 640,
        camera_height: int = 480,
        inference_every: int = 1,
    ) -> None:
        # Camera configuration is intentionally lightweight for laptop use.
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
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
        self.no_debug_window = no_debug_window
        self.last_time = time.time()
        self.last_fps = 0.0
        self.last_x: Optional[float] = None
        self.persisted_state = persisted_state or load_state()
        self.calibration: Optional[Calibration] = self.persisted_state.calibration
        self.calibration_mode = False
        self.calibration_progress: Tuple[Optional[float], Optional[float]] = (None, None)
        self.calibration_error: Optional[str] = None
        self.hud_scale = max(0.5, float(hud_scale))
        self.hud_alpha = max(0, min(255, int(hud_alpha)))
        self.inference_every = max(1, int(inference_every))
        # Preparing a dedicated window name keeps calibration discoverable.
        self.window_name = "Control Debug"
        # Shared state protected by a lock so the game loop can poll without blocking.
        self._state_lock = threading.Lock()
        self._latest_state = ControlState(
            x=None,
            pinch=False,
            pinch_pressed=False,
            pinch_released=False,
            confidence=None,
        )
        self._state_version = 0
        self._last_read_version = -1
        # Vision loop bookkeeping; landmarks and FPS are cached to allow inference skipping.
        self._last_landmarks: Optional[list[mp.framework.formats.landmark_pb2.NormalizedLandmark]] = None
        self._last_raw_x: Optional[float] = None
        self._last_calibrated_x: Optional[float] = None
        self._last_smoothed_x: Optional[float] = None
        self._last_detected: bool = False
        self._frame_count = 0
        self._stop_event = threading.Event()
        # Spin up the background thread so the game loop stays responsive even on heavy inference frames.
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    def toggle_calibration(self) -> None:
        """Expose calibration toggling so pygame's 'C' key can trigger it."""

        with self._state_lock:
            self.calibration_mode = not self.calibration_mode
            self.calibration_progress = (None, None)
            self.calibration_error = None

    def reset_calibration(self) -> None:
        """Clear calibration both in-memory and on disk for a clean slate."""

        with self._state_lock:
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

        This method lives inside the vision thread so we keep locking very
        tight—only state mutations are inside the critical section.
        """

        with self._state_lock:
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
                # Require a meaningful span so the mapping does not collapse.
                if right <= left + 0.05:
                    self.calibration_error = "Right must be at least 0.05 to the right; restarting."
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

    def _build_calibration_lines(self) -> Tuple[str, list[str]]:
        """Return the main instruction line and supporting status strings."""

        with self._state_lock:
            calibration_mode = self.calibration_mode
            left, right = self.calibration_progress
            calibration_error = self.calibration_error
            calibration = self.calibration

        if calibration_mode:
            if left is None:
                header = "Step 1: Move hand to LEFT edge and PINCH"
            elif right is None:
                header = "Step 2: Move hand to RIGHT edge and PINCH"
            else:
                header = "Processing calibration..."
        else:
            header = "Calibration idle (press C to start)"

        status = []
        if left is not None:
            status.append(f"LEFT captured: {left:.3f} ✅")
        if right is not None:
            status.append(f"RIGHT captured: {right:.3f} ✅")
        if calibration_error:
            status.append(f"Error: {calibration_error}")
        if calibration:
            status.append(f"Saved calibration: left={calibration.x_left:.3f}, right={calibration.x_right:.3f}")
        else:
            status.append("Saved calibration: none (press C to capture)")

        return header, status

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
        header, calibration_lines = self._build_calibration_lines()
        lines = []
        lines.append("Keys: C: toggle calibration | R: reset calibration | Esc: quit")
        lines.extend(calibration_lines)
        lines.append(f"Raw x: {raw_x:.3f}" if raw_x is not None else "Raw x: None")
        lines.append(f"Calibrated x: {calibrated_x:.3f}" if calibrated_x is not None else "Calibrated x: None")
        lines.append(f"Smoothed x: {smoothed_x:.3f}" if smoothed_x is not None else "Smoothed x: None")
        lines.append(f"Pinch: {pinch_state.pinch} (on: {pinch_state.pinch_pressed}, off: {pinch_state.pinch_released})")
        lines.append(
            f"Confidence: {pinch_state.confidence:.2f}" if pinch_state.confidence is not None else "Confidence: n/a"
        )
        lines.append(f"Detection: {'hand found' if detected else 'no hand'}")

        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.last_fps = 0.9 * self.last_fps + 0.1 * (1.0 / dt)
        self.last_time = now
        lines.append(f"Vision FPS: {self.last_fps:.1f}")

        # Render HUD with word-wrapping and padding.
        margin = 12
        panel_width = int(420 * self.hud_scale)
        line_height = int(22 * self.hud_scale)
        wrap_width = panel_width - margin * 2
        header_y = margin + int(8 * self.hud_scale)
        max_hud_lines = 12
        panel_height = margin * 2 + (1 + max_hud_lines) * line_height
        draw_panel(frame, 8, 8, panel_width, panel_height, alpha=self.hud_alpha)
        # Header uses a slightly larger font for clarity during calibration.
        cv2.putText(
            frame,
            header,
            (8 + margin, header_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * self.hud_scale,
            (0, 255, 255),
            2,
        )
        draw_lines(
            frame,
            lines,
            8 + margin,
            header_y + int(18 * self.hud_scale),
            line_height,
            wrap_width,
            font_scale=0.65 * self.hud_scale,
            max_lines=max_hud_lines,
        )

    def _snapshot_state(self) -> ControlState:
        """Return the latest control state without exposing shared references."""

        with self._state_lock:
            state = self._latest_state
            version = getattr(self, "_state_version", 0)
        return ControlState(
            x=state.x,
            pinch=state.pinch,
            pinch_pressed=state.pinch_pressed,
            pinch_released=state.pinch_released,
            confidence=state.confidence,
        ), version

    def _vision_loop(self) -> None:
        """Capture frames + run inference in the background to avoid game stalls."""

        # Versioning ensures pinch edge events are only delivered once per update.
        self._state_version = 0
        self._last_read_version = -1

        while not self._stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = self._transform_frame(frame)
            # Avoid copying frames unless we actually need the HUD.
            show_window = (self.show_debug_overlay or self.calibration_mode) and not self.no_debug_window
            debug_frame = frame.copy() if show_window else None

            run_inference = (self._frame_count % self.inference_every) == 0
            results = None
            if run_inference:
                # Downscale for inference to tame CPU spikes on lightweight laptops.
                resized = cv2.resize(frame, (self.camera_width, self.camera_height))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.hands.process(rgb_frame)
            self._frame_count += 1

            with self._state_lock:
                # Cache read-only values while minimizing lock duration.
                current_calibration = self.calibration
                current_state = self._latest_state

            raw_x_for_calibration = self._last_raw_x
            calibrated_for_overlay = self._last_calibrated_x
            smoothed_for_overlay = self._last_smoothed_x
            # Start from the last sent state so skipped inference frames stay stable.
            pinch_state = ControlState(
                x=current_state.x,
                pinch=current_state.pinch,
                pinch_pressed=False,
                pinch_released=False,
                confidence=current_state.confidence,
            )
            chosen_landmarks = self._last_landmarks
            detected = self._last_detected

            if run_inference and results and results.multi_hand_landmarks and results.multi_handedness:
                chosen, confidence = self._pick_hand(results.multi_hand_landmarks, results.multi_handedness)
                raw_x = (
                    self._compute_palm_center_x(chosen)
                    if self.control_mode == "palm"
                    else self._compute_index_tip_x(chosen)
                )
                if self.mirror:
                    raw_x = 1.0 - raw_x
                raw_x = max(0.0, min(1.0, raw_x))

                raw_x_for_calibration = raw_x
                calibrated_for_overlay = apply_calibration(raw_x, current_calibration)
                smoothed_for_overlay = self.smoother.update(calibrated_for_overlay)

                scale = self._compute_hand_scale(chosen)
                pinch_distance = self._compute_pinch_distance(chosen) / scale
                pinch_state = self.pinch_tracker.update(pinch_distance)
                pinch_state.confidence = confidence
                chosen_landmarks = chosen
                detected = True

                self.last_x = raw_x_for_calibration
                self._last_landmarks = chosen_landmarks
                self._last_detected = True
            elif run_inference:
                # Preserve prior EMA to avoid jumping when the hand is briefly lost.
                smoothed_for_overlay = self.smoother.update(None)
                pinch_state = self.pinch_tracker.update(None)
                detected = False
                self._last_landmarks = None
                self._last_detected = False

            self._last_raw_x = raw_x_for_calibration
            self._last_calibrated_x = calibrated_for_overlay
            self._last_smoothed_x = smoothed_for_overlay
            self._last_detected = detected

            self._update_calibration(pinch_state, raw_x_for_calibration)

            new_state = ControlState(
                x=smoothed_for_overlay,
                pinch=pinch_state.pinch,
                pinch_pressed=pinch_state.pinch_pressed,
                pinch_released=pinch_state.pinch_released,
                confidence=pinch_state.confidence,
            )
            with self._state_lock:
                self._latest_state = new_state
                self._state_version += 1

            if debug_frame is not None:
                self._overlay_debug(
                    debug_frame,
                    raw_x_for_calibration,
                    calibrated_for_overlay,
                    smoothed_for_overlay,
                    pinch_state,
                    detected,
                    chosen_landmarks,
                )
                cv2.imshow(self.window_name, debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    self.toggle_calibration()
                if key == ord("r"):
                    self.reset_calibration()
                if key == ord("q"):
                    self._stop_event.set()

        # Clean up windows if the loop exits unexpectedly.
        if self.show_debug_overlay or self.calibration_mode:
            cv2.destroyAllWindows()

    def read(self) -> ControlState:
        state, version = self._snapshot_state()
        # Make pinch edges one-shot so the game loop does not receive duplicates.
        if version == self._last_read_version:
            return ControlState(
                x=state.x,
                pinch=state.pinch,
                pinch_pressed=False,
                pinch_released=False,
                confidence=state.confidence,
            )
        self._last_read_version = version
        return state

    def close(self) -> None:
        self._stop_event.set()
        if hasattr(self, "_vision_thread") and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=1.5)
        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
