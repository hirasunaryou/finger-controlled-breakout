"""Camera + MediaPipe finger tracking kept separate from gameplay logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp


@dataclass
class ExponentialSmoother:
    """Simple exponential moving average to reduce fingertip jitter.

    Attributes:
        alpha: Weight for the latest sample. Smaller values = smoother output.
        value: Last smoothed value (``None`` until a sample is observed).
    """

    alpha: float
    value: Optional[float] = None

    def update(self, sample: float) -> float:
        """Blend the new sample with previous state and return the smoothed value."""

        if self.value is None:
            self.value = sample
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * sample
        return self.value


class FingerTracker:
    """Encapsulates OpenCV capture and MediaPipe Hands inference."""

    def __init__(
        self,
        camera_index: int = 0,
        smoothing_alpha: float = 0.25,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
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

    def read_normalized_x(self) -> Optional[float]:
        """Return the smoothed normalized x-position for the index fingertip.

        The coordinate is already normalized by MediaPipe to ``[0.0, 1.0]``.
        Returning ``None`` allows the caller to keep the paddle still and rely
        on keyboard controls until the finger is detected again.
        """

        success, frame = self.cap.read()
        if not success:
            return None

        # MediaPipe expects RGB input; OpenCV captures in BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        # Take the first detected hand and use the index fingertip (landmark 8).
        fingertip = results.multi_hand_landmarks[0].landmark[8]
        normalized_x = max(0.0, min(1.0, fingertip.x))
        return self.smoother.update(normalized_x)

    def release(self) -> None:
        """Release camera and MediaPipe resources gracefully."""

        if self.cap.isOpened():
            self.cap.release()
        self.hands.close()

