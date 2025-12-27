"""Utility routines for drawing readable, wrapped HUD text on OpenCV frames.

The debug/calibration overlay uses these helpers to keep text legible even
when long strings are present. All functions accept a BGR frame (NumPy array)
and intentionally avoid pygame so they can run inside the vision thread.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2

# Default styling tuned for a 640x480 preview; scaled in callers via font_scale.
DEFAULT_COLOR = (255, 255, 255)
DEFAULT_THICKNESS = 1


def draw_panel(surface, x: int, y: int, w: int, h: int, alpha: int = 140) -> None:
    """Overlay a semi-transparent rectangular panel onto ``surface``.

    Args:
        surface: BGR frame to draw on.
        x, y: Top-left corner of the panel.
        w, h: Panel width and height.
        alpha: Opacity ``[0, 255]``; higher = more opaque. Defaults to a gentle 140.

    The implementation copies the frame once and reuses ``cv2.addWeighted`` to
    keep blending costs low inside the vision thread.
    """

    alpha = max(0, min(255, alpha))
    overlay = surface.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha / 255.0, surface, 1 - alpha / 255.0, 0, surface)


def _wrap_line(text: str, wrap_width: int, font_scale: float, thickness: int) -> List[str]:
    """Split ``text`` into multiple lines that fit within ``wrap_width`` pixels."""

    words = text.split(" ")
    wrapped: List[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        size, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if size[0] <= wrap_width or not current:
            current = candidate
        else:
            wrapped.append(current)
            current = word

    if current:
        wrapped.append(current)
    return wrapped


def draw_lines(
    surface,
    lines: Iterable[str],
    x: int,
    y: int,
    line_height: int,
    wrap_width: int,
    *,
    font_scale: float = 0.65,
    color: Tuple[int, int, int] = DEFAULT_COLOR,
    thickness: int = DEFAULT_THICKNESS,
    max_lines: int | None = None,
) -> None:
    """Render wrapped lines with consistent vertical spacing.

    Args:
        surface: BGR frame to draw on.
        lines: Iterable of strings to render.
        x, y: Starting top-left position for the first line.
        line_height: Pixel offset applied to each subsequent rendered line.
        wrap_width: Maximum width in pixels before wrapping words.
        font_scale: OpenCV font scale multiplier (defaults to 0.65).
        color: BGR text color.
        thickness: Stroke thickness for ``cv2.putText``.
        max_lines: Optional hard limit to keep HUDs compact (keeps the first N).

    Long words that still exceed the wrap width will remain unbroken; callers
    should choose a wrap width that is reasonable for the font size in use.
    """

    wrapped_lines: List[str] = []
    for line in lines:
        wrapped_lines.extend(_wrap_line(line, wrap_width, font_scale, thickness))

    if max_lines is not None:
        wrapped_lines = wrapped_lines[:max_lines]

    for idx, text in enumerate(wrapped_lines):
        y_pos = y + idx * line_height
        cv2.putText(surface, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
