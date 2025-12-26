# Finger-Controlled Breakout

A lightweight Breakout clone that lets you move the paddle with your palm
center (default) or index fingertip using your laptop camera. The pygame-based
game loop is completely separated from the MediaPipe/OpenCV vision pipeline to
keep the design clean and testable.

## Features
- Classic Breakout mechanics (paddle, ball, bricks, score, and lives).
- Palm-center hand tracking for stable horizontal control (switchable to index).
- Pinch gesture (thumb–index) to launch/relaunch the ball; Space still works.
- Exponential moving average smoothing to reduce jitter in the control signal.
- Keyboard controls remain available as a fallback for development or `--no-camera`.
- Tasteful visuals: ball trail and brief hit flash on paddle/brick contact.

## Quick start (Windows 11 / Python 3.12)
1. Ensure Python 3.12 and pip are installed.
2. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Run the game (uses the default camera in palm mode):
   ```powershell
   python -m src.main
   ```

## Controls
- Hand: Move your palm left/right; pinch (thumb-index) to launch the ball.
- Keyboard fallback: Arrow keys or `A/D` move the paddle; Space launches; `Esc` quits.

## CLI options (examples)
- Force keyboard-only mode:
  ```powershell
  python -m src.main --no-camera
  ```
- Mirror input for some camera setups:
  ```powershell
  python -m src.main --mirror
  ```
- Switch to index-tip control and tweak pinch hysteresis:
  ```powershell
  python -m src.main --control-mode index --pinch-on-threshold 0.15 --pinch-off-threshold 0.2
  ```
- Show vision debug overlay (x value, pinch state, FPS):
  ```powershell
  python -m src.main --show-debug-overlay
  ```

## Project structure
- `src/control_types.py` — shared control state dataclass, smoothing helper, and control interface.
- `src/game.py` — pygame game loop and rendering (ball trail + hit flash).
- `src/vision.py` — OpenCV + MediaPipe palm/index tracking and pinch detection.
- `src/main.py` — Binds the chosen control source to the game.
- `requirements.txt` — Python dependencies.

## Tips
- Use `--no-camera` to develop or play without the camera.
- Adjust smoothing to taste with `--smoothing-alpha 0.15` (lower = smoother).
- Good lighting and keeping your hand within the frame improve detection stability.
- If MediaPipe wheels give trouble on Windows/Python 3.12, pin versions from `requirements.txt` and install via `python -m pip install --upgrade pip` first.
