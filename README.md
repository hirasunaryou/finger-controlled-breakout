# Finger-Controlled Breakout

A lightweight Breakout clone that lets you move the paddle with your index
finger using your laptop camera. The pygame-based game loop is completely
separated from the MediaPipe/OpenCV vision pipeline to keep the design clean
and testable.

## Features
- Classic Breakout mechanics (paddle, ball, bricks, score, and lives).
- MediaPipe Hands detects the index fingertip and maps it to paddle position.
- Exponential moving average smoothing to reduce jitter in the control signal.
- Keyboard controls remain available as a fallback for development.

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
4. Run the game (uses the default camera):
   ```powershell
   python -m src.main
   ```

## Controls
- Finger: Move your index finger left/right; the paddle follows.
- Keyboard fallback: Arrow keys or `A/D` move the paddle. `Esc` quits.

## Project structure
- `src/game.py` — pygame game loop and rendering.
- `src/vision.py` — OpenCV + MediaPipe fingertip tracking (returns normalized x).
- `src/main.py` — Binds the camera input to the game.
- `requirements.txt` — Python dependencies.

## Tips
- Use `--no-camera` to develop or play without the camera: `python -m src.main --no-camera`.
- Adjust smoothing to taste with `--smoothing-alpha 0.15` (lower = smoother).
- Good lighting and keeping your hand within the frame improve detection stability.
