# Finger-Controlled Breakout

A lightweight Breakout clone that lets you move the paddle with your palm
center (default) or index fingertip using your laptop camera. The pygame-based
game loop is completely separated from the MediaPipe/OpenCV vision pipeline to
keep the design clean and testable.

## Features
- Classic Breakout mechanics (paddle, ball, bricks, score, and lives).
- Palm-center hand tracking is the default, providing stable horizontal control.
- Optional index-fingertip control for a more direct feel.
- Pinch gesture (thumb–index) launches or relaunches the ball; Space still works as a keyboard backup.
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
5. If you prefer keyboard-only play (no camera needed):
   ```powershell
   python -m src.main --no-camera
   ```

## Controls
- With camera: Move your palm left/right (default) or use index fingertip; pinch (thumb-index) to launch/relaunch.
- Keyboard fallback: Arrow keys or `A/D` move the paddle; Space launches; `Esc` quits.
- Mirror input when needed: `python -m src.main --mirror`
- Adjust smoothing if motion feels too snappy or sluggish: `python -m src.main --smoothing-alpha 0.15`
- Show the debug overlay for hand detection and FPS: `python -m src.main --show-debug-overlay`

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

## CLI options (all)

| Option | Default | What it does |
| --- | --- | --- |
| `--no-camera` | `False` | Disable camera control and rely on keyboard input. |
| `--control-mode {palm,index}` | `palm` | Choose palm center (stable) or index fingertip (direct) tracking. |
| `--smoothing-alpha FLOAT` | `0.25` | EMA smoothing factor for hand x-position (`0-1`, higher = snappier). |
| `--mirror` | `False` | Mirror horizontal input, useful if the paddle moves opposite your hand. |
| `--pinch-on-threshold FLOAT` | `0.17` | Normalized thumb–index distance below which pinch becomes active. |
| `--pinch-off-threshold FLOAT` | `0.22` | Normalized thumb–index distance above which pinch releases. |
| `--show-debug-overlay` | `False` | Show a debug window with x-value, pinch state, and FPS. |

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

## Troubleshooting
- Ensure your hand is fully in frame with even lighting; partial hands reduce tracking stability.
- On Windows/Python 3.12, if MediaPipe or OpenCV fails to install or load, pin to the known versions listed in `requirements.txt`, for example: `python -m pip install --upgrade pip` then `python -m pip install \"mediapipe==0.10.14\" \"opencv-python==4.9.0.80\"`.
- If motion feels jittery, try lowering `--smoothing-alpha`; if it lags, raise it or temporarily switch to keyboard with `--no-camera`.

## 簡単なまとめ (Japanese)
- 手のひら中心がデフォルトで安定操作、指先モードも選択可能。つまむ動作でボールを開始/再開でき、キーボードでも遊べます。
- `python -m src.main` でカメラ操作、`--no-camera` でキーボードのみ、`--mirror` や `--smoothing-alpha` で調整できます。
- Windows / Python 3.12 で MediaPipe や OpenCV がうまく入らない場合は、上記の例のようにバージョンを固定して再インストールしてください。
