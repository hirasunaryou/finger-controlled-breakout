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
- Pinch + paddle contact can now *catch* the ball; release to angle your shot. Holding `Space`/`Shift` does the same.
- Exponential moving average smoothing to reduce jitter in the control signal.
- Keyboard controls remain available as a fallback for development or `--no-camera`.
- Tasteful visuals: ball trail and brief hit flash on paddle/brick contact.
- Start screen shows last/best score, with local best-score persistence.
- Zero-calibration default: uses raw normalized x directly so you can play instantly.
- Optional auto or manual calibration if you want the paddle to match your personal reach.

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

## Calibration options
- **Default (no calibration):** The paddle uses the raw normalized x position immediately. If you hit the wall before your hand reaches the frame edge, first try moving the camera or hand distance.
- **Auto-calibration (optional):** Enable with `--auto-calib-seconds 5.0` to keep a rolling window of recent raw x values, map them into `[0, 1]`, and clamp when the observed range is too small. Set `0` to disable.
- **Manual calibration (secondary):** Press `C` in the debug window, pinch once on your comfortable left edge and once on your right edge. Captured values show with ✅; the HUD also lists any errors (e.g., span too small) and saved values. Press `C` again to exit or `R` to clear.

### Stable Windows install (known-good pins)
If you hit dependency conflicts (for example, OpenCV requesting `numpy>=2` while MediaPipe wants `<2`), use the pinned Windows requirements:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -r requirements-windows.txt
```
This set aligns MediaPipe 0.10.21, NumPy 1.26.4, and OpenCV 4.11 for a smoother install.

## Controls
- With camera: Move your palm left/right (default) or use index fingertip; pinch (thumb-index) to launch/relaunch. Hold pinch while catching the ball with the paddle to stick it, then release to fire.
- Keyboard fallback: Arrow keys or `A/D` move the paddle; Space launches; holding `Space` or `Shift` catches; `Esc` quits.
- Calibration shortcuts: Press `C` to toggle calibration mode in the debug window, and `R` to clear saved calibration.
- Mirror input when needed: `python -m src.main --mirror`
- Adjust smoothing if motion feels too snappy or sluggish: `python -m src.main --smoothing-alpha 0.15 --smoothing-deadzone 0.01`
- Show the debug overlay (stick picture, x values, pinch state, FPS): `python -m src.main --show-debug-overlay` (use `--no-debug-window` to disable the preview entirely).
- Calibrate camera reach: by default no calibration is needed. For live auto calibration run with `--auto-calib-seconds 5.0`; for manual capture press `C` and pinch at your comfortable left/right edges (values save to `~/.finger_breakout.json` and errors show in the HUD).
- Rotate/flip tips: `--rotate 180` when your webcam is upside down, add `--flip-x` or `--flip-y` if the mirrored view feels wrong even after `--mirror`.
- Performance tips: lower camera resolution with `--camera-width 640 --camera-height 480`, skip some inference frames via `--inference-every 2`, or hide the preview window with `--no-debug-window`.

## CLI options (examples)
- Force keyboard-only mode:
  ```powershell
  python -m src.main --no-camera
  ```
- Mirror input for some camera setups:
  ```powershell
  python -m src.main --mirror
  ```
- Rotate or flip the camera preview if it appears upside down:
  ```powershell
  python -m src.main --rotate 180 --flip-y
  ```
- Switch to index-tip control and tweak pinch hysteresis:
  ```powershell
  python -m src.main --control-mode index --pinch-on-threshold 0.15 --pinch-off-threshold 0.2
  ```
- Show vision debug overlay (x value, pinch state, FPS):
  ```powershell
  python -m src.main --show-debug-overlay
  ```
- Enable rolling auto-calibration over 5 seconds:
  ```powershell
  python -m src.main --auto-calib-seconds 5.0
  ```

## CLI options (all)

| Option | Default | What it does |
| --- | --- | --- |
| `--no-camera` | `False` | Disable camera control and rely on keyboard input. |
| `--control-mode {palm,index}` | `palm` | Choose palm center (stable) or index fingertip (direct) tracking. |
| `--smoothing-alpha FLOAT` | `0.25` | EMA smoothing factor for hand x-position (`0-1`, higher = snappier). |
| `--smoothing-deadzone FLOAT` | `0.01` | Ignore tiny x-changes before smoothing to reduce jitter. |
| `--mirror` | `False` | Mirror horizontal input, useful if the paddle moves opposite your hand. |
| `--rotate {0,90,180,270}` | `0` | Rotate the camera preview before MediaPipe inference. |
| `--flip-x` | `False` | Flip the camera horizontally before inference. |
| `--flip-y` | `False` | Flip the camera vertically before inference. |
| `--pinch-on-threshold FLOAT` | `0.17` | Normalized thumb–index distance below which pinch becomes active. |
| `--pinch-off-threshold FLOAT` | `0.22` | Normalized thumb–index distance above which pinch releases. |
| `--show-debug-overlay` | `False` | Show a debug window with stick picture, x values, pinch state, and FPS. |
| `--no-debug-window` | `False` | Disable the OpenCV preview window entirely for maximum performance. |
| `--hud-scale FLOAT` | `1.0` | Scale factor for HUD text size in the camera preview. |
| `--hud-alpha INT` | `140` | Opacity (`0-255`) for the HUD background panel. |
| `--camera-width INT` | `640` | Camera capture width for both preview and inference. |
| `--camera-height INT` | `480` | Camera capture height for both preview and inference. |
| `--inference-every INT` | `1` | Run MediaPipe inference every N frames to reduce CPU load (1 = every frame). |
| `--auto-calib-seconds FLOAT` | `0.0` | Rolling auto-calibration window length in seconds (`>0` enables, e.g., `5.0`; `0` disables). |

## Project structure
- `src/control_types.py` — shared control state dataclass, smoothing helper, and control interface.
- `src/game.py` — pygame game loop and rendering (ball trail + hit flash).
- `src/vision.py` — OpenCV + MediaPipe palm/index tracking and pinch detection.
- `src/main.py` — Binds the chosen control source to the game.
- `requirements.txt` — Python dependencies.

## Tips
- Use `--no-camera` to develop or play without the camera.
- Adjust smoothing to taste with `--smoothing-alpha 0.15` (lower = smoother).
- If the paddle hits the wall early, first adjust camera distance; optionally enable auto-calibration (`--auto-calib-seconds 5.0`) or run manual calibration with `C`.
- Good lighting and keeping your hand within the frame improve detection stability.
- If the paddle feels mirrored relative to your hand, try `--mirror` first; some webcams already flip the image, so toggling this once usually resolves it.
- If MediaPipe wheels give trouble on Windows/Python 3.12, pin versions from `requirements.txt` and install via `python -m pip install --upgrade pip` first.

## Troubleshooting
- Ensure your hand is fully in frame with even lighting; partial hands reduce tracking stability.
- On Windows/Python 3.12, if MediaPipe or OpenCV fails to install or load, pin to the known versions listed in `requirements.txt`, for example: `python -m pip install --upgrade pip` then `python -m pip install \"mediapipe==0.10.14\" \"opencv-python==4.9.0.80\"`.
- If motion feels jittery, try lowering `--smoothing-alpha`; if it lags, raise it or temporarily switch to keyboard with `--no-camera`.

## 簡単なまとめ (Japanese)
- 手のひら中心がデフォルトで安定操作、指先モードも選択可能。つまむ動作でボールを開始/再開でき、キーボードでも遊べます。
- デフォルトはキャリブレーションなしですぐ遊べます。必要に応じて `--auto-calib-seconds 5.0` で自動補正、`C` で手動キャリブレーション（左右でピンチすると ✅ が出る、`R` でリセット）を使えます。キー案内は HUD に常時表示、`Esc` で終了。
- `python -m src.main` でカメラ操作、`--no-camera` でキーボードのみ、`--mirror` や `--smoothing-alpha` で調整できます。映像が重い場合は `--camera-width 640 --camera-height 480 --inference-every 2 --no-debug-window` で軽量化できます。
- Windows / Python 3.12 で MediaPipe や OpenCV がうまく入らない場合は、上記の例のようにバージョンを固定して再インストールしてください。
