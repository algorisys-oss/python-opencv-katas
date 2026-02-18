---
slug: 100-live-camera-fps
title: Live Camera Feed with FPS Overlay
level: live
concepts: [cv2.VideoCapture, FPS calculation, cv2.putText, real-time loop]
prerequisites: [70-reading-video-files, 71-webcam-capture]
---

## What Problem Are We Solving?

Every real-time computer vision application starts with the same foundation: capturing frames from a camera, processing them, and displaying the result — all inside a loop that runs as fast as possible. Understanding this loop, measuring its performance (frames per second), setting camera properties, and shutting down cleanly is the bedrock of every live camera project that follows.

This kata teaches you the **production-quality camera loop pattern** that all subsequent live camera katas build on.

## The Real-Time Camera Loop

The core pattern for any live camera application is a `while True` loop that captures, processes, and displays frames:

```python
cap = cv2.VideoCapture(0)       # Open default camera
while True:
    ret, frame = cap.read()     # Capture one frame
    if not ret:
        break
    # ... process frame ...
    cv2.imshow('Live', frame)   # Display result
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break                   # Exit on 'q' key
cap.release()                   # Release camera
cv2.destroyAllWindows()         # Close windows
```

Every piece of this pattern matters:

| Line | Purpose |
|---|---|
| `cv2.VideoCapture(0)` | Opens camera device 0 (usually the built-in webcam) |
| `cap.read()` | Grabs and decodes one frame; returns `(success_bool, frame_array)` |
| `if not ret: break` | Handles camera disconnect or end of stream gracefully |
| `cv2.imshow(...)` | Displays the frame in a named window |
| `cv2.waitKey(1)` | Waits 1ms for a keypress; **required** for imshow to actually render |
| `& 0xFF` | Bitmask needed on some platforms (Linux) to extract the key code correctly |
| `ord('q')` | Compare against ASCII code of 'q' for exit |
| `cap.release()` | Frees the camera hardware — **always** do this |
| `cv2.destroyAllWindows()` | Closes all OpenCV GUI windows |

## Understanding cv2.VideoCapture

`cv2.VideoCapture` is the unified interface for reading from cameras, video files, and network streams:

```python
cap = cv2.VideoCapture(0)        # Camera index 0 (default webcam)
cap = cv2.VideoCapture(1)        # Camera index 1 (external USB camera)
cap = cv2.VideoCapture("video.mp4")  # Video file
cap = cv2.VideoCapture("rtsp://...")  # Network stream
```

### Checking if the Camera Opened Successfully

Always verify the camera is available before entering the loop:

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()
```

Common reasons `isOpened()` returns `False`:
- No camera connected
- Camera is in use by another application
- Incorrect device index
- Missing camera drivers

### Camera Properties

You can query and set camera properties using `cap.get()` and `cap.set()`:

```python
# Query current settings
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {width}x{height}, FPS: {fps}")
```

```python
# Set resolution (request — camera may not support it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set FPS (request)
cap.set(cv2.CAP_PROP_FPS, 30)
```

| Property | Constant | Typical Values |
|---|---|---|
| Frame width | `cv2.CAP_PROP_FRAME_WIDTH` | 640, 1280, 1920 |
| Frame height | `cv2.CAP_PROP_FRAME_HEIGHT` | 480, 720, 1080 |
| FPS | `cv2.CAP_PROP_FPS` | 15, 30, 60 |
| Brightness | `cv2.CAP_PROP_BRIGHTNESS` | 0-255 (camera dependent) |
| Contrast | `cv2.CAP_PROP_CONTRAST` | 0-255 (camera dependent) |
| Auto-exposure | `cv2.CAP_PROP_AUTO_EXPOSURE` | 0 or 1 |

> **Important:** `cap.set()` sends a **request** to the camera driver. The camera may ignore it if it doesn't support that resolution or FPS. Always read back with `cap.get()` to confirm what was actually set.

## Measuring FPS (Frames Per Second)

FPS tells you how fast your pipeline is processing frames. There are two approaches:

### Approach 1: Instantaneous FPS (per-frame timing)

```python
import time

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

This measures the time between consecutive frames. It fluctuates a lot because individual frame times vary.

### Approach 2: Smoothed FPS (rolling average)

```python
import time
from collections import deque

frame_times = deque(maxlen=30)  # Keep last 30 frame timestamps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_times.append(time.time())

    if len(frame_times) > 1:
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
    else:
        fps = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

The rolling average gives you a **stable, readable** number. Use `maxlen=30` for a smooth reading or `maxlen=10` for faster response.

## What Affects FPS?

Your actual FPS depends on three bottlenecks:

1. **Camera capture rate** — The camera hardware itself (e.g., 30 FPS max for most webcams).
2. **Processing time** — How long your `process(frame)` step takes. Heavy processing (blur, detection, DNN) slows this down.
3. **Display overhead** — `cv2.imshow` and `cv2.waitKey` add a small delay.

```
Actual FPS = min(camera_fps, 1 / processing_time)
```

If your processing takes 50ms per frame, you can't exceed 20 FPS regardless of camera speed.

## The cv2.waitKey() Timing Parameter

`cv2.waitKey(delay)` serves two purposes:
1. **Renders the GUI** — Without it, `imshow` windows won't update.
2. **Waits for keypresses** — The `delay` parameter is the wait time in milliseconds.

```python
cv2.waitKey(1)    # Wait 1ms — fast as possible for real-time
cv2.waitKey(33)   # Wait 33ms — roughly limits display to ~30 FPS
cv2.waitKey(0)    # Wait forever — use for static images, NOT live video
```

For live video, **always use `cv2.waitKey(1)`**. Using `0` will freeze the feed after each frame.

## Graceful Shutdown

Always release resources in a `try/finally` block to handle unexpected errors:

```python
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

This ensures the camera is released even if an exception occurs mid-loop.

## Tips & Common Mistakes

- Always check `cap.isOpened()` before the loop — catching errors early saves debugging time.
- Always call `cap.release()` when done. Forgetting this can lock the camera until the process is killed.
- `cv2.waitKey(1)` is **mandatory** for `imshow` to work. Without it, the window will not render.
- The `& 0xFF` bitmask is needed on Linux (64-bit `waitKey` return value). On Windows it's optional but doesn't hurt.
- Camera index `0` is usually the built-in webcam. External USB cameras are typically `1`, `2`, etc.
- Setting resolution to an unsupported value silently falls back to the nearest supported resolution.
- FPS measurement should use `time.time()` or `time.perf_counter()`, not `cv2.getTickCount()` (which is lower-level and less portable).
- If the camera feed appears laggy, reduce the resolution — `640x480` processes much faster than `1920x1080`.
- On macOS, the first `cap.read()` after opening the camera may take a few seconds while the driver initializes. This is normal.

## Starter Code

```python
import cv2
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# --- Set resolution (optional — adjust to your camera) ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read back actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# --- FPS tracking with rolling average ---
frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Record frame timestamp
        frame_times.append(time.time())

        # Calculate smoothed FPS
        if len(frame_times) > 1:
            elapsed = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0
        else:
            fps = 0

        # --- Draw FPS overlay ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Draw resolution info ---
        cv2.putText(frame, f"{width}x{height}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Draw frame counter ---
        frame_count = len(frame_times)
        cv2.putText(frame, f"Press 'q' to quit", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # --- Display ---
        cv2.imshow('Live Camera Feed', frame)

        # --- Exit on 'q' key ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
