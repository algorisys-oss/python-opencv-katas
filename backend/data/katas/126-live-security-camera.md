---
slug: 126-live-security-camera
title: Live Security Camera System
level: live
concepts: [motion detection, cv2.CascadeClassifier, timestamp overlay, event logging, recording trigger]
prerequisites: [108-live-motion-detection-alarm, 105-live-face-detection]
---

## What Problem Are We Solving?

Real security camera systems don't just record video -- they **analyze** it. They detect motion to know when something is happening, identify faces to know **who** is there, overlay timestamps for legal evidence, log events so operators can review activity, and highlight active zones so that a human monitoring multiple feeds can immediately see which camera needs attention.

This kata combines multiple detection pipelines into a single coherent system: motion detection (via frame differencing), face detection (via Haar cascades), timestamp overlay, console event logging, and visual status indicators. It teaches you how to architect a multi-stage processing pipeline where the output of one stage feeds into the decisions of another.

## System Architecture

The security camera pipeline processes each frame through multiple stages:

```
Frame Capture
    |
    v
Motion Detection -----> Log "MOTION" event
    |                    Highlight motion zones
    v
Face Detection -------> Log "FACE" event
    |                    Draw face rectangles
    v
Timestamp Overlay        Snapshot capture
    |
    v
Status Indicators
    |
    v
Display
```

Each stage runs independently, but they share the same frame. Motion detection runs first because it's cheap (frame differencing), and face detection only needs to run when motion is detected -- this optimization dramatically reduces CPU usage.

## Motion Detection Pipeline

Motion detection uses the frame differencing technique from kata 73, with contour analysis to identify distinct motion zones:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
diff = cv2.absdiff(baseline_gray, gray)
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
thresh = cv2.dilate(thresh, None, iterations=2)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

| Parameter | Value | Why |
|---|---|---|
| Blur kernel | `(21, 21)` | Large kernel suppresses noise while keeping motion regions |
| Threshold | `25` | Balances sensitivity vs. noise rejection |
| Dilation iterations | `2` | Connects nearby motion fragments into solid regions |
| Min contour area | `1000` | Filters out small noise blobs (insects, shadows) |

### Baseline Frame Strategy

Rather than comparing consecutive frames (which misses slow-moving objects), compare against a **baseline frame** captured at startup. Update the baseline periodically or on command:

```python
baseline_gray = gray.copy()  # Set at startup

# Optionally update baseline with slow adaptation:
cv2.accumulateWeighted(gray, baseline_float, 0.005)
baseline_gray = cv2.convertScaleAbs(baseline_float)
```

`cv2.accumulateWeighted` slowly adapts the baseline to gradual changes (lighting shifts) while still detecting sudden motion.

| Parameter | Type | Description |
|---|---|---|
| `src` | `np.ndarray` | New input image (float or uint8) |
| `dst` | `np.ndarray` | Accumulator image (must be float32/float64) |
| `alpha` | `float` | Learning rate (0.0-1.0). Lower = slower adaptation |

## Face Detection with Haar Cascades

Haar cascade classifiers are pre-trained XML files that detect specific objects. OpenCV ships with classifiers for faces, eyes, and more:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                       minNeighbors=5, minSize=(30, 30))
```

| Parameter | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Grayscale input image |
| `scaleFactor` | `float` | Image size reduction at each scale (1.1 = 10% reduction) |
| `minNeighbors` | `int` | Minimum overlapping detections required (higher = fewer false positives) |
| `minSize` | `tuple` | Minimum face size in pixels `(width, height)` |
| **Returns** | `np.ndarray` | Array of `(x, y, w, h)` rectangles |

### Performance Optimization

Face detection is expensive. Run it conditionally:

```python
if motion_detected:
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
else:
    faces = []
```

This avoids wasting CPU on face detection when the scene is static.

## Timestamp Overlay

Every security camera needs a visible timestamp for legal and forensic purposes:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
cv2.putText(frame, timestamp, (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
```

Place the timestamp in a consistent location (bottom-left or top-right) with a contrasting color. Adding a dark background rectangle behind the text ensures readability against any scene:

```python
text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
cv2.rectangle(frame, (8, height - 25), (12 + text_size[0], height - 5), (0, 0, 0), -1)
cv2.putText(frame, timestamp, (10, height - 10), ...)
```

## Event Logging

Print structured events to the console with timestamps. In a production system these would go to a file or database:

```python
def log_event(event_type, details=""):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {event_type}: {details}")
```

Use **cooldown timers** to avoid flooding the log with repeated events:

```python
last_motion_log = 0
motion_log_cooldown = 2.0  # seconds

if motion_detected and (time.time() - last_motion_log) > motion_log_cooldown:
    log_event("MOTION", f"{len(motion_contours)} regions, area={total_area}")
    last_motion_log = time.time()
```

## Visual Status Indicators

A recording indicator (red dot) and status text help the operator understand the system state at a glance:

```python
# Recording indicator (blinking red dot)
if int(time.time() * 2) % 2 == 0:  # Blink at 1 Hz
    cv2.circle(frame, (width - 25, 25), 8, (0, 0, 255), -1)
cv2.putText(frame, "REC", (width - 60, 30), ...)

# Motion status bar
status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
cv2.rectangle(frame, (0, 0), (width, 5), status_color, -1)
```

| Indicator | Meaning |
|---|---|
| Green bar at top | Scene is calm, no motion |
| Red bar at top | Motion detected |
| Blinking red dot | System is active/recording |
| Blue rectangle on face | Face detected and tracked |
| Green rectangles | Motion zones highlighted |

## Tips & Common Mistakes

- Initialize the Haar cascade **once** outside the loop. Loading the XML file every frame is extremely slow and unnecessary.
- Use `cv2.data.haarcascades` to find the cascade file path. Hardcoding paths breaks across platforms.
- Run face detection only when motion is detected. This can reduce CPU usage by 80% or more in quiet scenes.
- The baseline frame for motion detection should be captured after the camera auto-exposure stabilizes (skip the first 30-60 frames or wait 2 seconds).
- Use cooldown timers for logging. Without them, a continuously moving object generates hundreds of log entries per second.
- Face detection returns bounding boxes that may extend outside the frame. Clamp coordinates before using them for cropping or drawing.
- `detectMultiScale` with `minNeighbors=3` produces many false positives. Use `minNeighbors=5` or higher for reliable results.
- Increasing `scaleFactor` (e.g., from 1.1 to 1.3) makes face detection faster but may miss smaller faces.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Wait for the "Initializing..." message to finish (about 1 second), then walk into frame — you should see green "MOTION" bounding boxes and blue "FACE" labels, with events logged to the console
- Check that the top status bar turns red during motion and returns to green when the scene is still
- Press **s** to save a snapshot (a timestamped PNG file) and **r** to reset the motion baseline
- Verify the blinking red "REC" indicator and timestamp overlay are visible at the corners of the frame

## Starter Code

```python
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# --- Load face detector ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("Error: Could not load face cascade classifier")
    exit()

# --- Motion detection state ---
baseline_gray = None
min_contour_area = 1000

# --- Logging state ---
def log_event(event_type, details=""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {event_type}: {details}")

last_motion_log = 0
last_face_log = 0
log_cooldown = 2.0  # seconds between repeated log entries

# --- FPS tracking ---
frame_times = deque(maxlen=30)
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame_count += 1
        frame_times.append(time.time())
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps = 0.0

        # --- Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize baseline on first frame (skip a few frames for auto-exposure)
        if baseline_gray is None:
            if frame_count > 30:
                baseline_gray = gray.copy()
                log_event("SYSTEM", "Baseline frame captured")
            else:
                cv2.putText(frame, "Initializing...", (width // 2 - 80, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow('Security Camera', frame)
                cv2.waitKey(1)
                continue

        # ==========================================
        # STAGE 1: Motion Detection
        # ==========================================
        diff = cv2.absdiff(baseline_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_regions = 0
        total_motion_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_contour_area:
                motion_detected = True
                motion_regions += 1
                total_motion_area += area
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "MOTION", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Log motion events with cooldown
        now = time.time()
        if motion_detected and (now - last_motion_log) > log_cooldown:
            log_event("MOTION", f"{motion_regions} region(s), total area={total_motion_area}")
            last_motion_log = now

        # ==========================================
        # STAGE 2: Face Detection (only if motion)
        # ==========================================
        faces = []
        if motion_detected:
            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(frame, "FACE", (fx, fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Log face events with cooldown
        if len(faces) > 0 and (now - last_face_log) > log_cooldown:
            log_event("FACE", f"{len(faces)} face(s) detected")
            last_face_log = now

        # ==========================================
        # STAGE 3: Timestamp Overlay
        # ==========================================
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (8, height - 28), (14 + ts_size[0], height - 6),
                      (0, 0, 0), -1)
        cv2.putText(frame, timestamp, (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # ==========================================
        # STAGE 4: Status Indicators
        # ==========================================
        # Top status bar: green = calm, red = motion
        bar_color = (0, 0, 255) if motion_detected else (0, 180, 0)
        cv2.rectangle(frame, (0, 0), (width, 4), bar_color, -1)

        # Blinking recording indicator
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (width - 20, 20), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # FPS display
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Detection summary
        status_text = f"Motion: {'YES' if motion_detected else 'NO'}  Faces: {len(faces)}"
        cv2.putText(frame, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, "'r'=reset baseline  's'=snapshot  'q'=quit",
                    (10, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Security Camera', frame)

        # --- Handle keypresses ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            baseline_gray = gray.copy()
            log_event("SYSTEM", "Baseline reset by user")
        elif key == ord('s'):
            filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, frame)
            log_event("SNAPSHOT", f"Saved {filename}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    log_event("SYSTEM", "Camera released. Goodbye!")
```
