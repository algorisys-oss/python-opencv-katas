---
slug: 108-live-motion-detection-alarm
title: Live Motion Detection Alarm
level: live
concepts: [cv2.absdiff, frame differencing, motion zones, contour area threshold, visual alarm]
prerequisites: [100-live-camera-fps, 73-frame-differencing]
---

## What Problem Are We Solving?

Motion detection is the backbone of security cameras, wildlife monitoring, and smart home systems. The core idea is deceptively simple: if nothing is moving, consecutive frames look almost identical. When something moves, the difference between frames produces non-zero regions that reveal the motion. But raw frame differencing is noisy — sensor noise, slight camera vibrations, and lighting flicker all produce small differences even in a static scene. A practical motion detector needs to threshold the difference, filter out noise, quantify the motion, and trigger an alarm when motion exceeds a threshold.

This kata builds a complete motion detection system with visual alarm feedback: red border flashing when motion is detected, bounding boxes around motion zones, and a motion percentage indicator.

## Frame Differencing with cv2.absdiff

`cv2.absdiff` computes the absolute difference between two images, pixel by pixel:

```python
diff = cv2.absdiff(frame1_gray, frame2_gray)
```

| Parameter | Type | Meaning |
|---|---|---|
| `src1` | ndarray | First grayscale frame |
| `src2` | ndarray | Second grayscale frame (same size and type) |
| **Returns** | ndarray | Absolute per-pixel difference |

Each pixel in the output is `|frame1[y,x] - frame2[y,x]|`. Static regions produce values near 0; moving regions produce higher values.

### Why Grayscale?

You could diff color frames, but:
1. Grayscale is 3x faster (one channel vs three)
2. Motion is about spatial change, not color change
3. The threshold step requires a single-channel image anyway

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to suppress sensor noise
```

The Gaussian blur is critical — without it, sensor noise between frames produces a speckled difference image even with no actual motion.

## Thresholding the Difference

The raw difference image contains low-value noise everywhere. Thresholding converts it to a clean binary mask:

```python
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
```

| Parameter | Value | Meaning |
|---|---|---|
| `diff` | — | Grayscale difference image |
| `25` | — | Threshold value: differences below 25 are treated as noise |
| `255` | — | Value assigned to pixels above threshold |
| `cv2.THRESH_BINARY` | — | Simple binary thresholding |

Choosing the threshold value:

| Threshold | Behavior |
|---|---|
| 10-15 | Very sensitive — detects subtle lighting changes, camera shake |
| 20-30 | Moderate — good for indoor scenes with stable lighting |
| 40-60 | Low sensitivity — only detects significant motion |

After thresholding, dilate the mask to connect nearby motion regions:

```python
thresh = cv2.dilate(thresh, None, iterations=2)
```

## Contour-Based Motion Zones

Find contours on the thresholded image to identify distinct motion regions:

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) < MIN_AREA:
        continue  # Skip small noise regions
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

The `MIN_AREA` filter is essential. Without it, you get hundreds of tiny rectangles from noise. A typical minimum area is 500-3000 pixels depending on frame resolution and how close subjects are to the camera.

| Frame Resolution | Suggested MIN_AREA | Rationale |
|---|---|---|
| 320x240 | 300-500 | Small frame, small features |
| 640x480 | 500-1500 | Standard webcam resolution |
| 1280x720 | 1500-3000 | HD — larger pixel counts per object |

## Calculating Motion Percentage

Motion percentage tells you **how much of the frame** is moving, which is more useful than raw pixel counts:

```python
motion_pixels = cv2.countNonZero(thresh)
total_pixels = thresh.shape[0] * thresh.shape[1]
motion_pct = (motion_pixels / total_pixels) * 100
```

| Motion % | Interpretation |
|---|---|
| 0-0.5% | No meaningful motion (noise floor) |
| 0.5-5% | Small motion — a person walking, hand waving |
| 5-20% | Moderate motion — person walking close to camera |
| 20%+ | Large motion — camera shake, scene change, lights toggle |

## Visual Alarm: Flashing Border

A flashing red border provides unmistakable visual feedback when motion exceeds your alarm threshold:

```python
ALARM_THRESHOLD = 3.0  # Trigger alarm above 3% motion

if motion_pct > ALARM_THRESHOLD:
    # Flash: alternate border based on frame count
    border_color = (0, 0, 255) if (frame_count % 4 < 2) else (0, 0, 200)
    thickness = 8
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)
    cv2.putText(frame, "MOTION DETECTED!", (w // 2 - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
```

The flash effect uses modular arithmetic on the frame counter: every 2 frames, the border alternates between bright red and dark red, creating a visible flash at 15 FPS.

## Two-Frame vs Three-Frame Differencing

The basic approach diffs the current frame against the previous one. A more robust variant uses three frames:

```python
# Two-frame: detects leading and trailing edges of motion
diff = cv2.absdiff(prev_gray, curr_gray)

# Three-frame: detects only the region where motion is happening NOW
diff1 = cv2.absdiff(prev_gray, curr_gray)
diff2 = cv2.absdiff(curr_gray, next_gray)
motion = cv2.bitwise_and(diff1, diff2)
```

Three-frame differencing reduces "ghost" artifacts (the shadow left behind when an object moves away), but requires buffering one extra frame and is slightly slower.

## Keyboard Sensitivity Adjustment

Since different environments require different sensitivity levels, allow runtime tuning with keyboard controls:

```python
key = cv2.waitKey(1) & 0xFF

if key == ord('t'):    # Increase threshold (less sensitive)
    DIFF_THRESHOLD = min(DIFF_THRESHOLD + 5, 100)
elif key == ord('T'):  # Decrease threshold (more sensitive)
    DIFF_THRESHOLD = max(DIFF_THRESHOLD - 5, 5)
elif key == ord('a'):  # Increase alarm threshold
    ALARM_THRESHOLD = min(ALARM_THRESHOLD + 0.5, 50.0)
elif key == ord('A'):  # Decrease alarm threshold
    ALARM_THRESHOLD = max(ALARM_THRESHOLD - 0.5, 0.5)
```

Display the current thresholds on the frame so the user can see the sensitivity level while tuning.

## Tips & Common Mistakes

- **Always blur before differencing.** Camera sensor noise creates a speckled difference image even with no motion. A 21x21 Gaussian blur eliminates this.
- The first frame has no previous frame to compare against. Initialize `prev_gray` to `None` and skip processing on the first iteration.
- Auto-exposure adjustments cause the entire frame to brighten or darken, triggering a false alarm across the whole image. Disable auto-exposure if possible, or increase the threshold.
- `cv2.dilate` after thresholding connects nearby motion blobs into coherent regions. Without it, a single moving person may produce dozens of tiny disconnected contours.
- The alarm threshold (motion percentage) needs tuning for your specific scene. Start with 3% and adjust using keyboard controls.
- Sudden lighting changes (turning on a light, clouds passing) will trigger motion detection. For production systems, consider background subtraction (MOG2) instead of frame differencing.
- `cv2.absdiff` requires both frames to have the same size and type. If you resize frames for performance, ensure both the current and previous frames are resized.
- On laptops, the webcam indicator light turning on/off can cause a brief brightness change that triggers false alarms on the first few frames. Skip the first 10 frames or use a warm-up period.
- Use the keyboard sensitivity adjustment to find the right threshold for your scene before settling on hardcoded values.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Keep still for a moment — the motion percentage should stay near 0% with no alarm
- Wave your hand or walk in front of the camera — red bounding boxes should appear around motion zones and the motion bar should rise
- Move enough to exceed the alarm threshold — the border should flash red and "MOTION DETECTED!!" should appear
- Press **'t'/'T'** to adjust the pixel difference threshold and **'a'/'A'** to raise or lower the alarm trigger percentage

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)

# --- Configuration ---
DIFF_THRESHOLD = 25       # Pixel difference threshold
MIN_CONTOUR_AREA = 1000   # Minimum motion region area (pixels)
ALARM_THRESHOLD = 3.0     # Motion percentage to trigger alarm
BLUR_KERNEL = (21, 21)    # Gaussian blur size for noise suppression

prev_gray = None
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
        frame_count += 1

        # --- Preprocess: grayscale + blur ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

        # --- Initialize previous frame on first iteration ---
        if prev_gray is None:
            prev_gray = gray
            continue

        # --- Frame differencing ---
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray

        # --- Threshold the difference ---
        _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # --- Calculate motion percentage ---
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_pct = (motion_pixels / total_pixels) * 100

        # --- Find motion zones ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_zones = 0

        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            motion_zones += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # --- Visual alarm: flashing red border ---
        alarm_active = motion_pct > ALARM_THRESHOLD
        if alarm_active:
            border_color = (0, 0, 255) if (frame_count % 4 < 2) else (0, 0, 180)
            h_frame, w_frame = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), border_color, 8)
            cv2.putText(frame, "!! MOTION DETECTED !!", (w_frame // 2 - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Motion bar indicator ---
        bar_width = int(min(motion_pct, 100) / 100 * 200)
        bar_color = (0, 0, 255) if alarm_active else (0, 255, 0)
        cv2.rectangle(frame, (10, 100), (210, 120), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 100), (10 + bar_width, 120), bar_color, -1)
        cv2.rectangle(frame, (10, 100), (210, 120), (200, 200, 200), 1)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Motion: {motion_pct:.1f}%  Zones: {motion_zones}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Alarm threshold: {ALARM_THRESHOLD}%", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "t/T: threshold +/-  a/A: alarm +/-  q: quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Motion Detection Alarm', frame)

        # --- Keyboard controls for sensitivity ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            DIFF_THRESHOLD = min(DIFF_THRESHOLD + 5, 100)
        elif key == ord('T'):
            DIFF_THRESHOLD = max(DIFF_THRESHOLD - 5, 5)
        elif key == ord('a'):
            ALARM_THRESHOLD = min(ALARM_THRESHOLD + 0.5, 50.0)
        elif key == ord('A'):
            ALARM_THRESHOLD = max(ALARM_THRESHOLD - 0.5, 0.5)

finally:
    cap.release()
    cv2.destroyAllWindows()
```
