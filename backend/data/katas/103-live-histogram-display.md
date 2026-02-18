---
slug: 103-live-histogram-display
title: Live Histogram Display
level: live
concepts: [cv2.calcHist, real-time histogram, exposure analysis, per-channel visualization]
prerequisites: [100-live-camera-fps, 18-understanding-histograms]
---

## What Problem Are We Solving?

A histogram tells you everything about an image's exposure at a glance — is it too dark? Too bright? Is the contrast washed out? In photography and videography, professionals monitor a **live histogram** to ensure proper exposure while filming. In computer vision, watching the histogram in real-time helps you understand how lighting changes affect your processing pipeline.

This kata builds a real-time histogram display that runs alongside your camera feed, showing per-channel (B, G, R) distributions updating every frame.

## Why Monitor Histograms in Real-Time?

Static histogram analysis (kata 18) shows you the distribution of a single image. But in live video, conditions change constantly:

- **Lighting shifts** — Someone turns on a light, clouds cover the sun, camera auto-exposure adjusts.
- **Scene changes** — Panning from a dark room to a bright window.
- **Processing feedback** — When applying filters or adjustments, the histogram shows their effect instantly.

A live histogram lets you:
- Detect **overexposure** (spike at 255 — highlights clipped).
- Detect **underexposure** (spike at 0 — shadows crushed).
- Monitor **contrast** (spread of the distribution).
- Verify color balance (compare B, G, R channel shapes).

## Computing Per-Channel Histograms

For a BGR color image, compute three separate histograms:

```python
hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])  # Blue
hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])  # Green
hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])  # Red
```

Each histogram is a `(256, 1)` float32 array. `hist[i]` = number of pixels with intensity `i` in that channel.

### Normalization for Display

Raw histograms have huge values (hundreds of thousands of pixels per bin). Normalize to fit in your display area:

```python
hist_h = 200  # Height of histogram canvas
cv2.normalize(hist_b, hist_b, 0, hist_h, cv2.NORM_MINMAX)
cv2.normalize(hist_g, hist_g, 0, hist_h, cv2.NORM_MINMAX)
cv2.normalize(hist_r, hist_r, 0, hist_h, cv2.NORM_MINMAX)
```

`cv2.NORM_MINMAX` scales values so the minimum becomes 0 and the maximum becomes `hist_h`. This ensures the tallest bar exactly fills the canvas height.

## Drawing the Histogram

Since we're using only OpenCV (no matplotlib), draw bars manually:

```python
def draw_histogram(hist_b, hist_g, hist_r, width=256, height=200):
    """Draw overlapping B, G, R histograms on a black canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(width):
        # Blue channel
        bh = int(hist_b[x])
        if bh > 0:
            cv2.line(canvas, (x, height), (x, height - bh), (255, 0, 0), 1)
        # Green channel
        gh = int(hist_g[x])
        if gh > 0:
            cv2.line(canvas, (x, height), (x, height - gh), (0, 255, 0), 1)
        # Red channel
        rh = int(hist_r[x])
        if rh > 0:
            cv2.line(canvas, (x, height), (x, height - rh), (0, 0, 255), 1)

    return canvas
```

### Drawing with Polylines (Smoother)

For a line-graph style (smoother, less cluttered with overlapping channels):

```python
def draw_histogram_lines(hist_b, hist_g, hist_r, width=256, height=200):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for hist, color in [(hist_b, (255, 100, 0)), (hist_g, (0, 255, 0)), (hist_r, (0, 0, 255))]:
        points = []
        for x in range(width):
            y = height - int(hist[x][0])
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)

    return canvas
```

`cv2.polylines` draws a smooth connected line through all points — much cleaner for overlapping channels.

## Reading Histogram Statistics

Beyond visualization, extract numerical stats:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mean_brightness = np.mean(gray)
std_dev = np.std(gray)
min_val = np.min(gray)
max_val = np.max(gray)

# Dynamic range (how much of the 0-255 range is used)
dynamic_range = max_val - min_val

# Exposure assessment
if mean_brightness < 60:
    exposure = "UNDEREXPOSED"
elif mean_brightness > 200:
    exposure = "OVEREXPOSED"
else:
    exposure = "OK"
```

| Metric | What It Tells You |
|---|---|
| Mean brightness | Overall exposure level (ideal: 100-160) |
| Standard deviation | Contrast (low = flat/washed out, high = punchy) |
| Min/Max | Clipping detection (min=0 = shadow clipping, max=255 = highlight clipping) |
| Dynamic range | How much of the tonal range is used (ideal: close to 255) |

## Handling Performance

Computing three histograms every frame adds processing time. Optimizations:

```python
# Option 1: Resize frame before computing histogram (fastest)
small = cv2.resize(frame, (160, 120))
hist_b = cv2.calcHist([small], [0], None, [256], [0, 256])

# Option 2: Compute every N frames instead of every frame
if frame_count % 3 == 0:
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    # ... reuse last histogram for other frames

# Option 3: Use fewer bins for faster computation
hist_b = cv2.calcHist([frame], [0], None, [64], [0, 256])  # 64 bins instead of 256
```

In practice, `cv2.calcHist` is highly optimized in C++ — even at full resolution and 256 bins, it typically takes <1ms per channel.

## Tips & Common Mistakes

- Normalize histograms **every frame** — the distribution changes, so the normalization must update.
- Use polylines for overlapping channels — bar charts get messy when three channels overlap.
- `cv2.calcHist` requires the image in a **list**: `[frame]`, not just `frame`.
- Channel indices: `[0]` = Blue, `[1]` = Green, `[2]` = Red (BGR order).
- The histogram range `[0, 256]` has an exclusive upper bound — it covers pixel values 0 through 255.
- On auto-exposure cameras, the histogram shifts constantly as the camera adjusts. Disable auto-exposure (`cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)`) for stable readings.
- A spike at bin 0 or bin 255 means **clipping** — detail is lost in shadows or highlights.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Verify the B, G, R histogram curves update in real-time in the right panel alongside the camera feed
- Cover the camera lens with your hand — the histogram should shift heavily toward the left (low values) and the stats panel should show "UNDEREXPOSED"
- Point the camera at a bright light or window — the histogram should shift right and may show "OVEREXPOSED"
- Wave a brightly colored object in front of the camera and watch the corresponding channel spike in the histogram

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

HIST_W = 256
HIST_H = 200

frame_times = deque(maxlen=30)

def draw_histogram_lines(hist_b, hist_g, hist_r):
    """Draw overlapping B, G, R histogram curves."""
    canvas = np.zeros((HIST_H, HIST_W, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)  # Dark gray background

    # Draw grid lines
    for y_val in [64, 128, 192]:
        y_pos = HIST_H - int(y_val * HIST_H / 256)
        cv2.line(canvas, (0, y_pos), (HIST_W, y_pos), (40, 40, 40), 1)
    for x_val in [64, 128, 192]:
        cv2.line(canvas, (x_val, 0), (x_val, HIST_H), (40, 40, 40), 1)

    for hist, color in [(hist_b, (255, 100, 0)), (hist_g, (0, 220, 0)), (hist_r, (0, 0, 255))]:
        pts = np.array([[x, HIST_H - int(hist[x][0])] for x in range(HIST_W)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)

    # Axis labels
    cv2.putText(canvas, '0', (2, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    cv2.putText(canvas, '255', (HIST_W - 30, HIST_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    return canvas

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Compute per-channel histograms ---
        hist_b = cv2.calcHist([frame], [0], None, [HIST_W], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [HIST_W], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [HIST_W], [0, 256])

        # Normalize to canvas height
        cv2.normalize(hist_b, hist_b, 0, HIST_H - 10, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, HIST_H - 10, cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, 0, HIST_H - 10, cv2.NORM_MINMAX)

        # --- Compute statistics ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))

        if mean_val < 60:
            exposure = "UNDEREXPOSED"
            exp_color = (0, 100, 255)
        elif mean_val > 200:
            exposure = "OVEREXPOSED"
            exp_color = (0, 0, 255)
        else:
            exposure = "GOOD"
            exp_color = (0, 255, 0)

        # --- Draw histogram canvas ---
        hist_canvas = draw_histogram_lines(hist_b, hist_g, hist_r)

        # --- Draw stats panel below histogram ---
        stats_h = 80
        stats_panel = np.zeros((stats_h, HIST_W, 3), dtype=np.uint8)
        stats_panel[:] = (30, 30, 30)

        cv2.putText(stats_panel, f"Mean: {mean_val:.0f}  Std: {std_val:.0f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(stats_panel, f"Range: [{min_val}, {max_val}]  DR: {max_val - min_val}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(stats_panel, f"Exposure: {exposure}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, exp_color, 1)

        # Stack histogram + stats
        right_panel = np.vstack([hist_canvas, stats_panel])

        # --- Resize right panel to match frame height ---
        frame_h = frame.shape[0]
        right_panel = cv2.resize(right_panel, (HIST_W, frame_h))

        # --- Draw FPS on frame ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Combine: frame + histogram panel ---
        display = np.hstack([frame, right_panel])

        cv2.imshow('Live Histogram', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
