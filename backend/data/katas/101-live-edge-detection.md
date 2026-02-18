---
slug: 101-live-edge-detection
title: Live Edge Detection
level: live
concepts: [cv2.Canny, cv2.createTrackbar, real-time processing, threshold tuning]
prerequisites: [100-live-camera-fps, 32-canny-edge-detection]
---

## What Problem Are We Solving?

Edge detection is one of the most fundamental operations in computer vision — it reveals the structure of a scene by highlighting boundaries where pixel intensity changes sharply. But choosing the right threshold values for Canny edge detection is tricky: values that work for one scene may miss edges in another. Running edge detection **live** with adjustable trackbars lets you see the effect of each parameter **instantly** and develop intuition for threshold selection.

This kata teaches real-time parameter tuning — a technique you'll use in almost every live camera project.

## Canny Edge Detection — Quick Recap

`cv2.Canny()` is a multi-stage edge detection algorithm:

```python
edges = cv2.Canny(gray_image, threshold1, threshold2)
```

| Parameter | Meaning |
|---|---|
| `gray_image` | Input must be **single-channel** (grayscale) |
| `threshold1` | Lower threshold — edges below this are discarded |
| `threshold2` | Upper threshold — edges above this are strong edges |

The algorithm has four stages internally:

1. **Gaussian smoothing** — Reduces noise (uses 5x5 kernel internally).
2. **Gradient computation** — Finds intensity change magnitude and direction using Sobel.
3. **Non-maximum suppression** — Thins edges to single-pixel width by keeping only local maxima along the gradient direction.
4. **Hysteresis thresholding** — Uses the two thresholds to classify edges:
   - Gradient > `threshold2` → **strong edge** (always kept)
   - Gradient < `threshold1` → **discarded** (never an edge)
   - Between `threshold1` and `threshold2` → **weak edge** (kept only if connected to a strong edge)

### How the Two Thresholds Interact

```
threshold1 = 50,  threshold2 = 150  → Standard: catches main edges
threshold1 = 10,  threshold2 = 50   → Sensitive: catches subtle edges + noise
threshold1 = 100, threshold2 = 200  → Strict: only the strongest edges
```

The **ratio** between the thresholds matters more than absolute values. A common recommendation is `threshold2 = 2 * threshold1` or `threshold2 = 3 * threshold1`.

## OpenCV Trackbars for Real-Time Parameter Tuning

`cv2.createTrackbar()` adds a slider to an OpenCV window that lets you adjust values while the program runs:

```python
cv2.namedWindow('Controls')
cv2.createTrackbar('Threshold1', 'Controls', 50, 500, lambda x: None)
cv2.createTrackbar('Threshold2', 'Controls', 150, 500, lambda x: None)
```

| Parameter | Meaning |
|---|---|
| `'Threshold1'` | Trackbar label displayed in the window |
| `'Controls'` | Name of the window to attach the trackbar to |
| `50` | Initial value of the slider |
| `500` | Maximum value of the slider |
| `lambda x: None` | Callback function — called whenever the slider moves. We use a no-op because we read the value inside the loop instead. |

Reading the current slider position:

```python
t1 = cv2.getTrackbarPos('Threshold1', 'Controls')
t2 = cv2.getTrackbarPos('Threshold2', 'Controls')
```

This pattern — **create trackbars, read values each frame** — is the standard way to do interactive parameter tuning in OpenCV. It works for any numeric parameter: blur kernel size, HSV ranges, threshold values, etc.

## Pre-Processing: Why Blur Before Edge Detection?

Real camera feeds contain noise — random pixel variations from the sensor. Canny's internal Gaussian blur (5x5) helps, but for noisy scenes you may want additional smoothing:

```python
# Additional blur before Canny reduces noise-induced false edges
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, t1, t2)
```

The trade-off:
- **More blur** → fewer false edges from noise, but fine details may be lost
- **Less blur** → preserves fine edges, but noise produces speckle artifacts

A trackbar for blur kernel size lets you tune this interactively too.

## Displaying Edges on the Live Feed

There are several ways to visualize edges alongside the original:

### Side-by-side display

```python
# Convert edges to BGR for stacking (edges is single-channel)
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
combined = np.hstack([frame, edges_bgr])
cv2.imshow('Live', combined)
```

### Edge overlay on original

```python
# Color the edges green and overlay on the original frame
edge_overlay = frame.copy()
edge_overlay[edges > 0] = (0, 255, 0)  # Green where edges exist
cv2.imshow('Live', edge_overlay)
```

### Blended overlay

```python
edges_colored = np.zeros_like(frame)
edges_colored[edges > 0] = (0, 255, 0)
blended = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
cv2.imshow('Live', blended)
```

## The L2gradient Parameter

Canny has an optional parameter `L2gradient` that changes how gradient magnitude is calculated:

```python
edges = cv2.Canny(gray, t1, t2, L2gradient=True)
```

- `False` (default): Uses L1 norm: `|Gx| + |Gy|` — faster
- `True`: Uses L2 norm: `sqrt(Gx² + Gy²)` — more accurate

For real-time applications, the default L1 norm is fine. L2 is slightly more precise but slower.

## Aperture Size

The `apertureSize` parameter controls the Sobel kernel size used internally:

```python
edges = cv2.Canny(gray, t1, t2, apertureSize=3)  # default
edges = cv2.Canny(gray, t1, t2, apertureSize=5)  # larger kernel, smoother gradients
edges = cv2.Canny(gray, t1, t2, apertureSize=7)  # even smoother
```

Larger aperture sizes detect broader edges but are slower. Must be 3, 5, or 7.

## Tips & Common Mistakes

- **Always convert to grayscale** before Canny — passing a color image will cause an error or unexpected results.
- Start with `threshold1=50, threshold2=150` and adjust from there.
- If you see too many noisy edges, increase `threshold1` or add more pre-blur.
- If you see too few edges, decrease `threshold2`.
- Trackbar callback `lambda x: None` is required even if unused — `None` alone causes an error.
- `cv2.getTrackbarPos()` returns an `int`. For parameters that need odd values (like blur kernel size), enforce it: `k = max(1, k | 1)`.
- The combined display (side-by-side) doubles the window width — resize if your screen is small.
- On low-end hardware, adding Gaussian blur before Canny may reduce total FPS. Profile before adding.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Drag the **Thresh1** and **Thresh2** trackbars and observe edges appearing or disappearing in real-time
- Drag the **Blur** trackbar to increase pre-blur and see noisy false edges get suppressed
- Switch the **Mode** trackbar between 0 (side-by-side), 1 (edges only), and 2 (green overlay) to see three different visualization styles

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

# --- Create windows and trackbars ---
cv2.namedWindow('Live Edge Detection')
cv2.createTrackbar('Thresh1', 'Live Edge Detection', 50, 500, lambda x: None)
cv2.createTrackbar('Thresh2', 'Live Edge Detection', 150, 500, lambda x: None)
cv2.createTrackbar('Blur', 'Live Edge Detection', 1, 10, lambda x: None)
cv2.createTrackbar('Mode', 'Live Edge Detection', 0, 2, lambda x: None)
# Mode: 0 = side-by-side, 1 = edges only, 2 = green overlay

frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Read trackbar values ---
        t1 = cv2.getTrackbarPos('Thresh1', 'Live Edge Detection')
        t2 = cv2.getTrackbarPos('Thresh2', 'Live Edge Detection')
        blur_level = cv2.getTrackbarPos('Blur', 'Live Edge Detection')
        mode = cv2.getTrackbarPos('Mode', 'Live Edge Detection')

        # --- Convert to grayscale ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Optional pre-blur (kernel must be odd and >= 1) ---
        if blur_level > 0:
            ksize = blur_level * 2 + 1  # 1->3, 2->5, 3->7, ...
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # --- Canny edge detection ---
        edges = cv2.Canny(gray, t1, t2)

        # --- Display based on mode ---
        if mode == 0:
            # Side-by-side: original + edges
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display = np.hstack([frame, edges_bgr])
        elif mode == 1:
            # Edges only (full screen)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            # Green edge overlay on original
            display = frame.copy()
            display[edges > 0] = (0, 255, 0)

        # --- Draw info overlay ---
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Canny({t1}, {t2}) blur={blur_level}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        modes = ['Side-by-side', 'Edges only', 'Green overlay']
        cv2.putText(display, f"Mode: {modes[mode]}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Live Edge Detection', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
