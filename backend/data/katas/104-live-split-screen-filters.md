---
slug: 104-live-split-screen-filters
title: Live Split-Screen Filters
level: live
concepts: [np.hstack, np.vstack, cv2.cvtColor, cv2.GaussianBlur, cv2.Canny, real-time multi-view]
prerequisites: [100-live-camera-fps, 21-gaussian-blur, 32-canny-edge-detection]
---

## What Problem Are We Solving?

When developing a computer vision pipeline, you often need to **compare** multiple processing stages side by side — the original feed, a blurred version, edge detection, a grayscale view. Switching between windows or running separate scripts is slow and awkward. A **split-screen display** shows all views simultaneously in a single window, making it easy to see how each filter transforms the same frame in real-time.

This pattern — a 2x2 or NxM grid of processed views — is one of the most useful debugging and development tools in live CV work.

## Building a Multi-View Grid

The core technique is `np.hstack` (horizontal stack) and `np.vstack` (vertical stack) to combine images:

```python
# 2x2 grid from 4 images of the same size
top_row = np.hstack([img1, img2])
bottom_row = np.hstack([img3, img4])
grid = np.vstack([top_row, bottom_row])
```

### All Images Must Match

For `hstack`, all images must have the **same height and number of channels**. For `vstack`, same **width and channels**.

**Common trap:** Grayscale images are 2D `(h, w)` while color images are 3D `(h, w, 3)`. You must convert grayscale to 3-channel before stacking:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Now (h, w, 3)
# Can now hstack with the original color frame
```

Similarly for edge images (Canny output is single-channel):

```python
edges = cv2.Canny(gray, 50, 150)
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Ready for stacking
```

## Adding Labels to Each View

Without labels, you'll forget which quadrant shows what. Add text to each view before stacking:

```python
def add_label(img, text, bg_color=(0, 0, 0)):
    """Add a label bar at the top of an image."""
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (len(text) * 12 + 10, 28), bg_color, -1)
    cv2.putText(labeled, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled
```

## Resizing for Smaller Grids

A 2x2 grid of 640x480 frames creates a 1280x960 window — too large for most screens. Resize each view before stacking:

```python
# Resize each quadrant to half the original
h, w = frame.shape[:2]
small_w, small_h = w // 2, h // 2

view1 = cv2.resize(frame, (small_w, small_h))
view2 = cv2.resize(blurred_bgr, (small_w, small_h))
view3 = cv2.resize(edges_bgr, (small_w, small_h))
view4 = cv2.resize(thresh_bgr, (small_w, small_h))
```

## Filter Ideas for the Grid

Here are common filters to display side by side:

| View | Code | Purpose |
|---|---|---|
| Original | `frame` | Reference baseline |
| Grayscale | `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | Luminance only |
| Gaussian Blur | `cv2.GaussianBlur(frame, (15, 15), 0)` | Noise reduction |
| Canny Edges | `cv2.Canny(gray, 50, 150)` | Edge structure |
| Threshold | `cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` | Binary segmentation |
| HSV Hue | `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,0]` | Color distribution |
| Laplacian | `cv2.Laplacian(gray, cv2.CV_8U)` | Second-derivative edges |
| Bilateral | `cv2.bilateralFilter(frame, 9, 75, 75)` | Edge-preserving smooth |

## Dynamic Grid: Cycling Through Filters

You can cycle through different filter combinations using a keypress:

```python
filter_set = 0  # Toggle with a key

if filter_set == 0:
    views = [original, gray, blur, edges]
elif filter_set == 1:
    views = [original, threshold, laplacian, bilateral]

# In the key handling:
key = cv2.waitKey(1) & 0xFF
if key == ord('n'):
    filter_set = (filter_set + 1) % 2
```

## Performance Considerations

Each filter adds processing time per frame:

| Filter | Typical Time (640x480) |
|---|---|
| `cvtColor` (grayscale) | <1 ms |
| `GaussianBlur` (15x15) | ~2 ms |
| `Canny` | ~3 ms |
| `bilateralFilter` | ~15-30 ms |
| `threshold` | <1 ms |
| `Laplacian` | ~2 ms |

If your FPS drops too low, remove the heaviest filter (`bilateralFilter`) or process a downscaled frame.

## Tips & Common Mistakes

- **Always convert single-channel to 3-channel** before `hstack`/`vstack` with color images. Mismatched channel counts cause a cryptic error.
- All images in a row must have the **same height**; all images in a column must have the **same width**. Use `cv2.resize` to enforce.
- Labels should be added **after** resizing — otherwise the text gets scaled and looks blurry.
- For a 3x3 or larger grid, consider a helper function that takes a list of `(label, image)` tuples and builds the grid automatically.
- `np.hstack` and `np.vstack` create a **new** array — they allocate memory each frame. For maximum performance, pre-allocate a canvas and copy views into it using slicing.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Verify you see a 2x2 grid with labeled quadrants: Original, Grayscale, Gaussian Blur, and Canny Edges
- Press **'n'** to switch to the second filter set (Threshold, Laplacian, HSV Hue Map) and press **'n'** again to cycle back
- Check the FPS counter at the bottom-left to confirm smooth real-time performance

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
filter_set = 0  # 0 = basic filters, 1 = edge filters

def add_label(img, text):
    """Add a text label bar at the top of an image."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 10, 25), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]
        small_w, small_h = w // 2, h // 2

        # --- Compute all filtered views ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        blur = cv2.GaussianBlur(frame, (15, 15), 0)

        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        hue_colored = cv2.applyColorMap(hue, cv2.COLORMAP_HSV)

        # --- Select filter set ---
        if filter_set == 0:
            views = [
                ("Original", frame),
                ("Grayscale", gray_bgr),
                ("Gaussian Blur", blur),
                ("Canny Edges", edges_bgr),
            ]
        else:
            views = [
                ("Original", frame),
                ("Threshold", thresh_bgr),
                ("Laplacian", laplacian_bgr),
                ("HSV Hue Map", hue_colored),
            ]

        # --- Resize and label each view ---
        labeled = []
        for label, img in views:
            small = cv2.resize(img, (small_w, small_h))
            labeled.append(add_label(small, label))

        # --- Build 2x2 grid ---
        top_row = np.hstack([labeled[0], labeled[1]])
        bot_row = np.hstack([labeled[2], labeled[3]])
        grid = np.vstack([top_row, bot_row])

        # --- FPS and instructions overlay ---
        cv2.putText(grid, f"FPS: {fps:.1f}", (10, grid.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        set_name = "Basic" if filter_set == 0 else "Edge"
        cv2.putText(grid, f"Set: {set_name} | 'n'=next | 'q'=quit",
                    (small_w - 50, grid.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Split-Screen Filters', grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            filter_set = (filter_set + 1) % 2

finally:
    cap.release()
    cv2.destroyAllWindows()
```
