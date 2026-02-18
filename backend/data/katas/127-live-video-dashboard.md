---
slug: 127-live-video-dashboard
title: Live Video Streaming Dashboard
level: live
concepts: [multiple video views, cv2.resize, layout composition, per-view processing, stats overlay]
prerequisites: [100-live-camera-fps, 104-live-split-screen-filters]
---

## What Problem Are We Solving?

Professional video monitoring and debugging systems rarely show a single camera feed. Instead, they present a **multi-view dashboard** where the same camera input is processed through several pipelines simultaneously and displayed in a grid. A surveillance operator might see the raw feed alongside an edge-detected view, a motion detection overlay, a color segmentation mask, a thresholded binary image, and a histogram visualization -- all updating in real-time, all from the same camera.

This pattern is equally valuable during development. When you are tuning a computer vision pipeline, seeing the intermediate results of every processing stage side by side -- edges, thresholds, color masks, motion -- saves enormous time compared to switching between individual windows or print statements. You see immediately how a parameter change affects every stage.

Building this dashboard also forces you to solve practical engineering problems: resizing heterogeneous outputs to a consistent size, converting single-channel results to 3-channel for stacking, managing per-view processing efficiently so the overall frame rate stays acceptable, and composing a clean layout using NumPy array operations.

## Dashboard Layout: The 3x2 Grid

The dashboard arranges 6 views from a single camera feed in a 3-column, 2-row grid:

```
+-------------------+-------------------+-------------------+
| (1) Original      | (2) Edge          | (3) Color         |
|     with FPS      |     Detection     |     Segmentation  |
+-------------------+-------------------+-------------------+
| (4) Motion        | (5) Histogram     | (6) Thresholded   |
|     Detection     |     Visualization |                   |
+-------------------+-------------------+-------------------+
```

Each cell is the same width and height. With a 640x480 camera, resizing each view to 320x240 produces a 960x480 dashboard -- large enough to see detail but small enough to fit most screens.

## Resizing Views to Consistent Dimensions with cv2.resize

Every view must be exactly the same size before stacking. `cv2.resize` enforces this regardless of the original processing output dimensions:

```python
cell_w, cell_h = 320, 240
view = cv2.resize(source, (cell_w, cell_h))
```

| Parameter | Type | Description |
|---|---|---|
| `src` | `np.ndarray` | Input image (any size) |
| `dsize` | `tuple` | Target size as `(width, height)` -- note the order is (w, h), not (h, w) |
| `interpolation` | `int` | Resampling method. Default: `cv2.INTER_LINEAR` |
| **Returns** | `np.ndarray` | Resized image |

| Interpolation | Speed | Quality | Use Case |
|---|---|---|---|
| `cv2.INTER_NEAREST` | Fastest | Blocky | Binary masks, speed priority |
| `cv2.INTER_LINEAR` | Fast | Good | General-purpose resizing (default) |
| `cv2.INTER_AREA` | Medium | Best for shrinking | Downscaling camera frames |
| `cv2.INTER_CUBIC` | Slow | Best for enlarging | Upscaling small images |

For a dashboard where every frame is downscaled from 640x480 to 320x240, `cv2.INTER_AREA` produces the cleanest results. For speed, `cv2.INTER_LINEAR` is acceptable.

## Building Each View

### View 1: Original with FPS

The raw camera feed with an FPS counter overlay. This serves as the reference for comparing all processed views:

```python
view_original = frame.copy()
cv2.putText(view_original, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
```

### View 2: Edge Detection (Canny)

Canny edge detection highlights boundaries and contours in the scene:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
view_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
```

The `cvtColor` conversion is critical -- `cv2.Canny` returns a single-channel image, but `np.hstack` requires all images to have the same number of channels.

### View 3: Color Segmentation

HSV-based color segmentation isolates objects of a specific color. The mask is displayed as a color overlay on the original frame:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
view_color = cv2.bitwise_and(frame, frame, mask=mask)
```

### View 4: Motion Detection

Frame differencing against the previous frame highlights moving regions:

```python
if prev_gray is not None:
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    view_motion = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
```

### View 5: Histogram Visualization

A live histogram drawn as colored curves on a dark canvas shows the intensity distribution of each BGR channel:

```python
def draw_histogram(frame, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B, G, R
    for ch, color in enumerate(colors):
        hist = cv2.calcHist([frame], [ch], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, height - 10, cv2.NORM_MINMAX)
        pts = []
        for x in range(256):
            px = int(x * width / 256)
            py = height - 5 - int(hist[x])
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    return canvas
```

| cv2.calcHist Parameter | Type | Description |
|---|---|---|
| `images` | `list` | List of source images (wrap in `[]`) |
| `channels` | `list` | Channel index: `[0]` for blue, `[1]` for green, `[2]` for red |
| `mask` | `np.ndarray` or `None` | Optional mask to compute histogram for a subregion |
| `histSize` | `list` | Number of bins, typically `[256]` |
| `ranges` | `list` | Pixel value range, typically `[0, 256]` |

### View 6: Thresholded (Binary)

Global thresholding converts the grayscale image to pure black and white:

```python
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
view_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
```

## Composing the Grid with np.hstack and np.vstack

`np.hstack` joins arrays horizontally (side by side). `np.vstack` joins them vertically (top to bottom). Together they build the grid:

```python
# Resize all views to cell size
views = [cv2.resize(v, (cell_w, cell_h)) for v in all_views]

# Build rows
top_row = np.hstack([views[0], views[1], views[2]])
bottom_row = np.hstack([views[3], views[4], views[5]])

# Stack rows into dashboard
dashboard = np.vstack([top_row, bottom_row])
```

| Function | Constraint | Error If Violated |
|---|---|---|
| `np.hstack` | All arrays must have the **same height** (axis 0) | `ValueError: all dimensions must match` |
| `np.vstack` | All arrays must have the **same width** (axis 1) | `ValueError: all dimensions must match` |
| Both | All arrays must have the **same number of channels** | `ValueError: all dimensions must match` |

## Adding Per-View Labels

Each view needs a label so the user knows what processing is applied. Draw labels with a dark background for readability:

```python
def add_label(view, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(view, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(view, text, (5, th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
```

Call `add_label` on each view **after resizing** but **before stacking**. Drawing labels before resize causes them to be scaled and blurry.

## Per-View Statistics Overlay

In addition to labels, each view can show its own stats -- for example, the number of edge pixels, the motion percentage, or the segmented pixel count:

```python
# On the edge view
edge_count = cv2.countNonZero(edges)
cv2.putText(view_edges, f"Edges: {edge_count}", (5, cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

# On the motion view
motion_pct = 100.0 * cv2.countNonZero(motion_mask) / (cell_w * cell_h)
cv2.putText(view_motion, f"Motion: {motion_pct:.1f}%", (5, cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
```

## Performance: Managing Multiple Processing Pipelines

Running 6 processing pipelines per frame is CPU-intensive. Key optimizations:

1. **Share intermediate results.** Compute `gray` once and reuse it for edges, thresholding, and motion detection. Compute `hsv` once for color segmentation.

2. **Update expensive views less frequently.** The histogram and color segmentation views are visually meaningful even when updated every 2-3 frames:

```python
if frame_count % 2 == 0:
    hist_view = draw_histogram(frame, cell_w, cell_h)
```

3. **Process at reduced resolution.** Run heavy algorithms on a smaller image and resize the result up:

```python
small = cv2.resize(frame, (320, 240))
edges_small = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 50, 150)
view_edges = cv2.resize(cv2.cvtColor(edges_small, cv2.COLOR_GRAY2BGR), (cell_w, cell_h))
```

| Optimization | FPS Impact | Quality Impact |
|---|---|---|
| Share grayscale conversion | +5-10% | None |
| Update histogram every 3 frames | +5% | Slight lag on histogram |
| Process at half resolution | +30-50% | Slightly less detail |
| Use `INTER_NEAREST` for masks | +2-3% | Blocky edges on masks |

## Tips & Common Mistakes

- All views must have exactly the same `(height, width, channels)` before stacking. Even a 1-pixel mismatch causes a `ValueError`. Always use explicit `cv2.resize(view, (cell_w, cell_h))` on every view.
- Single-channel outputs (edges, threshold, motion mask) must be converted to 3-channel BGR with `cv2.cvtColor(..., cv2.COLOR_GRAY2BGR)` before stacking with color views.
- Draw labels and stats **after** resizing each view but **before** stacking. Drawing on the full-resolution frame and then resizing makes text unreadably small.
- The histogram view is not a processed camera image -- it is drawn on a fresh canvas. Make sure the canvas is created at `(cell_w, cell_h, 3)` to match other views.
- If your dashboard is too large for the screen, resize the final dashboard: `dashboard = cv2.resize(dashboard, (target_w, target_h))`. This is cheaper than resizing each view individually.
- Cache views that do not need per-frame updates (histogram, stats). Only recompute them every N frames. This can boost overall FPS by 10-20%.
- Use `cv2.INTER_AREA` when downscaling for the cleanest results. `cv2.INTER_LINEAR` is acceptable for speed but can produce aliasing artifacts.
- The motion detection view requires a previous frame. On the first frame, display a black panel or the raw grayscale until `prev_gray` is available.

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

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {cam_w}x{cam_h}")

# --- Grid cell dimensions ---
CELL_W, CELL_H = 320, 240  # Each view in the 3x2 grid
GRID_COLS, GRID_ROWS = 3, 2

# --- Color segmentation settings (green objects by default) ---
lower_hsv = np.array([35, 80, 80])
upper_hsv = np.array([85, 255, 255])
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- Motion detection state ---
prev_gray = None
motion_thresh_val = 25

# --- FPS tracking ---
frame_times = deque(maxlen=30)
frame_count = 0

# --- Cached views for periodic updates ---
hist_view = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def add_label(view, label, stats_text=None):
    """Add a label and optional stats to a view panel."""
    # Label background
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(view, (0, 0), (tw + 12, th + 10), (0, 0, 0), -1)
    cv2.putText(view, label, (5, th + 5), FONT, 0.45,
                (0, 255, 0), 1, cv2.LINE_AA)
    # Optional stats at bottom
    if stats_text:
        cv2.putText(view, stats_text, (5, CELL_H - 8), FONT, 0.35,
                    (180, 180, 180), 1, cv2.LINE_AA)
    return view

def draw_histogram(frame):
    """Draw per-channel histogram curves on a dark canvas."""
    canvas = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)

    # Draw grid lines
    for frac in [0.25, 0.5, 0.75]:
        y_pos = int(CELL_H * frac)
        cv2.line(canvas, (0, y_pos), (CELL_W, y_pos), (35, 35, 35), 1)
    for frac in [0.25, 0.5, 0.75]:
        x_pos = int(CELL_W * frac)
        cv2.line(canvas, (x_pos, 0), (x_pos, CELL_H), (35, 35, 35), 1)

    # Draw B, G, R channel histograms
    channel_colors = [(255, 80, 0), (0, 200, 0), (0, 0, 255)]
    channel_names = ["B", "G", "R"]

    for ch in range(3):
        hist = cv2.calcHist([frame], [ch], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, CELL_H - 20, cv2.NORM_MINMAX)
        pts = np.array([[int(x * CELL_W / 256),
                         CELL_H - 5 - int(hist[x])]
                        for x in range(256)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, channel_colors[ch], 1, cv2.LINE_AA)

    # Channel legend
    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        cx = CELL_W - 70 + i * 25
        cv2.putText(canvas, name, (cx, CELL_H - 8), FONT, 0.35, color, 1, cv2.LINE_AA)

    return canvas

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

        # --- Shared preprocessing (compute once, reuse everywhere) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ==============================================
        # View 1: Original with FPS
        # ==============================================
        v1 = cv2.resize(frame, (CELL_W, CELL_H))
        add_label(v1, "1. Original", f"FPS: {fps:.1f}")
        cv2.putText(v1, f"{fps:.1f} FPS", (CELL_W - 100, 20),
                    FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # ==============================================
        # View 2: Edge Detection (Canny)
        # ==============================================
        edges = cv2.Canny(gray_blur, 50, 150)
        edge_count = cv2.countNonZero(edges)
        v2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        v2 = cv2.resize(v2, (CELL_W, CELL_H))
        add_label(v2, "2. Edge Detection", f"Edge px: {edge_count}")

        # ==============================================
        # View 3: Color Segmentation (HSV)
        # ==============================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morph_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, morph_kernel)
        color_result = cv2.bitwise_and(frame, frame, mask=color_mask)
        seg_pixels = cv2.countNonZero(color_mask)

        v3 = cv2.resize(color_result, (CELL_W, CELL_H))
        add_label(v3, "3. Color Segment", f"Seg px: {seg_pixels}")

        # ==============================================
        # View 4: Motion Detection
        # ==============================================
        if prev_gray is not None:
            diff = cv2.absdiff(gray_blur, prev_gray)
            _, motion_mask = cv2.threshold(diff, motion_thresh_val, 255,
                                           cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
            motion_pct = 100.0 * cv2.countNonZero(motion_mask) / (cam_w * cam_h)

            # Color-code motion: show motion regions in green on dark background
            v4 = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            v4[:, :, 1] = motion_mask  # Green channel = motion
            v4 = cv2.resize(v4, (CELL_W, CELL_H))
            add_label(v4, "4. Motion Detect", f"Motion: {motion_pct:.1f}%")
        else:
            v4 = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            add_label(v4, "4. Motion Detect", "Initializing...")

        prev_gray = gray_blur.copy()

        # ==============================================
        # View 5: Histogram Visualization (update every 2 frames)
        # ==============================================
        if frame_count % 2 == 0:
            hist_view = draw_histogram(frame)
            add_label(hist_view, "5. Histogram")

        v5 = hist_view.copy()

        # ==============================================
        # View 6: Thresholded (Binary)
        # ==============================================
        _, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        white_pct = 100.0 * cv2.countNonZero(thresh_img) / (cam_w * cam_h)
        v6 = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        v6 = cv2.resize(v6, (CELL_W, CELL_H))
        add_label(v6, "6. Thresholded", f"White: {white_pct:.1f}%")

        # ==============================================
        # Assemble 3x2 Grid
        # ==============================================
        top_row = np.hstack([v1, v2, v3])
        bottom_row = np.hstack([v4, v5, v6])
        dashboard = np.vstack([top_row, bottom_row])

        # --- Global dashboard info ---
        dash_h, dash_w = dashboard.shape[:2]
        cv2.putText(dashboard, f"Dashboard | {cam_w}x{cam_h} | Frame {frame_count}",
                    (10, dash_h - 8), FONT, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(dashboard, "'q'=quit", (dash_w - 80, dash_h - 8),
                    FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Video Dashboard', dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
