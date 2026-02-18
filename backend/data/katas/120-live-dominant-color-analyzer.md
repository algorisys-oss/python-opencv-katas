---
slug: 120-live-dominant-color-analyzer
title: Live Dominant Color Analyzer
level: live
concepts: [K-means clustering, cv2.kmeans, dominant colors, color palette extraction, real-time analysis]
prerequisites: [100-live-camera-fps, 53-color-quantization]
---

## What Problem Are We Solving?

Designers extract color palettes from photographs to inspire website themes, brand identities, and interior design choices. Fashion apps analyze outfits to suggest complementary colors. Retail systems identify the dominant colors of products for search and categorization. All of these tasks reduce a complex image with millions of possible colors down to a handful of representative colors -- the **dominant color palette**.

In a live camera feed, dominant color extraction lets you point the camera at anything -- a painting, a flower, a room, a piece of clothing -- and instantly see the top colors with their percentages. This is K-means clustering applied in real-time: the algorithm groups all pixels into K clusters and reports the cluster centers (representative colors) and their sizes (percentages).

The challenge for real-time operation is speed. K-means on a full 640x480 frame (307,200 pixels) is too slow for smooth video. The solution is **downsampling** -- running K-means on a small version of the frame. Since we are looking for dominant colors (not precise spatial detail), a 64x48 version with 3,072 pixels produces nearly identical color results at a fraction of the cost.

## K-Means Clustering for Color Extraction

K-means groups data points (pixels) into K clusters by iteratively:
1. Assigning each pixel to the nearest cluster center.
2. Recomputing each cluster center as the mean of its assigned pixels.
3. Repeating until convergence (centers stop moving).

For color extraction, each pixel is a 3D point (B, G, R), and the K cluster centers are the K dominant colors.

### Reshaping the Image for cv2.kmeans

`cv2.kmeans` expects a 2D `float32` array with one row per data point:

```python
# Original: (H, W, 3) -> Reshaped: (H*W, 3)
pixels = frame.reshape((-1, 3)).astype(np.float32)
```

| Frame Size | Reshaped Size | Pixels |
|---|---|---|
| `(480, 640, 3)` | `(307200, 3)` | 307,200 |
| `(48, 64, 3)` | `(3072, 3)` | 3,072 |
| `(24, 32, 3)` | `(768, 3)` | 768 |

### The cv2.kmeans API

```python
compactness, labels, centers = cv2.kmeans(
    data, K, bestLabels, criteria, attempts, flags
)
```

| Parameter | Type | Meaning |
|---|---|---|
| `data` | `float32 (N, 3)` | Input pixel data |
| `K` | `int` | Number of clusters (dominant colors) |
| `bestLabels` | `None` or `ndarray` | Initial labels; pass `None` to let OpenCV allocate |
| `criteria` | `tuple` | Stopping criteria: `(type, max_iter, epsilon)` |
| `attempts` | `int` | Number of runs with different random starts; best result kept |
| `flags` | `int` | Initialization method |

| Return Value | Type | Meaning |
|---|---|---|
| `compactness` | `float` | Sum of squared distances from each point to its center (lower = tighter clusters) |
| `labels` | `int32 (N, 1)` | Cluster assignment for each pixel |
| `centers` | `float32 (K, 3)` | The K cluster center colors (BGR) |

### Criteria Parameter

```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
```

| Criteria Type | Meaning |
|---|---|
| `cv2.TERM_CRITERIA_EPS` | Stop when centers move less than `epsilon` |
| `cv2.TERM_CRITERIA_MAX_ITER` | Stop after `max_iter` iterations |
| Combined with `+` | Stop when either condition is met |

For real-time use, fewer iterations and a larger epsilon trade accuracy for speed:

| Use Case | `max_iter` | `epsilon` | Attempts | Speed |
|---|---|---|---|---|
| High quality (offline) | 100 | 0.1 | 10 | Slow |
| Balanced | 20 | 1.0 | 5 | Medium |
| Real-time (live video) | 10 | 2.0 | 2 | Fast |

### Initialization Flags

| Flag | Meaning |
|---|---|
| `cv2.KMEANS_RANDOM_CENTERS` | Random initial centers (most common) |
| `cv2.KMEANS_PP_CENTERS` | K-means++ initialization (smarter, slightly slower, often better) |
| `cv2.KMEANS_USE_INITIAL_LABELS` | Use the `bestLabels` parameter as starting point |

For real-time, `cv2.KMEANS_PP_CENTERS` with fewer attempts often beats `cv2.KMEANS_RANDOM_CENTERS` with many attempts -- better initialization compensates for fewer retries.

## Speed Optimization: Downsampling

Running K-means on every pixel of a 640x480 frame is expensive. Downsampling first makes it tractable:

```python
# Downsample to 64x48 (100x reduction in pixel count)
small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
pixels = small.reshape((-1, 3)).astype(np.float32)
```

Why this works: dominant colors are a **global property** of the image. A 64x48 thumbnail contains the same general color distribution as the full-resolution frame. The K-means result is nearly identical.

| Downsample Size | Pixels | Relative Speed | Color Accuracy |
|---|---|---|---|
| `(640, 480)` | 307,200 | 1x (baseline) | Perfect |
| `(160, 120)` | 19,200 | ~15x faster | Excellent |
| `(64, 48)` | 3,072 | ~100x faster | Very good |
| `(32, 24)` | 768 | ~400x faster | Good (may miss subtle colors) |

Use `cv2.INTER_AREA` for downsampling -- it is the correct interpolation mode for shrinking images, as it averages pixels rather than aliasing them.

## Computing Percentages

After K-means, `labels` tells you which cluster each pixel belongs to. Count pixels per cluster to get percentages:

```python
_, counts = np.unique(labels, return_counts=True)
percentages = counts / counts.sum() * 100

# Sort by percentage (largest first)
sorted_idx = np.argsort(-percentages)
sorted_colors = centers[sorted_idx].astype(np.uint8)
sorted_pcts = percentages[sorted_idx]
```

## Drawing the Color Palette Bar

Display the dominant colors as a horizontal bar where each color's width is proportional to its percentage:

```python
def draw_palette_bar(colors, percentages, width=400, height=50):
    """Draw a horizontal color palette bar."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    x_start = 0
    for color, pct in zip(colors, percentages):
        x_end = x_start + int(width * pct / 100)
        cv2.rectangle(bar, (x_start, 0), (x_end, height), color.tolist(), -1)
        # Label with percentage
        if x_end - x_start > 30:
            cx = (x_start + x_end) // 2
            cv2.putText(bar, f"{pct:.0f}%", (cx - 12, height // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        x_start = x_end
    return bar
```

## Approximating Color Names

Map BGR values to human-readable names for common colors. A simple approach uses distance in HSV space:

```python
COLOR_NAMES = {
    "Red":     (0, 0, 255),
    "Orange":  (0, 128, 255),
    "Yellow":  (0, 255, 255),
    "Green":   (0, 200, 0),
    "Cyan":    (255, 255, 0),
    "Blue":    (255, 0, 0),
    "Purple":  (128, 0, 128),
    "Pink":    (147, 20, 255),
    "White":   (255, 255, 255),
    "Gray":    (128, 128, 128),
    "Black":   (0, 0, 0),
    "Brown":   (19, 69, 139),
}

def closest_color_name(bgr):
    """Find the closest named color to the given BGR value."""
    min_dist = float('inf')
    best_name = "Unknown"
    for name, ref_bgr in COLOR_NAMES.items():
        dist = sum((int(a) - int(b)) ** 2 for a, b in zip(bgr, ref_bgr))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name
```

This is a rough approximation. For more accuracy, convert both colors to LAB space before computing distance, since LAB is perceptually uniform (equal distances correspond to equal perceived differences).

## Tips & Common Mistakes

- Always convert pixel data to `float32` before passing to `cv2.kmeans`. Passing `uint8` data causes an error or incorrect results.
- Downsample the frame before K-means for real-time performance. A 64x48 thumbnail is sufficient for dominant color extraction and runs 100x faster than full resolution.
- `labels` from `cv2.kmeans` has shape `(N, 1)`, not `(N,)`. Use `.flatten()` before passing to `np.unique` or using as an index.
- Convert `centers` back to `uint8` for display: `centers.astype(np.uint8)`. They are returned as `float32`.
- K=5 is a good default for color palettes. K=3 is too few for varied scenes; K=8+ produces many similar-looking colors.
- The `attempts` parameter runs K-means multiple times with different random starts. For real-time, 2 attempts is a good balance between speed and quality.
- `cv2.KMEANS_PP_CENTERS` (K-means++ initialization) produces better results than `cv2.KMEANS_RANDOM_CENTERS` with fewer attempts, making it ideal for real-time where you want to minimize `attempts`.
- Sort colors by percentage descending so the most dominant color is always first in the display.
- The color name approximation using Euclidean distance in BGR space is imprecise. Brown and dark red look similar in BGR distance. For better results, convert to HSV or LAB before computing distances.
- If the palette flickers between frames, consider averaging the palette over the last few frames or only updating every N frames.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Point the camera at a colorful object (book cover, clothing, fruit) — you should see a color palette bar at the bottom and labeled swatches on the right showing dominant colors with percentages
- Press **+** or **-** to increase or decrease the number of colors (K) and verify the palette updates accordingly
- Check the FPS counter in the top-left corner to confirm real-time performance

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

# --- K-means parameters ---
K = 5  # Number of dominant colors
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2.0)
attempts = 2
flags = cv2.KMEANS_PP_CENTERS

# --- Color name lookup ---
COLOR_NAMES = {
    "Red":     (0, 0, 255),
    "Orange":  (0, 128, 255),
    "Yellow":  (0, 255, 255),
    "Green":   (0, 200, 0),
    "Cyan":    (255, 255, 0),
    "Blue":    (255, 0, 0),
    "Purple":  (128, 0, 128),
    "Pink":    (147, 20, 255),
    "White":   (255, 255, 255),
    "Gray":    (128, 128, 128),
    "Black":   (0, 0, 0),
    "Brown":   (19, 69, 139),
}


def closest_color_name(bgr):
    """Find the closest named color by Euclidean distance in BGR space."""
    min_dist = float('inf')
    best_name = "Unknown"
    for name, ref in COLOR_NAMES.items():
        dist = sum((int(a) - int(b)) ** 2 for a, b in zip(bgr, ref))
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


def draw_palette(colors, percentages, width, height=50):
    """Draw a horizontal color palette bar with percentages."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    x_start = 0
    for color, pct in zip(colors, percentages):
        x_end = x_start + int(width * pct / 100)
        if x_end <= x_start:
            x_end = x_start + 1
        bgr = tuple(int(c) for c in color)
        cv2.rectangle(bar, (x_start, 0), (x_end, height), bgr, -1)
        # Label if segment is wide enough
        seg_w = x_end - x_start
        if seg_w > 35:
            # Choose text color for contrast
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cx = (x_start + x_end) // 2
            cv2.putText(bar, f"{pct:.0f}%", (cx - 14, height // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
        x_start = x_end
    return bar


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]

        # --- Downsample for speed ---
        small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
        pixels = small.reshape((-1, 3)).astype(np.float32)

        # --- Run K-means ---
        compactness, labels, centers = cv2.kmeans(
            pixels, K, None, criteria, attempts, flags
        )

        # --- Compute percentages and sort ---
        _, counts = np.unique(labels, return_counts=True)
        percentages = counts / counts.sum() * 100.0
        sorted_idx = np.argsort(-percentages)
        sorted_colors = centers[sorted_idx].astype(np.uint8)
        sorted_pcts = percentages[sorted_idx]

        # --- Draw palette bar at the bottom of the frame ---
        palette_bar = draw_palette(sorted_colors, sorted_pcts, w, height=40)
        frame[-40:, :] = palette_bar

        # --- Draw color swatches with names on the right side ---
        swatch_size = 30
        swatch_x = w - 160
        swatch_y_start = 70
        # Dark background for swatch panel
        panel_h = K * (swatch_size + 8) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (swatch_x - 10, swatch_y_start - 10),
                      (w - 5, swatch_y_start + panel_h),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i in range(K):
            y = swatch_y_start + i * (swatch_size + 8)
            color_bgr = tuple(int(c) for c in sorted_colors[i])
            pct = sorted_pcts[i]
            name = closest_color_name(sorted_colors[i])

            # Color swatch
            cv2.rectangle(frame, (swatch_x, y), (swatch_x + swatch_size, y + swatch_size),
                          color_bgr, -1)
            cv2.rectangle(frame, (swatch_x, y), (swatch_x + swatch_size, y + swatch_size),
                          (200, 200, 200), 1)

            # Color name and percentage
            cv2.putText(frame, f"{name}", (swatch_x + swatch_size + 8, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{pct:.1f}%", (swatch_x + swatch_size + 8, y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Dominant Colors (K={K})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit | +/- to change K", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Dominant Color Analyzer', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            K = min(K + 1, 10)
            print(f"K = {K}")
        elif key == ord('-') or key == ord('_'):
            K = max(K - 1, 2)
            print(f"K = {K}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
