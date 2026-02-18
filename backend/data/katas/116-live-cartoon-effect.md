---
slug: 116-live-cartoon-effect
title: "Live Cartoon Effect"
level: live
concepts: [cv2.bilateralFilter, cv2.adaptiveThreshold, edge overlay, cartoon rendering pipeline]
prerequisites: [100-live-camera-fps, 23-bilateral-filter, 27-adaptive-thresholding]
---

## What Problem Are We Solving?

Turning a live camera feed into a cartoon or comic book is one of the most visually satisfying real-time effects you can build. Apps like Snapchat, Instagram, and TikTok offer cartoon filters that make you look like an animated character. Professional tools use similar pipelines for rotoscoping and non-photorealistic rendering in film production.

The cartoon look has two defining characteristics: **smooth, flat color regions** (like painted areas in a comic panel) and **bold dark outlines** around objects and features (like the ink lines a cartoonist draws). These two components map directly to two OpenCV operations -- bilateral filtering for color smoothing and adaptive thresholding for edge extraction.

Understanding why this specific combination works -- and why other combinations do not produce as good a result -- requires knowing what makes bilateral filtering different from Gaussian blur, and why adaptive thresholding produces better cartoon outlines than Canny edge detection or simple global thresholding.

## The Cartoon Rendering Pipeline

The complete pipeline processes each frame in four stages:

```
Camera Frame
    |
    v
[1] Bilateral Filter (x N iterations) --> Smooth, flat color regions
    |
    v
[2] Grayscale + Median Blur --> Noise-free grayscale for edge detection
    |
    v
[3] Adaptive Threshold --> Bold black/white edge mask
    |
    v
[4] Bitwise AND (smooth color + edge mask) --> Final cartoon frame
```

Each stage has tunable parameters that control the style -- from subtle illustration to heavy comic book ink.

## Step 1: Bilateral Filtering for Color Smoothing

Most blur filters (Gaussian, box, median) smooth everything uniformly, including edges. This makes the image look out-of-focus rather than cartoony. **Bilateral filtering** is fundamentally different: it smooths flat regions while **preserving sharp edges**.

```python
smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
```

### How Bilateral Filtering Works

For each pixel, the filter considers nearby pixels but weights them by **two independent factors**:

1. **Spatial proximity** (like Gaussian blur) -- nearby pixels get more weight than distant ones.
2. **Color similarity** -- pixels with similar color values get more weight; pixels across a color edge get much less weight.

The result: within a uniform region (e.g., a cheek), nearby pixels are averaged together (producing smooth flat color). Across an edge (e.g., cheek-to-background boundary), the averaging does not mix colors from both sides (preserving the edge).

### bilateralFilter Parameters

| Parameter | Type | Description |
|---|---|---|
| `src` | ndarray | Input image (BGR color or grayscale) |
| `d` | int | Diameter of the pixel neighborhood. `-1` to auto-compute from `sigmaSpace`. Use 5-9 for real-time |
| `sigmaColor` | float | Filter sigma in the color space. Larger values mean more distant colors get blended together. 50-100 is typical |
| `sigmaSpace` | float | Filter sigma in the coordinate space. Larger values mean spatially distant pixels have more influence. 50-100 is typical |

### The Effect of Multiple Iterations

A single bilateral filter pass produces mild smoothing. For the characteristic flat-color cartoon look, apply it multiple times:

```python
smooth = frame.copy()
for _ in range(num_iterations):
    smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)
```

| Iterations | Visual Effect | Typical Time (640x480) |
|---|---|---|
| 1 | Subtle smoothing, still looks mostly photographic | ~15 ms |
| 2 | Noticeable flat regions forming, edges still sharp | ~30 ms |
| 3 | Strong cartoon look with clearly flat color regions | ~45 ms |
| 4 | Very flat, approaching watercolor / painterly style | ~60 ms |
| 5+ | Over-smoothed, fine details lost, diminishing returns | ~75+ ms |

**Performance warning:** The bilateral filter is the most expensive standard OpenCV filter by a large margin. It is the dominant bottleneck in this pipeline. For real-time use at reasonable FPS, 2-3 iterations at `d=9` is the practical maximum on most hardware.

### sigmaColor Effect on Smoothing Style

| sigmaColor | Visual Effect |
|---|---|
| 25-40 | Mild smoothing -- textures partially retained, subtle illustration look |
| 50-75 | Moderate smoothing -- good default, clearly cartoonish |
| 100-150 | Strong smoothing -- very flat color regions, almost poster-like |
| 200+ | Extreme smoothing -- large color regions merge together |

## Step 2: Edge Extraction with Adaptive Thresholding

The bold black outlines come from edge detection applied to a grayscale version of the frame. While Canny produces clean single-pixel edges, **adaptive thresholding** on a median-blurred grayscale produces thicker, more ink-like outlines that look hand-drawn:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)

edges = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=9,
    C=2
)
```

### Why Adaptive Thresholding Instead of Canny?

| Method | Edge Style | Why It Does / Does Not Work for Cartoons |
|---|---|---|
| Canny | Thin, precise, single-pixel edges | Too thin and detailed -- looks like a technical diagram, not a cartoon |
| Simple threshold | Uniform global threshold | Ignores local contrast variations, misses edges in dark or bright regions |
| Adaptive threshold | Block-based, locally adaptive, produces thick filled outlines | Adapts to local brightness, creates the bold hand-drawn ink look |

### adaptiveThreshold Parameters

| Parameter | Type | Description |
|---|---|---|
| `src` | ndarray | Input grayscale image (single channel `uint8`) |
| `maxValue` | float | Value assigned to pixels that pass the threshold (255 for white) |
| `adaptiveMethod` | int | `ADAPTIVE_THRESH_MEAN_C` (mean of local block) or `ADAPTIVE_THRESH_GAUSSIAN_C` (Gaussian-weighted mean) |
| `thresholdType` | int | `THRESH_BINARY` -- pixels above the local threshold become maxValue, below become 0 |
| `blockSize` | int | Size of the local neighborhood block (must be odd, >= 3). Larger = thicker, fewer outlines |
| `C` | float | Constant subtracted from the computed local mean. Larger C = fewer edges detected |

### blockSize Controls Edge Thickness

| blockSize | Effect |
|---|---|
| 5 | Many thin edges -- detailed but potentially noisy |
| 7 | Moderate edges -- good balance for most faces |
| 9 | Thicker edges -- standard cartoon/comic look |
| 11 | Bold edges -- heavy ink style |
| 15+ | Very thick, sparse edges -- abstract sketch style |

### C Controls Edge Sensitivity

| C Value | Effect |
|---|---|
| 0 | Maximum sensitivity -- every subtle gradient becomes an edge |
| 2 | Standard sensitivity -- good default |
| 5 | Reduced sensitivity -- only strong contrast boundaries |
| 10+ | Very few edges -- only the most dramatic boundaries survive |

### Why Median Blur Before Thresholding?

Without pre-blurring, adaptive thresholding picks up texture noise -- fabric patterns, skin pores, hair strands, wood grain -- as outlines, making the cartoon look messy and overdetailed. `cv2.medianBlur` removes this fine texture while preserving the structural edges we actually want:

```python
gray = cv2.medianBlur(gray, 7)
```

| Median Kernel Size | Effect |
|---|---|
| 3 | Minimal noise removal -- texture outlines still visible |
| 5 | Good balance -- removes fine texture, keeps object edges |
| 7 | Strong smoothing -- only major structural edges survive |
| 9+ | Too much smoothing -- thin features like glasses frames may disappear |

## Step 3: Combining Smooth Color with Edge Outlines

The final cartoon image combines the bilateral-smoothed color with the edge mask. The adaptive threshold output uses `THRESH_BINARY`, which means:
- **255 (white)** = flat region (no edge)
- **0 (black)** = edge

A `cv2.bitwise_and` merges them:

```python
edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cartoon = cv2.bitwise_and(smooth, edges_3ch)
```

### Why This Works

`bitwise_and` with 255 (all bits set) keeps the pixel unchanged. `bitwise_and` with 0 (no bits set) forces the pixel to black. So:
- **Flat regions** (edge mask = 255): the smoothed color pixel passes through unchanged
- **Edge regions** (edge mask = 0): the pixel becomes pure black, creating the bold ink outline

This is exactly the cartoon look: flat painted colors surrounded by dark outlines.

## Tuning for Different Cartoon Styles

| Style | bilateral d | sigmaColor | Iterations | medianBlur | blockSize | C | Look |
|---|---|---|---|---|---|---|---|
| Light illustration | 5 | 50 | 1 | 5 | 9 | 3 | Subtle, semi-realistic |
| Standard cartoon | 9 | 75 | 2 | 7 | 9 | 2 | Classic comic book |
| Bold comic | 9 | 100 | 3 | 7 | 7 | 1 | Heavy ink, flat colors |
| Painterly / watercolor | 9 | 150 | 4 | 5 | 5 | 0 | Very flat, many fine edges |
| Minimal sketch | 5 | 50 | 1 | 9 | 13 | 5 | Few bold lines, subtle color |

## Performance Optimization

The bilateral filter dominates processing time. Two practical strategies to maintain acceptable FPS:

### Strategy 1: Downscale Before Filtering

Process the bilateral filter at half resolution, then upscale. Edge detection still runs at full resolution for crisp outlines:

```python
h, w = frame.shape[:2]
small = cv2.resize(frame, (w // 2, h // 2))
smooth = small.copy()
for _ in range(num_bilateral):
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
smooth = cv2.resize(smooth, (w, h))
```

This reduces bilateral filter time by approximately 75% (quarter the pixels).

### Strategy 2: Reduce Filter Diameter

Using `d=5` instead of `d=9` is significantly faster with only a modest quality reduction:

| d Value | Relative Speed | Quality |
|---|---|---|
| 5 | ~2.5x faster | Slightly less smooth, still good |
| 7 | ~1.5x faster | Good compromise |
| 9 | Baseline | Best smoothing quality |

### Timing Breakdown (640x480)

| Operation | Typical Time |
|---|---|
| `bilateralFilter` (d=9, 1 pass) | 12-18 ms |
| `cvtColor` to grayscale | < 1 ms |
| `medianBlur` | < 1 ms |
| `adaptiveThreshold` | < 1 ms |
| `cvtColor` gray to BGR | < 1 ms |
| `bitwise_and` | < 1 ms |

## Tips & Common Mistakes

- **Bilateral filter is slow.** It is the number one bottleneck. If FPS drops below 15, reduce `d` to 5 or reduce iterations from 3 to 2.
- The `blockSize` for adaptive thresholding **must be odd** and at least 3. Even values cause an OpenCV error with no useful error message.
- **Always apply median blur before adaptive thresholding.** Without it, fabric texture, skin pores, and background grain all become outlines, making the cartoon look cluttered.
- The edge image from `adaptiveThreshold` with `THRESH_BINARY` has white = flat and black = edge. This is the correct orientation for `bitwise_and`. Using `THRESH_BINARY_INV` reverses the logic and produces white outlines on dark color.
- `ADAPTIVE_THRESH_GAUSSIAN_C` produces slightly smoother edge detection than `ADAPTIVE_THRESH_MEAN_C` but is marginally slower.
- Converting the single-channel edge mask to 3-channel with `cv2.cvtColor` before `bitwise_and` is mandatory -- a channel count mismatch causes a crash.
- If the cartoon looks too dark overall, increase `C` in the adaptive threshold to reduce the total number of edge pixels.
- Use the original (unfiltered) frame for edge detection, not the bilateral-filtered version. Bilateral filtering already removes the edges you want to detect.
- Adding keyboard controls for edge thickness and smoothing strength lets you tune the look interactively rather than restarting the program for each parameter change.

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

# --- Tunable parameters ---
num_bilateral = 2       # Bilateral filter iterations (1-6)
edge_block_size = 9     # Adaptive threshold block size (odd, 5-15)
median_k = 7            # Median blur kernel size (odd, 3-9)
cartoon_on = True       # Toggle effect on/off

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        if cartoon_on:
            # --- Step 1: Bilateral filter for smooth color regions ---
            smooth = frame.copy()
            for _ in range(num_bilateral):
                smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

            # --- Step 2: Edge mask from adaptive threshold ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, median_k)
            edges = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=edge_block_size,
                C=2
            )

            # --- Step 3: Combine smooth color + edge outlines ---
            edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display = cv2.bitwise_and(smooth, edges_3ch)
        else:
            display = frame.copy()

        # --- HUD overlay ---
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        status = "ON" if cartoon_on else "OFF"
        cv2.putText(display, f"Cartoon: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Bilateral: {num_bilateral}x | Edges: {edge_block_size} | Median: {median_k}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(display, "'c'=toggle  'e/d'=edges  's/a'=smooth  'q'=quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Cartoon Effect', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cartoon_on = not cartoon_on
        elif key == ord('e'):
            # Thicker edges (smaller block size)
            edge_block_size = max(5, edge_block_size - 2)
        elif key == ord('d'):
            # Thinner edges (larger block size)
            edge_block_size = min(15, edge_block_size + 2)
        elif key == ord('s'):
            # More smoothing
            num_bilateral = min(num_bilateral + 1, 6)
        elif key == ord('a'):
            # Less smoothing
            num_bilateral = max(num_bilateral - 1, 1)

finally:
    cap.release()
    cv2.destroyAllWindows()
```
