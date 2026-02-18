---
slug: 32-canny-edge-detection
title: Canny Edge Detection
level: intermediate
concepts: [cv2.Canny, double threshold, non-maximum suppression]
prerequisites: [29-sobel-edge-detection, 21-gaussian-blur]
---

## What Problem Are We Solving?

Simple gradient-based edge detectors (Sobel, Laplacian) produce **thick, noisy edges** with many false positives. The **Canny edge detector** is a multi-stage algorithm that produces **thin, clean, well-connected edges** — it's the gold standard for edge detection in computer vision.

## The Canny Pipeline

The Canny algorithm has four stages:

1. **Gaussian Blur** — smooth the image to reduce noise
2. **Gradient Computation** — find edge strength and direction (using Sobel internally)
3. **Non-Maximum Suppression (NMS)** — thin edges to 1-pixel width by keeping only the local maximum along the gradient direction
4. **Double Threshold + Hysteresis** — classify pixels as strong edge, weak edge, or non-edge. Weak edges are kept only if they connect to a strong edge.

## Using cv2.Canny()

```python
edges = cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
```

| Parameter | Meaning |
|---|---|
| `image` | 8-bit input image (grayscale or color) |
| `threshold1` | Lower threshold for hysteresis |
| `threshold2` | Upper threshold for hysteresis |
| `apertureSize` | Sobel kernel size (3, 5, or 7) |
| `L2gradient` | If `True`, uses L2 norm for gradient magnitude (more accurate but slower) |

The function returns a **binary image** — pixels are either 0 (not an edge) or 255 (edge).

## Understanding the Double Threshold

The two thresholds control edge sensitivity:

- Pixels with gradient **above `threshold2`** are **strong edges** (definitely edges)
- Pixels with gradient **below `threshold1`** are **suppressed** (definitely not edges)
- Pixels **between the two thresholds** are **weak edges** — kept only if they connect to a strong edge

```python
# Low thresholds: more edges (including noise)
edges_sensitive = cv2.Canny(gray, 30, 80)

# High thresholds: fewer, stronger edges only
edges_strict = cv2.Canny(gray, 100, 200)
```

> **Rule of thumb:** A common ratio is `threshold2 = 2 * threshold1` or `threshold2 = 3 * threshold1`. Start with `threshold1=50, threshold2=150` and adjust.

## Auto-Canny Technique

Instead of guessing thresholds, compute them from the image statistics:

```python
def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)
```

This uses the median pixel value to set thresholds that adapt to the image's overall brightness. The `sigma` parameter controls sensitivity — smaller sigma means tighter thresholds (fewer edges).

## Pre-blurring for Better Results

Although Canny applies its own internal blur, you often get better results by pre-blurring:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
```

More blur means fewer fine details and noise, producing cleaner edges for the main structures.

## Tips & Common Mistakes

- **Always blur first** for noisy images. Canny's built-in blur may not be enough.
- The input should be **8-bit** (`uint8`). Canny does not accept float images.
- If you pass a color image, Canny converts it to grayscale internally. For more control, convert to grayscale yourself first.
- **Low thresholds** = more edges (noisy). **High thresholds** = fewer edges (may miss weak edges). When in doubt, use the auto-Canny technique.
- The output is a **binary image** (0 or 255), not a gradient magnitude. This makes it directly usable for contour detection.
- `L2gradient=True` gives slightly more accurate gradient magnitudes at the cost of speed. For most applications, the default `False` (L1 norm) is fine.
- Canny is designed for **grayscale** images. If you need color edge detection, run Canny on each channel separately or convert to grayscale.

## Starter Code

```python
import cv2
import numpy as np

# Create a test scene with shapes and textures
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (60, 60, 60)

# Geometric shapes
cv2.rectangle(img, (30, 30), (180, 180), (200, 200, 200), -1)
cv2.circle(img, (300, 110), 80, (220, 220, 220), -1)
cv2.ellipse(img, (500, 110), (70, 50), 0, 0, 360, (180, 180, 180), -1)

# Triangle
pts = np.array([[100, 380], [250, 230], [400, 380]], np.int32)
cv2.fillPoly(img, [pts], (190, 190, 190))

# Small details (text)
cv2.putText(img, 'Canny', (420, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

# Add some noise to make it realistic
noise = np.random.normal(0, 8, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Auto-Canny function ---
def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)

# --- Different threshold settings ---
edges_low = cv2.Canny(blurred, 30, 80)       # Sensitive (more edges)
edges_mid = cv2.Canny(blurred, 50, 150)      # Balanced
edges_high = cv2.Canny(blurred, 100, 200)    # Strict (fewer edges)
edges_auto = auto_canny(blurred)              # Adaptive

# No pre-blur for comparison
edges_no_blur = cv2.Canny(gray, 50, 150)

# --- Build comparison display ---
def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
panels = [
    (img.copy(), 'Original (noisy)'),
    (to_bgr(edges_no_blur), 'No blur (50/150)'),
    (to_bgr(edges_low), 'Low (30/80)'),
    (to_bgr(edges_mid), 'Mid (50/150)'),
    (to_bgr(edges_high), 'High (100/200)'),
    (to_bgr(edges_auto), f'Auto-Canny'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

top_row = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
bottom_row = np.hstack([panels[3][0], panels[4][0], panels[5][0]])
result = np.vstack([top_row, bottom_row])

median_val = np.median(blurred)
print(f'Image median: {median_val:.0f}')
print(f'Auto-Canny thresholds: {int(max(0, 0.67 * median_val))} - {int(min(255, 1.33 * median_val))}')
print(f'Edge pixels (low/mid/high/auto): {np.count_nonzero(edges_low)}, {np.count_nonzero(edges_mid)}, {np.count_nonzero(edges_high)}, {np.count_nonzero(edges_auto)}')

cv2.imshow('Canny Edge Detection', result)
```
