---
slug: 49-distance-transform
title: Distance Transform
level: intermediate
concepts: [cv2.distanceTransform, DIST_L2, skeleton]
prerequisites: [26-simple-thresholding]
---

## What Problem Are We Solving?

Given a binary image, you often need to know **how far each white pixel is from the nearest black pixel** (or vice versa). The **distance transform** computes this for every pixel simultaneously, producing a grayscale image where brighter pixels are farther from the nearest boundary. This is essential for separating touching objects (the centers of objects have the highest distance values), creating skeleton representations, building markers for watershed segmentation, and computing clearance maps for robotics path planning.

## What the Distance Transform Computes

For every foreground (white) pixel in a binary image, the distance transform computes the distance to the nearest background (black) pixel:

```python
dist = cv2.distanceTransform(binary_image, distanceType, maskSize)
```

| Parameter | Meaning |
|---|---|
| `binary_image` | 8-bit single-channel binary image |
| `distanceType` | Distance metric to use |
| `maskSize` | Size of the distance transform mask (3, 5, or `cv2.DIST_MASK_PRECISE`) |

The output is a float32 image where each pixel's value is its distance to the nearest zero-valued pixel.

```python
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
```

## Distance Types

OpenCV supports several distance metrics:

| Constant | Distance | Description |
|---|---|---|
| `cv2.DIST_L1` | Manhattan | `|x1-x2| + |y1-y2|` — fast, diamond-shaped iso-distance lines |
| `cv2.DIST_L2` | Euclidean | `sqrt((x1-x2)^2 + (y1-y2)^2)` — true distance, circular iso-lines |
| `cv2.DIST_C` | Chebyshev | `max(|x1-x2|, |y1-y2|)` — square-shaped iso-distance lines |

```python
dist_l1 = cv2.distanceTransform(thresh, cv2.DIST_L1, 3)
dist_l2 = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
dist_c = cv2.distanceTransform(thresh, cv2.DIST_C, 3)
```

For most applications, `cv2.DIST_L2` with `maskSize=5` gives the best results. `DIST_L1` is faster but produces diamond-shaped artifacts.

## Normalizing the Output for Display

The raw distance values range from 0 to the maximum distance in the image. To display as a visible image, normalize to 0-255:

```python
# Normalize to 0-255 for display
dist_display = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_display = dist_display.astype(np.uint8)
```

Or apply a colormap for better visualization:

```python
dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dist_colored = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)
```

## Using Distance Transform for Marker Creation

The most common use of the distance transform is creating markers for watershed segmentation of touching objects:

```python
# Distance transform of binary image
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Threshold to find "sure foreground" (centers of objects)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
```

Pixels with high distance values are deep inside objects — far from any edge — making them reliable markers for the center of each object. The threshold factor (0.5 in this example) controls how conservative the markers are:

- **High threshold (0.7)**: Very small markers, only the deepest centers. Safer but may miss small objects.
- **Low threshold (0.3)**: Larger markers. May connect adjacent objects through narrow gaps.

## Skeleton Extraction with Distance Transform

A **skeleton** (or medial axis) is the set of pixels equidistant from two or more boundary points. While the full skeletonization algorithm is more involved, the distance transform provides a quick approximation:

```python
# Compute distance transform
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Apply Laplacian to find ridges (skeleton-like structure)
laplacian = cv2.Laplacian(dist, cv2.CV_64F)

# Threshold the Laplacian to get skeleton
_, skeleton = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY)

# A simpler approach: thin by successive erosion and subtraction
# (morphological skeleton)
skeleton = np.zeros_like(binary)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
temp = binary.copy()
while True:
    eroded = cv2.erode(temp, element)
    opened = cv2.dilate(eroded, element)
    diff = cv2.subtract(temp, opened)
    skeleton = cv2.bitwise_or(skeleton, diff)
    temp = eroded.copy()
    if cv2.countNonZero(temp) == 0:
        break
```

## Distance Transform on Real Shapes

The distance transform reveals the internal structure of shapes:

- A **circle** produces a cone-shaped distance map with its peak at the center.
- A **rectangle** produces a pyramid with its peak along the medial axis.
- **Touching circles** have distinct peaks, one per circle — perfect for counting and separating them.

```python
# Find peaks (local maxima) in the distance transform
# These correspond to object centers
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, peaks = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
```

## Tips & Common Mistakes

- The input must be a **single-channel 8-bit binary image**. Convert to grayscale and threshold first.
- The output is `float32`, not `uint8`. You must normalize or convert before displaying.
- `maskSize` must be 3, 5, or `cv2.DIST_MASK_PRECISE` (0). A mask size of 5 gives more accurate results than 3 for `DIST_L2`.
- The distance is measured **from foreground to the nearest background**. If your objects are black on white, invert the image first with `cv2.bitwise_not()`.
- The threshold for marker creation is relative to `dist.max()`. If the image has objects of very different sizes, a single global threshold may not work — consider adaptive approaches.
- `cv2.DIST_L2` with `maskSize=5` is the best general-purpose choice. Use `DIST_L1` only when speed is critical.
- The distance transform is often combined with `cv2.connectedComponents` to label the thresholded peaks as individual markers.
- For skeleton extraction, the morphological approach (successive erosion) is more robust than Laplacian-based methods.

## Starter Code

```python
import cv2
import numpy as np

# Create a binary image with various shapes
canvas = np.zeros((400, 700), dtype=np.uint8)

# Touching circles (classic distance transform use case)
cv2.circle(canvas, (100, 120), 60, 255, -1)
cv2.circle(canvas, (200, 120), 60, 255, -1)
cv2.circle(canvas, (150, 200), 55, 255, -1)

# Rectangle
cv2.rectangle(canvas, (320, 50), (500, 200), 255, -1)

# Elongated shape
cv2.ellipse(canvas, (600, 130), (70, 30), 30, 0, 360, 255, -1)

# L-shape for skeleton demo
l_shape = np.array([
    [50, 270], [200, 270], [200, 310],
    [90, 310], [90, 390], [50, 390]
], dtype=np.int32)
cv2.fillPoly(canvas, [l_shape], 255)

# Cross shape
cv2.rectangle(canvas, (280, 260), (320, 390), 255, -1)
cv2.rectangle(canvas, (240, 300), (360, 340), 255, -1)

# Small circle (to test threshold sensitivity)
cv2.circle(canvas, (500, 330), 30, 255, -1)

# Large circle
cv2.circle(canvas, (620, 330), 50, 255, -1)

# --- Compute distance transforms with different metrics ---
dist_l2 = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
dist_l1 = cv2.distanceTransform(canvas, cv2.DIST_L1, 3)
dist_c = cv2.distanceTransform(canvas, cv2.DIST_C, 3)

# --- Normalize for display ---
def normalize_dist(d):
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

dist_l2_display = normalize_dist(dist_l2)
dist_l1_display = normalize_dist(dist_l1)
dist_c_display = normalize_dist(dist_c)

# --- Threshold to find object centers (markers) ---
thresholds = [0.3, 0.5, 0.7]
marker_images = []
for t in thresholds:
    _, markers = cv2.threshold(dist_l2, t * dist_l2.max(), 255, 0)
    markers = markers.astype(np.uint8)
    marker_images.append(markers)

# --- Morphological skeleton ---
skeleton = np.zeros_like(canvas)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
temp = canvas.copy()
while True:
    eroded = cv2.erode(temp, element)
    opened = cv2.dilate(eroded, element)
    diff = cv2.subtract(temp, opened)
    skeleton = cv2.bitwise_or(skeleton, diff)
    temp = eroded.copy()
    if cv2.countNonZero(temp) == 0:
        break

# --- Build visualization ---
# Convert all to 3-channel for stacking
def to_bgr(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# Apply colormap to distance transforms
dist_l2_color = cv2.applyColorMap(dist_l2_display, cv2.COLORMAP_JET)
dist_l1_color = cv2.applyColorMap(dist_l1_display, cv2.COLORMAP_JET)

# Row 1: Original, L2 distance, L1 distance
panel_w, panel_h = 350, 200

def resize_panel(image, tw=panel_w, th=panel_h):
    return cv2.resize(image, (tw, th))

p1 = resize_panel(to_bgr(canvas))
cv2.putText(p1, 'Binary Input', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

p2 = resize_panel(dist_l2_color)
cv2.putText(p2, 'DIST_L2 (Euclidean)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Row 2: Markers at different thresholds, and skeleton
p3 = resize_panel(to_bgr(marker_images[0]))
cv2.putText(p3, f'Markers (t=0.3)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

p4 = resize_panel(to_bgr(marker_images[1]))
cv2.putText(p4, f'Markers (t=0.5)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Row 3: High threshold markers + skeleton
p5 = resize_panel(to_bgr(marker_images[2]))
cv2.putText(p5, f'Markers (t=0.7)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Skeleton overlay on original
skel_overlay = to_bgr(canvas).copy()
skel_overlay = resize_panel(skel_overlay)
skel_resized = resize_panel(skeleton)
skel_overlay[skel_resized > 0] = (0, 0, 255)
cv2.putText(skel_overlay, 'Skeleton (red)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Stack into grid
top_row = np.hstack([p1, p2])
mid_row = np.hstack([p3, p4])
bot_row = np.hstack([p5, skel_overlay])
result = np.vstack([top_row, mid_row, bot_row])

# Print statistics
print(f'Distance transform max (L2): {dist_l2.max():.1f} pixels')
print(f'Distance transform max (L1): {dist_l1.max():.1f} pixels')
print(f'Distance transform max (C):  {dist_c.max():.1f} pixels')
for i, t in enumerate(thresholds):
    n_components, _ = cv2.connectedComponents(marker_images[i])
    print(f'Threshold {t}: {n_components - 1} markers found')
print(f'Skeleton pixels: {cv2.countNonZero(skeleton)}')

cv2.imshow('Distance Transform', result)
```
