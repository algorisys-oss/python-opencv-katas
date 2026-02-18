---
slug: 47-watershed-segmentation
title: Watershed Algorithm
level: intermediate
concepts: [cv2.watershed, marker-based segmentation]
prerequisites: [33-morphology-erosion, 37-connected-components]
---

## What Problem Are We Solving?

When objects in an image are touching or overlapping, simple thresholding can't separate them — they appear as one connected blob. The **watershed algorithm** solves this by treating the image as a topographic surface (bright pixels = peaks, dark pixels = valleys) and "flooding" from known markers. Where the floods from different markers meet, a boundary is drawn. This is how you separate touching coins, overlapping cells in microscopy, or clustered objects in industrial inspection.

## The Watershed Concept

Imagine the grayscale image as a 3D landscape where pixel intensity is elevation:

1. You place **markers** (labels) at locations you're certain belong to different objects — like dropping colored dye into the valleys.
2. The algorithm "floods" upward from each marker simultaneously.
3. Where the floods from two different markers would meet, a **dam** (boundary) is built.
4. The result is a segmented image where each pixel is assigned to one of the markers (or to a boundary).

The key insight is that you must provide the initial markers. The watershed algorithm is **marker-based** — it doesn't segment automatically.

## Creating Markers with Distance Transform

The standard pipeline for separating touching objects is:

1. Threshold the image to get a binary mask.
2. Apply the **distance transform** to find pixels far from any edge (the centers of objects).
3. Threshold the distance transform to get "sure foreground" markers.
4. Use morphological dilation to find "sure background."
5. The unknown region is everything in between.
6. Label the markers with `cv2.connectedComponents`.
7. Run `cv2.watershed`.

```python
# Step 1: Binary threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 2: Remove noise with morphology
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 3: Sure background (dilated region)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 4: Sure foreground (distance transform + threshold)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# Step 5: Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 6: Label markers
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Background becomes 1, not 0
markers[unknown == 255] = 0  # Unknown region marked as 0
```

## Running the Watershed Algorithm

```python
markers = cv2.watershed(img, markers)
```

After `cv2.watershed`:
- Boundary pixels are set to `-1`
- Each object region gets its marker label (1, 2, 3, ...)
- Background pixels get label 1 (from our `+1` offset)

```python
# Mark boundaries in red on the original image
img[markers == -1] = [0, 0, 255]
```

## Understanding the Marker Array

The marker array is a 32-bit signed integer array where:

| Value | Meaning |
|---|---|
| `0` | Unknown region (to be determined by watershed) |
| `1` | Background (or first region) |
| `2, 3, 4, ...` | Individual object regions |
| `-1` | Boundary between regions (set by watershed) |

The `+1` offset is important: `cv2.watershed` treats `0` as "unknown" and will try to assign those pixels. If your background is labeled `0`, the algorithm won't know it's background.

## Coloring the Segmentation Result

To visualize the result, assign a random color to each marker label:

```python
# Create a colored output
result = np.zeros_like(img)
unique_labels = np.unique(markers)

for label in unique_labels:
    if label <= 0:
        continue  # Skip boundaries and unknown
    color = (np.random.randint(50, 255),
             np.random.randint(50, 255),
             np.random.randint(50, 255))
    result[markers == label] = color

# Draw boundaries
result[markers == -1] = (0, 0, 255)
```

## Tips & Common Mistakes

- `cv2.watershed` requires a **3-channel (BGR) input image**, even if your original is grayscale. Convert with `cv2.cvtColor`.
- The markers array must be **`int32` (CV_32SC1)** type. Use `markers.astype(np.int32)` if needed.
- Markers of `0` mean "unknown" — the algorithm will assign these pixels. Never leave foreground or background as `0`.
- The distance transform threshold (`0.5 * dist.max()`) is a tuning parameter. Lower values create larger markers (may merge objects), higher values create smaller markers (may miss objects).
- Always apply morphological opening before the distance transform to remove noise and thin connections between objects.
- The watershed algorithm tends to over-segment if there are too many markers. Use fewer, well-placed markers for cleaner results.
- Boundaries (`-1` values) are single-pixel-wide lines. They may not be visible at low resolution.
- The algorithm modifies the markers array **in place**.

## Starter Code

```python
import cv2
import numpy as np

# Create an image with touching/overlapping circles (simulating coins)
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (30, 30, 30)  # Dark background

# Draw overlapping circles with slight color variation
circle_centers = [
    (120, 150), (210, 140), (170, 230),
    (380, 130), (450, 160), (420, 240),
    (300, 330), (400, 340), (500, 320)
]
for cx, cy in circle_centers:
    shade = np.random.randint(180, 240)
    cv2.circle(img, (cx, cy), 55, (shade, shade, shade), -1)

# Add slight noise for realism
noise = np.random.randint(-10, 11, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Watershed segmentation pipeline ---

# Step 1: Grayscale + Otsu threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 2: Morphological opening to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 3: Sure background via dilation
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 4: Distance transform -> sure foreground
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)

# Step 5: Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 6: Label markers
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1          # Shift so background = 1
markers[unknown == 255] = 0    # Mark unknown as 0

num_objects = markers.max() - 1  # Subtract 1 for background

# --- Visualize intermediate steps ---
dist_display = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dist_color = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)

markers_before = markers.copy()

# Step 7: Apply watershed
markers = cv2.watershed(img, markers)

# --- Build visualization ---
# Colored segmentation result
segmented = np.zeros_like(img)
np.random.seed(42)
colors = {}
for label in range(2, markers_before.max() + 1):
    colors[label] = (np.random.randint(60, 255),
                     np.random.randint(60, 255),
                     np.random.randint(60, 255))

for label, color in colors.items():
    segmented[markers == label] = color

# Draw boundaries in red on original
boundary_img = img.copy()
boundary_img[markers == -1] = [0, 0, 255]

# Create display panels
panel_h, panel_w = 200, 300

def resize_panel(image, target_w, target_h):
    return cv2.resize(image, (target_w, target_h))

p1 = resize_panel(img, panel_w, panel_h)
cv2.putText(p1, 'Original', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

p2 = resize_panel(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), panel_w, panel_h)
cv2.putText(p2, 'Threshold', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

p3 = resize_panel(dist_color, panel_w, panel_h)
cv2.putText(p3, 'Distance Transform', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

p4 = resize_panel(cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR), panel_w, panel_h)
cv2.putText(p4, 'Sure Foreground', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

p5 = resize_panel(segmented, panel_w, panel_h)
cv2.putText(p5, f'Segmented ({num_objects} objects)', (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

p6 = resize_panel(boundary_img, panel_w, panel_h)
cv2.putText(p6, 'Boundaries (red)', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

top_row = np.hstack([p1, p2, p3])
bot_row = np.hstack([p4, p5, p6])
result = np.vstack([top_row, bot_row])

print(f'Objects detected before watershed: {num_objects}')
print(f'Unique labels after watershed: {np.unique(markers)}')
print(f'Boundary pixels: {np.sum(markers == -1)}')

cv2.imshow('Watershed Algorithm', result)
```
