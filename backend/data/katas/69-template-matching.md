---
slug: 69-template-matching
title: Template Matching
level: advanced
concepts: [cv2.matchTemplate, TM_CCOEFF_NORMED, multi-location]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

Sometimes you need to find a specific known pattern (a **template**) within a larger image — like finding an icon on a screenshot, locating a logo in a photo, or detecting a specific object whose appearance is fixed. **Template matching** slides the template across the image, computing a similarity score at each position. The location(s) with the highest score indicate where the template appears. Unlike feature-based methods, template matching works directly on pixel values and requires no feature detection.

## How cv2.matchTemplate Works

`cv2.matchTemplate()` slides a template image over the input image and computes a similarity (or difference) metric at each position:

```python
result = cv2.matchTemplate(image, template, method)
```

| Parameter | Meaning |
|---|---|
| `image` | Input image to search in (grayscale or color) |
| `template` | The small image to search for |
| `method` | Comparison method (determines how similarity is computed) |

The output `result` is a 2D array of size `(H - h + 1, W - w + 1)` where `(H, W)` is the image size and `(h, w)` is the template size. Each value represents the match quality at that position.

## Template Matching Methods

OpenCV provides six matching methods:

```python
# Squared difference (lower = better match)
cv2.TM_SQDIFF
cv2.TM_SQDIFF_NORMED

# Cross-correlation (higher = better match)
cv2.TM_CCORR
cv2.TM_CCORR_NORMED

# Correlation coefficient (higher = better match, handles brightness changes)
cv2.TM_CCOEFF
cv2.TM_CCOEFF_NORMED
```

| Method | Best Match | Range (normalized) | Brightness Invariant |
|---|---|---|---|
| `TM_SQDIFF_NORMED` | Minimum value | 0 to 1 | No |
| `TM_CCORR_NORMED` | Maximum value | 0 to 1 | No |
| `TM_CCOEFF_NORMED` | Maximum value | -1 to 1 | Yes |

`TM_CCOEFF_NORMED` is the most commonly used because it is **normalized** (values between -1 and 1) and **handles brightness variations** by subtracting the mean.

## Finding the Best Match Location

Use `cv2.minMaxLoc()` on the result to find the best match:

```python
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# For TM_CCOEFF_NORMED, the best match is at max_loc
top_left = max_loc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
```

For `TM_SQDIFF` methods, the best match is at `min_loc` (minimum difference). For all other methods, use `max_loc` (maximum similarity).

## Multi-Object Detection with Thresholding

When the template appears **multiple times** in the image, `minMaxLoc` only finds one. To detect all occurrences, threshold the result map:

```python
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

# Find all locations above the threshold
locations = np.where(result >= threshold)
h, w = template.shape[:2]

# Draw rectangles at all match locations
for pt in zip(*locations[::-1]):  # Switch to (x, y) from (row, col)
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
```

## Handling Overlapping Detections

Thresholding often produces many overlapping detections at each match location. Use non-maximum suppression or grouping to clean them up:

```python
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
locations = np.where(result >= threshold)

# Group nearby detections by collecting rectangles
rectangles = []
h, w = template.shape[:2]
for pt in zip(*locations[::-1]):
    rectangles.append([pt[0], pt[1], w, h])
    rectangles.append([pt[0], pt[1], w, h])  # Duplicate for groupRectangles

# Group overlapping rectangles
grouped, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
for (x, y, w, h) in grouped:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

## Tips & Common Mistakes

- Template matching is **not scale-invariant** and **not rotation-invariant**. The template must appear at approximately the same size and orientation in the image. For scale/rotation tolerance, you need multi-scale matching or feature-based approaches.
- Use `TM_CCOEFF_NORMED` as your default method — it handles brightness differences and produces scores between -1 and 1 that are easy to threshold.
- For `TM_SQDIFF` and `TM_SQDIFF_NORMED`, the best match is the **minimum** (not maximum). This is a common source of bugs.
- The threshold for multi-object detection depends on the method and image content. Start with 0.8 for `TM_CCOEFF_NORMED` and adjust.
- Template matching on color images matches all channels simultaneously. Using grayscale is faster and often sufficient, but color can help when similar shapes have different colors.
- The result map has size `(H - h + 1, W - w + 1)`, not the same size as the input image. The top-left corner of the match rectangle corresponds to each position in the result map.
- `np.where` returns `(rows, cols)` but OpenCV uses `(x, y)` coordinates. Use `zip(*locations[::-1])` to convert from (row, col) to (x, y).
- For large images or many templates, template matching can be slow. Consider resizing the image and template together (maintaining the ratio) to speed things up.

## Starter Code

```python
import cv2
import numpy as np

# Create a scene with repeated patterns (multiple targets)
scene = np.zeros((400, 600, 3), dtype=np.uint8)
scene[:] = (60, 60, 60)

# Draw a distinctive small shape that we'll use as the template
def draw_target(img, x, y, size=30):
    cv2.rectangle(img, (x, y), (x + size, y + size), (0, 200, 200), -1)
    cv2.rectangle(img, (x, y), (x + size, y + size), (0, 255, 255), 2)
    cv2.line(img, (x, y), (x + size, y + size), (0, 0, 0), 2)
    cv2.line(img, (x + size, y), (x, y + size), (0, 0, 0), 2)

# Place the target at multiple known locations
target_positions = [(50, 50), (200, 80), (400, 60), (100, 250), (350, 280), (500, 200)]
for (tx, ty) in target_positions:
    draw_target(scene, tx, ty)

# Add distracting shapes (non-targets)
cv2.circle(scene, (300, 180), 30, (200, 200, 200), -1)
cv2.rectangle(scene, (450, 130), (510, 190), (150, 150, 255), -1)
cv2.putText(scene, 'Find all targets!', (150, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

# Add subtle noise
noise = np.random.randint(-8, 8, scene.shape, dtype=np.int16)
scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Extract the template from the scene ---
template = scene[50:50 + 34, 50:50 + 34].copy()  # Slightly larger to include border

gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
th, tw = gray_template.shape[:2]

# --- Method 1: Best single match ---
result_ccoeff = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_ccoeff)

img_single = scene.copy()
cv2.rectangle(img_single, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2)

# --- Method 2: Multi-object detection with thresholding ---
threshold = 0.7
locations = np.where(result_ccoeff >= threshold)

img_multi = scene.copy()
match_count = 0
# Group nearby detections
rectangles = []
for pt in zip(*locations[::-1]):
    rectangles.append([pt[0], pt[1], tw, th])
    rectangles.append([pt[0], pt[1], tw, th])

if len(rectangles) > 0:
    grouped, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
    for (x, y, w, h) in grouped:
        cv2.rectangle(img_multi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        match_count = len(grouped)
else:
    match_count = 0

# --- Method 3: Compare different methods ---
result_sqdiff = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_SQDIFF_NORMED)
result_ccorr = cv2.matchTemplate(gray_scene, gray_template, cv2.TM_CCORR_NORMED)

# Normalize result maps for display
def result_to_display(result_map, size):
    disp = cv2.normalize(result_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return cv2.resize(disp, size)

display_size = (300, 200)
disp_ccoeff = result_to_display(result_ccoeff, display_size)
disp_sqdiff = result_to_display(result_sqdiff, display_size)
disp_ccorr = result_to_display(result_ccorr, display_size)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_single, f'Best match (score={max_val:.3f})', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_multi, f'All matches ({match_count} found, thresh={threshold})', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(disp_ccoeff, 'TM_CCOEFF_NORMED', (5, 18), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(disp_sqdiff, 'TM_SQDIFF_NORMED', (5, 18), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(disp_ccorr, 'TM_CCORR_NORMED', (5, 18), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# Show the template (enlarged for visibility)
template_display = cv2.resize(template, (100, 100), interpolation=cv2.INTER_NEAREST)

# Build layout
# Top row: single match result | multi-match result
img_single_resized = cv2.resize(img_single, (300, 200))
img_multi_resized = cv2.resize(img_multi, (300, 200))
top_row = np.hstack([img_single_resized, img_multi_resized])

# Bottom row: the three method heatmaps
bottom_row = np.hstack([disp_ccoeff, disp_sqdiff])

# Pad bottom row to match top row width
if bottom_row.shape[1] < top_row.shape[1]:
    pad = np.zeros((bottom_row.shape[0], top_row.shape[1] - bottom_row.shape[1], 3), dtype=np.uint8)
    bottom_row = np.hstack([bottom_row, pad])

result = np.vstack([top_row, bottom_row])

print(f'Template size: {tw}x{th}')
print(f'Result map size: {result_ccoeff.shape[1]}x{result_ccoeff.shape[0]}')
print(f'Best match score (TM_CCOEFF_NORMED): {max_val:.4f}')
print(f'Best match location: {max_loc}')
print(f'Matches found (threshold={threshold}): {match_count}')
print(f'Actual targets placed: {len(target_positions)}')

cv2.imshow('Template Matching', result)
```
