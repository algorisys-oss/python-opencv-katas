---
slug: 37-connected-components
title: Connected Components
level: intermediate
concepts: [cv2.connectedComponents, cv2.connectedComponentsWithStats, labeling]
prerequisites: [26-simple-thresholding]
---

## What Problem Are We Solving?

After thresholding, you have a binary image with white blobs. But how many separate objects are there? Which pixels belong to which object? How big is each one? **Connected component labeling** answers all these questions — it assigns a unique **label** (integer ID) to each group of connected white pixels, turning a binary image into a labeled map where you can identify and measure individual objects.

## What Are Connected Components?

Two white pixels are "connected" if they are neighbors. By default, OpenCV uses **8-connectivity** (a pixel's 8 surrounding neighbors). Each isolated group of connected white pixels forms one **component**. The background (black) is always label 0.

## Using cv2.connectedComponents()

```python
num_labels, labels = cv2.connectedComponents(binary_image)
```

| Return Value | Meaning |
|---|---|
| `num_labels` | Total number of labels (including background) — so `num_labels - 1` is the number of objects |
| `labels` | A matrix the same size as the input, where each pixel's value is its label ID (0 = background) |

## Using connectedComponentsWithStats()

The `WithStats` version gives you measurements for each component:

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
```

| Return Value | Meaning |
|---|---|
| `stats` | A `(num_labels, 5)` array with columns: `[x, y, width, height, area]` |
| `centroids` | A `(num_labels, 2)` array with `[cx, cy]` for each component |

The stats columns are accessed via constants:

```python
x = stats[label, cv2.CC_STAT_LEFT]
y = stats[label, cv2.CC_STAT_TOP]
w = stats[label, cv2.CC_STAT_WIDTH]
h = stats[label, cv2.CC_STAT_HEIGHT]
area = stats[label, cv2.CC_STAT_AREA]
```

## Visualizing Labels with Color

Labels are integers (0, 1, 2, ...). To visualize them, map each label to a unique color:

```python
# Create a color map
label_hue = np.uint8(179 * labels / np.max(labels))
blank = 255 * np.ones_like(label_hue)
colored = cv2.merge([label_hue, blank, blank])
colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)
colored[labels == 0] = 0  # Background stays black
```

## Filtering Components by Size

A common task is to keep only components above (or below) a certain area:

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

min_area = 500
filtered = np.zeros_like(binary)

for i in range(1, num_labels):  # Skip label 0 (background)
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        filtered[labels == i] = 255
```

## Tips & Common Mistakes

- Label 0 is always the **background**. Real objects start at label 1. Loop from `range(1, num_labels)` to skip the background.
- The input must be a **binary image** (8-bit, single channel). Apply thresholding first.
- `connectedComponents` uses 8-connectivity by default. Pass `connectivity=4` for stricter 4-connectivity if needed.
- The `labels` matrix has dtype `int32` — don't confuse it with a displayable image. You need to map it to colors or normalize it for visualization.
- Filtering by area is the simplest way to remove noise after thresholding. It's often more effective than morphological operations for removing scattered small blobs.
- For very large images, `connectedComponentsWithStats` is more efficient than finding contours when you only need area and bounding box — it doesn't store point lists.

## Starter Code

```python
import cv2
import numpy as np

# Create a binary image with multiple objects of different sizes
img = np.zeros((400, 600), dtype=np.uint8)

# Large objects
cv2.rectangle(img, (20, 20), (140, 140), 255, -1)
cv2.circle(img, (260, 80), 60, 255, -1)
cv2.ellipse(img, (440, 80), (80, 50), 20, 0, 360, 255, -1)

# Medium objects
cv2.rectangle(img, (50, 220), (130, 300), 255, -1)
cv2.circle(img, (250, 280), 35, 255, -1)

# Small noise blobs
np.random.seed(42)
for _ in range(30):
    x = np.random.randint(0, 600)
    y = np.random.randint(180, 400)
    r = np.random.randint(2, 8)
    cv2.circle(img, (x, y), r, 255, -1)

# Irregularly shaped object
pts = np.array([[380, 220], [480, 200], [550, 270], [520, 370], [400, 350]], np.int32)
cv2.fillPoly(img, [pts], 255)

# --- Connected components with stats ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

print(f'Found {num_labels - 1} connected components (excluding background)')
print(f'{"Label":<8} {"Area":<8} {"X":<6} {"Y":<6} {"W":<6} {"H":<6} {"Centroid"}')
print('-' * 60)
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = centroids[i]
    print(f'{i:<8} {area:<8} {x:<6} {y:<6} {w:<6} {h:<6} ({cx:.1f}, {cy:.1f})')

# --- Colorize labels ---
label_hue = np.uint8(179 * labels / max(np.max(labels), 1))
blank_ch = 255 * np.ones_like(label_hue)
colored = cv2.merge([label_hue, blank_ch, blank_ch])
colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)
colored[labels == 0] = 0

# Draw bounding boxes and centroids on the colored image
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = int(centroids[i][0]), int(centroids[i][1])

    cv2.rectangle(colored, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.circle(colored, (cx, cy), 3, (255, 255, 255), -1)
    cv2.putText(colored, f'{area}', (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

# --- Filter by minimum area ---
min_area = 500
filtered = np.zeros_like(img)
kept = 0
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        filtered[labels == i] = 255
        kept += 1

print(f'\nAfter filtering (min area={min_area}): {kept} components kept')

# --- Build comparison display ---
original_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(original_bgr, 'Original Binary', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(colored, 'Labeled (with stats)', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(filtered_bgr, f'Filtered (area >= {min_area})', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

result = np.hstack([original_bgr, colored, filtered_bgr])

cv2.imshow('Connected Components', result)
```
