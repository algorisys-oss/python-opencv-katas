---
slug: 38-finding-contours
title: Finding Contours
level: intermediate
concepts: [cv2.findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE]
prerequisites: [26-simple-thresholding, 32-canny-edge-detection]
---

## What Problem Are We Solving?

Connected components tell you **which pixels** belong to each object, but they don't give you the **shape boundary** as a list of points. **Contours** are curves that trace the boundary of objects in a binary image. Once you have contours, you can measure shape properties, draw outlines, match shapes, and much more. Contour detection is one of the most fundamental operations in OpenCV.

## What Is a Contour?

A contour is a list of `(x, y)` points that form a continuous curve along the boundary of a white region in a binary image. Think of it as "tracing the outline" of each object.

## Using cv2.findContours()

```python
contours, hierarchy = cv2.findContours(binary, mode, method)
```

| Parameter | Meaning |
|---|---|
| `binary` | 8-bit single-channel binary image |
| `mode` | Contour retrieval mode (how to handle nested contours) |
| `method` | Contour approximation method (how many points to store) |

Returns:
- `contours` — a Python list of contours. Each contour is a NumPy array of shape `(N, 1, 2)` containing the boundary points.
- `hierarchy` — a NumPy array describing parent-child relationships between contours.

## Retrieval Modes

The retrieval mode controls **which** contours are returned and how they're organized:

```python
# Only outermost contours (ignores holes and nested shapes)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# All contours as a flat list (no hierarchy)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# All contours with full hierarchy (parent-child relationships)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

| Mode | Description | When to use |
|---|---|---|
| `RETR_EXTERNAL` | Only outermost contours | Counting separate objects |
| `RETR_LIST` | All contours, flat list | When you need every boundary but don't care about nesting |
| `RETR_TREE` | Full hierarchy tree | When shapes have holes or nested objects matter |
| `RETR_CCOMP` | Two-level hierarchy | Object boundaries and their holes |

## Approximation Methods

The approximation method controls **how many points** are stored per contour:

```python
# Store only endpoints of line segments (compact)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store every single boundary pixel (verbose)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
```

`CHAIN_APPROX_SIMPLE` is almost always the right choice — a rectangle stores just 4 points instead of hundreds.

## Drawing Contours

```python
# Draw all contours in green with thickness 2
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Draw only the 3rd contour (index 2)
cv2.drawContours(img, contours, 2, (0, 0, 255), 2)

# Fill contours (thickness = -1)
cv2.drawContours(img, contours, -1, (0, 255, 0), -1)
```

## Understanding Hierarchy

When using `RETR_TREE`, the `hierarchy` array has shape `(1, N, 4)` with columns:

```python
[next_sibling, prev_sibling, first_child, parent]
```

A value of `-1` means "none." For example, a contour with `parent=-1` is a top-level contour (no parent).

```python
# Check if contour i has a parent
if hierarchy[0][i][3] == -1:
    print(f'Contour {i} is a top-level contour')
else:
    print(f'Contour {i} is inside contour {hierarchy[0][i][3]}')
```

## Tips & Common Mistakes

- `findContours` modifies the input image in some OpenCV versions. Always pass a **copy**: `cv2.findContours(binary.copy(), ...)`.
- The input must be **binary** — apply thresholding or Canny first. Contours trace boundaries between black (0) and white (255).
- Contours are found on **white objects against a black background**. If your objects are dark on a light background, invert first.
- Each contour is shaped `(N, 1, 2)`, not `(N, 2)`. To access points normally, you can reshape: `contour.reshape(-1, 2)`.
- Use `RETR_EXTERNAL` when you only care about outer boundaries — it's faster and simpler.
- Sort contours by area to find the largest: `sorted(contours, key=cv2.contourArea, reverse=True)`.
- The number of contours is simply `len(contours)`.

## Starter Code

```python
import cv2
import numpy as np

# Create binary image with various shapes (some nested)
img = np.zeros((400, 600), dtype=np.uint8)

# Filled rectangle with a hole inside
cv2.rectangle(img, (20, 20), (180, 180), 255, -1)
cv2.rectangle(img, (60, 60), (140, 140), 0, -1)  # Hole

# Filled circle with a hole
cv2.circle(img, (300, 100), 80, 255, -1)
cv2.circle(img, (300, 100), 40, 0, -1)  # Hole
cv2.circle(img, (300, 100), 15, 255, -1)  # Object inside hole

# Simple shapes (no nesting)
pts = np.array([[450, 30], [550, 30], [570, 130], [430, 130]], np.int32)
cv2.fillPoly(img, [pts], 255)

cv2.ellipse(img, (120, 310), (80, 50), 0, 0, 360, 255, -1)
cv2.rectangle(img, (280, 250), (430, 370), 255, -1)
cv2.circle(img, (530, 310), 50, 255, -1)

# --- Find contours with different modes ---
# RETR_EXTERNAL: only outermost
contours_ext, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# RETR_LIST: all contours, flat list
contours_list, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# RETR_TREE: full hierarchy
contours_tree, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# --- Draw results ---
result_ext = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
result_list = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
result_tree = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

# Draw external contours
cv2.drawContours(result_ext, contours_ext, -1, (0, 255, 0), 2)

# Draw all contours (flat list) with different colors
for i, cnt in enumerate(contours_list):
    color = (
        int(np.random.randint(50, 256)),
        int(np.random.randint(50, 256)),
        int(np.random.randint(50, 256))
    )
    cv2.drawContours(result_list, [cnt], 0, color, 2)

# Draw tree contours: green for parents, red for children
for i, cnt in enumerate(contours_tree):
    parent = hierarchy[0][i][3]
    if parent == -1:
        cv2.drawContours(result_tree, [cnt], 0, (0, 255, 0), 2)  # Top-level: green
    else:
        cv2.drawContours(result_tree, [cnt], 0, (0, 0, 255), 2)  # Child: red

# --- Add labels ---
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(result_ext, f'RETR_EXTERNAL ({len(contours_ext)})', (10, 25),
            font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(result_list, f'RETR_LIST ({len(contours_list)})', (10, 25),
            font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(result_tree, f'RETR_TREE ({len(contours_tree)})', (10, 25),
            font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Print contour info
print(f'RETR_EXTERNAL: {len(contours_ext)} contours (outer boundaries only)')
print(f'RETR_LIST: {len(contours_list)} contours (all, flat)')
print(f'RETR_TREE: {len(contours_tree)} contours (all, with hierarchy)')
print(f'\nContour point counts (RETR_EXTERNAL):')
for i, cnt in enumerate(contours_ext):
    print(f'  Contour {i}: {len(cnt)} points, area = {cv2.contourArea(cnt):.0f}')

# Print hierarchy for RETR_TREE
print(f'\nHierarchy (RETR_TREE): [next, prev, child, parent]')
for i in range(len(contours_tree)):
    print(f'  Contour {i}: {hierarchy[0][i]}')

result = np.hstack([result_ext, result_list, result_tree])

cv2.imshow('Finding Contours', result)
```
