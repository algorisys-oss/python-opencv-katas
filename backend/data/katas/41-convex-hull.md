---
slug: 41-convex-hull
title: Convex Hull & Defects
level: intermediate
concepts: [cv2.convexHull, cv2.convexityDefects, convexity]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

Many real-world shapes are not perfectly convex — a hand has fingers sticking out, a star has points, and an irregular object has concavities. The **convex hull** is the smallest convex polygon that completely encloses a contour, like stretching a rubber band around the shape. The gaps between the hull and the actual contour are called **convexity defects**, and they reveal important structural features — like the valleys between fingers in hand gesture recognition.

## What Is a Convex Hull?

A shape is **convex** if, for any two points inside it, the straight line between them also lies entirely inside the shape. A circle is convex. A star is not (the line between two tips passes outside the shape).

The convex hull of a contour is the tightest convex polygon that contains all the contour points. Think of it as the shape you'd get by wrapping a rubber band around pushpins placed at each contour point.

```python
hull = cv2.convexHull(contour)
```

This returns the hull as an array of points, which you can draw like any contour:

```python
cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)
```

## Checking Convexity

You can test whether a contour is already convex:

```python
is_convex = cv2.isContourConvex(contour)
```

This returns `True` if the contour has no concavities, `False` otherwise. A rectangle is convex; a hand silhouette is not.

## Finding Convexity Defects

Convexity defects are the regions where the contour deviates inward from its convex hull. Each defect is defined by a start point, end point, and the farthest point (the deepest point of the concavity):

```python
# IMPORTANT: hull must be computed with returnPoints=False for defects
hull_indices = cv2.convexHull(contour, returnPoints=False)
defects = cv2.convexityDefects(contour, hull_indices)
```

The `returnPoints=False` parameter is critical — it returns hull indices into the contour array instead of the actual point coordinates, which is what `cv2.convexityDefects()` requires.

Each row in the `defects` array has 4 values:

| Value | Meaning |
|---|---|
| `start_index` | Index into contour of the defect start point |
| `end_index` | Index into contour of the defect end point |
| `farthest_index` | Index into contour of the deepest point |
| `fixpt_depth` | Distance of farthest point from hull, in **fixed-point** (divide by 256.0 to get pixels) |

```python
if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        depth = d / 256.0  # Convert to pixels

        if depth > 10:  # Only significant defects
            cv2.circle(img, far, 5, (0, 0, 255), -1)
            cv2.line(img, start, end, (0, 255, 0), 2)
```

## Practical Use: Hand Gesture Recognition

Convexity defects are the foundation of simple hand gesture recognition. The idea:

1. Segment the hand from the background.
2. Find the hand contour.
3. Compute the convex hull and defects.
4. Count the defects with sufficient depth — each deep defect corresponds to a valley between fingers.
5. Number of deep defects + 1 approximates the number of extended fingers.

```python
# Count fingers (simplified logic)
deep_defects = 0
if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth > 20:  # Threshold for "real" finger valley
            deep_defects += 1

finger_count = deep_defects + 1
```

## Convex Hull Area vs Contour Area

Comparing the contour area to the convex hull area gives you a measure called **solidity**:

```python
contour_area = cv2.contourArea(contour)
hull_area = cv2.contourArea(cv2.convexHull(contour))
solidity = contour_area / hull_area if hull_area > 0 else 0
```

A solidity close to 1.0 means the shape is nearly convex (like a circle or rectangle). A low solidity means there are significant concavities (like a star or hand).

## Tips & Common Mistakes

- Use `returnPoints=False` when computing the hull for `cv2.convexityDefects()`. If you pass the hull points directly, you'll get an error.
- The `fixpt_depth` value in defects is in **fixed-point format** — divide by 256.0 to get the depth in pixels.
- Filter defects by depth. Small defects (a few pixels) are usually noise from contour irregularities.
- `cv2.convexityDefects()` returns `None` if there are no defects (contour is already convex). Always check for `None` before iterating.
- The contour must have at least 3 points for a convex hull to be computed.
- For hand gesture detection, you'll also want to filter defects by angle — shallow angles between start/far/end points indicate real finger valleys, while steep angles are noise.

## Starter Code

```python
import cv2
import numpy as np
import math

# Create a canvas and draw a star-like hand shape
canvas = np.zeros((500, 700, 3), dtype=np.uint8)

# Draw a hand-like shape using a polygon
# Palm center at (350, 300), with finger-like protrusions
hand_points = np.array([
    # Thumb
    [150, 300], [130, 220], [160, 150], [190, 220],
    # Index finger
    [210, 200], [220, 100], [250, 90], [260, 190],
    # Middle finger
    [280, 170], [300, 60], [330, 55], [340, 160],
    # Ring finger
    [360, 170], [380, 80], [410, 85], [410, 170],
    # Pinky
    [430, 190], [460, 120], [490, 140], [470, 220],
    # Palm right side and bottom
    [480, 280], [470, 380], [400, 430],
    # Bottom of palm
    [300, 440], [200, 430], [150, 380],
], dtype=np.int32)

cv2.fillPoly(canvas, [hand_points], (255, 255, 255))

# Convert to grayscale and threshold
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# --- Compute convex hull (for drawing) ---
hull_points = cv2.convexHull(contour)

# --- Compute convex hull (for defects - need indices) ---
hull_indices = cv2.convexHull(contour, returnPoints=False)
defects = cv2.convexityDefects(contour, hull_indices)

# --- Draw results ---
# Left: contour + hull overlay
left = canvas.copy()
cv2.drawContours(left, [contour], 0, (0, 255, 0), 2)
cv2.drawContours(left, [hull_points], 0, (255, 0, 0), 2)
cv2.putText(left, 'Contour + Hull', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Right: contour + defect points
right = canvas.copy()
cv2.drawContours(right, [contour], 0, (0, 255, 0), 2)

deep_defect_count = 0
if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        depth = d / 256.0

        if depth > 15:
            deep_defect_count += 1
            # Draw the defect: line from start to end, circle at farthest point
            cv2.line(right, start, end, (255, 0, 0), 2)
            cv2.circle(right, far, 6, (0, 0, 255), -1)
            cv2.circle(right, start, 4, (255, 255, 0), -1)
            cv2.circle(right, end, 4, (255, 255, 0), -1)
            # Label the depth
            cv2.putText(right, f'{depth:.0f}', (far[0]+8, far[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

cv2.putText(right, f'Defects (depth>15): {deep_defect_count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv2.putText(right, f'Approx fingers: {deep_defect_count + 1}', (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

# Print properties
is_convex = cv2.isContourConvex(contour)
contour_area = cv2.contourArea(contour)
hull_area = cv2.contourArea(hull_points)
solidity = contour_area / hull_area if hull_area > 0 else 0

print(f'Contour is convex: {is_convex}')
print(f'Contour area: {contour_area:.0f}')
print(f'Hull area: {hull_area:.0f}')
print(f'Solidity: {solidity:.3f}')
print(f'Total defects: {defects.shape[0] if defects is not None else 0}')
print(f'Deep defects (depth > 15): {deep_defect_count}')

result = np.hstack([left, right])

cv2.imshow('Convex Hull & Defects', result)
```
