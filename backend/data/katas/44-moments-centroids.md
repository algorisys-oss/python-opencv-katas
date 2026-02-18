---
slug: 44-moments-centroids
title: Moments & Centroids
level: intermediate
concepts: [cv2.moments, center of mass, hu moments]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

You've found contours in an image, but now you need to know **where** each shape is centered, how to compare shapes regardless of their size or rotation, or how to compute weighted properties of a region. **Image moments** are a set of statistical measures computed from a contour (or image region) that capture its area, center of mass, orientation, and more abstract shape characteristics. The centroid tells you the center point, and Hu moments give you rotation/scale-invariant descriptors for shape matching.

## What Are Image Moments?

Moments are weighted averages of pixel positions. For a contour, `cv2.moments()` returns a dictionary of spatial moments, central moments, and normalized central moments:

```python
M = cv2.moments(contour)
```

The returned dictionary contains:

| Key Pattern | Meaning |
|---|---|
| `m00`, `m10`, `m01`, `m20`, `m11`, `m02`, `m30`, `m21`, `m12`, `m03` | **Spatial moments** — raw weighted sums |
| `mu20`, `mu11`, `mu02`, `mu30`, `mu21`, `mu12`, `mu03` | **Central moments** — translation-invariant |
| `nu20`, `nu11`, `nu02`, `nu30`, `nu21`, `nu12`, `nu03` | **Normalized central moments** — translation and scale invariant |

The most important one for basic use is `m00`, which equals the **contour area** (same as `cv2.contourArea()`).

## Computing the Centroid

The centroid (center of mass) is computed directly from the first-order spatial moments:

```python
M = cv2.moments(contour)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
else:
    cx, cy = 0, 0
```

The formula is:
- `cx = m10 / m00` (average x-coordinate, weighted by area)
- `cy = m01 / m00` (average y-coordinate, weighted by area)

Always guard against `m00 == 0`, which happens for degenerate contours (e.g., a single point or a line with zero area).

```python
# Draw the centroid on the image
cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
```

## Moments from an Image Region (Not Just Contours)

You can also compute moments directly from a grayscale or binary image:

```python
# Moments of a binary mask
M = cv2.moments(binary_image)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
```

When computed from an image, the pixel intensities act as weights. In a binary image (0 and 255), this is equivalent to computing the centroid of the white region.

## Hu Moments for Shape Matching

Central moments are translation-invariant, and normalized central moments add scale-invariance. **Hu moments** go one step further — they are a set of 7 values that are invariant to translation, scale, and rotation:

```python
M = cv2.moments(contour)
hu = cv2.HuMoments(M)
```

This returns a 7x1 array. The values span many orders of magnitude, so it's common to log-transform them:

```python
# Log-transform for easier comparison
hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
```

## Comparing Shapes with Hu Moments

Two shapes that are the same but differ in position, size, or rotation will have similar Hu moments. OpenCV provides `cv2.matchShapes()` to compare contours using Hu moments:

```python
score = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
```

| Method | Description |
|---|---|
| `cv2.CONTOURS_MATCH_I1` | Sum of absolute differences of Hu moment log values |
| `cv2.CONTOURS_MATCH_I2` | Sum of absolute differences of reciprocals |
| `cv2.CONTOURS_MATCH_I3` | Maximum of relative differences |

A **lower score** means the shapes are more similar. A score of 0 means identical shapes.

```python
# Compare a circle contour to a rotated ellipse contour
score = cv2.matchShapes(circle_cnt, ellipse_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
print(f'Similarity score: {score:.4f}')  # Low = similar, high = different
```

## Orientation from Moments

The central moments can give you the **orientation** (angle of the major axis) of a shape:

```python
M = cv2.moments(contour)
# Angle in radians
angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
angle_deg = np.degrees(angle)
```

This is the angle of the axis along which the shape has the most spread. It's useful for aligning objects or detecting rotation.

## Tips & Common Mistakes

- Always check `M['m00'] != 0` before computing the centroid. A zero-area contour will cause a division-by-zero error.
- `cv2.moments()` works on both contours (point arrays) and images (2D arrays). When used on images, pixel values are weights.
- Hu moments are invariant to translation, scale, and rotation, but **not** to reflection. A mirrored shape may have a different 7th Hu moment.
- `cv2.matchShapes()` uses Hu moments internally — you don't need to compute them yourself for shape comparison.
- The centroid of a contour may lie **outside** the contour for concave shapes (like a crescent or donut).
- Log-transform Hu moments before comparing them manually, as their raw values span many orders of magnitude.
- The 4th parameter of `cv2.matchShapes()` is unused and should be `0.0`.

## Starter Code

```python
import cv2
import numpy as np

# Create canvas with shapes to analyze
canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# Row 1: Different shapes (for centroid computation)
# Triangle
tri_pts = np.array([[100, 30], [40, 180], [160, 180]], dtype=np.int32)
cv2.fillPoly(canvas, [tri_pts], (255, 255, 255))

# Circle
cv2.circle(canvas, (300, 110), 80, (255, 255, 255), -1)

# Rectangle
cv2.rectangle(canvas, (430, 40), (580, 180), (255, 255, 255), -1)

# L-shape (concave - centroid may be outside)
l_pts = np.array([
    [640, 30], [750, 30], [750, 80], [690, 80],
    [690, 180], [640, 180]
], dtype=np.int32)
cv2.fillPoly(canvas, [l_pts], (255, 255, 255))

# Row 2: Shape matching pairs
# Small circle
cv2.circle(canvas, (80, 350), 40, (255, 255, 255), -1)
# Large circle (should match small circle)
cv2.circle(canvas, (220, 350), 70, (255, 255, 255), -1)

# Small triangle
tri_sm = np.array([[350, 300], [320, 390], [380, 390]], dtype=np.int32)
cv2.fillPoly(canvas, [tri_sm], (255, 255, 255))
# Rotated triangle (should match small triangle)
tri_rot = np.array([[500, 380], [460, 300], [540, 310]], dtype=np.int32)
cv2.fillPoly(canvas, [tri_rot], (255, 255, 255))

# Square for comparison
cv2.rectangle(canvas, (620, 300), (720, 400), (255, 255, 255), -1)

# Find contours
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours left-to-right, top-to-bottom for consistent labeling
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))

result = canvas.copy()

# --- Compute and draw centroids for each shape ---
print('=== Moments & Centroids ===')
for i, cnt in enumerate(contours):
    M = cv2.moments(cnt)
    area = M['m00']
    if area == 0:
        continue

    # Compute centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Compute orientation
    angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
    angle_deg = np.degrees(angle)

    # Draw contour, centroid, and label
    color = (0, 255, 0)
    cv2.drawContours(result, [cnt], 0, color, 2)
    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(result, f'{i}', (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw orientation line
    line_len = 40
    x2 = int(cx + line_len * np.cos(angle))
    y2 = int(cy + line_len * np.sin(angle))
    cv2.arrowedLine(result, (cx, cy), (x2, y2), (255, 0, 255), 2)

    # Compute Hu moments
    hu = cv2.HuMoments(M).flatten()
    hu_log = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu]

    print(f'Shape {i}: centroid=({cx},{cy}), area={area:.0f}, '
          f'angle={angle_deg:.1f} deg, hu[0]={hu_log[0]:.3f}')

# --- Shape matching comparisons ---
print('\n=== Shape Matching (lower = more similar) ===')
if len(contours) >= 5:
    # Compare shapes in row 2
    pairs = [
        (4, 5, 'Small circle vs Large circle'),
        (4, 6, 'Small circle vs Small triangle'),
        (6, 7, 'Small triangle vs Rotated triangle'),
        (6, 8, 'Small triangle vs Square'),
    ]
    # Adjust indices based on actual contour count
    n = len(contours)
    for i, j, desc in pairs:
        if i < n and j < n:
            score = cv2.matchShapes(contours[i], contours[j], cv2.CONTOURS_MATCH_I1, 0.0)
            print(f'  {desc}: {score:.4f}')

            # Draw matching line
            M1 = cv2.moments(contours[i])
            M2 = cv2.moments(contours[j])
            if M1['m00'] > 0 and M2['m00'] > 0:
                p1 = (int(M1['m10']/M1['m00']), int(M1['m01']/M1['m00']))
                p2 = (int(M2['m10']/M2['m00']), int(M2['m01']/M2['m00']))
                cv2.line(result, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

cv2.putText(result, 'Red dots = centroids', (10, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
cv2.putText(result, 'Purple arrows = orientation', (300, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

cv2.imshow('Moments & Centroids', result)
```
