---
slug: 39-contour-properties
title: Contour Properties
level: intermediate
concepts: [cv2.contourArea, cv2.arcLength, cv2.boundingRect]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

Finding contours gives you the boundary points of each object. But to actually **identify, classify, or filter** objects, you need to measure their properties: How big is it? How round? How compact? OpenCV provides functions to compute **area, perimeter, bounding rectangles**, and derived ratios that let you distinguish circles from rectangles, filter by size, and analyze shapes.

## Area

The area is the number of pixels enclosed by the contour:

```python
area = cv2.contourArea(contour)
```

This computes the area using Green's theorem (works for any shape, not just convex). Use it to filter out small noise contours or find the largest object.

## Perimeter (Arc Length)

The perimeter is the total length of the contour boundary:

```python
perimeter = cv2.arcLength(contour, closed=True)
```

Set `closed=True` for closed contours (which is almost always the case). The perimeter increases with both size and complexity — a jagged shape has a longer perimeter than a smooth one of the same area.

## Bounding Rectangle (Upright)

The smallest axis-aligned rectangle that contains the contour:

```python
x, y, w, h = cv2.boundingRect(contour)
```

Returns the top-left corner `(x, y)` and dimensions `(w, h)`. This is the simplest way to locate an object.

```python
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

## Rotated (Minimum Area) Bounding Rectangle

The tightest-fitting rectangle, rotated to match the contour's orientation:

```python
rect = cv2.minAreaRect(contour)  # Returns ((cx, cy), (w, h), angle)
box = cv2.boxPoints(rect)        # Get 4 corner points
box = np.int32(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
```

## Aspect Ratio

The ratio of width to height of the bounding rectangle:

```python
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = float(w) / h
```

- `aspect_ratio ~ 1.0` means roughly square
- `aspect_ratio > 1.0` means wider than tall
- `aspect_ratio < 1.0` means taller than wide

## Extent

The ratio of contour area to bounding rectangle area — how much of the bounding box the object fills:

```python
area = cv2.contourArea(contour)
x, y, w, h = cv2.boundingRect(contour)
extent = float(area) / (w * h)
```

- A filled rectangle has `extent ~ 1.0`
- A circle has `extent ~ 0.785` (pi/4)
- An irregular shape has lower extent

## Solidity

The ratio of contour area to its **convex hull** area — how "solid" (non-concave) the shape is:

```python
area = cv2.contourArea(contour)
hull = cv2.convexHull(contour)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area
```

- A convex shape (circle, rectangle) has `solidity ~ 1.0`
- A star or crescent has lower solidity (lots of concavities)

## Minimum Enclosing Circle

The smallest circle that contains the entire contour:

```python
(cx, cy), radius = cv2.minEnclosingCircle(contour)
cv2.circle(img, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
```

## Tips & Common Mistakes

- `cv2.contourArea()` can return 0 for very small contours (fewer than 3 points). Always check before dividing.
- `cv2.arcLength()` requires the `closed` parameter. Forgetting it causes a TypeError.
- The bounding rectangle from `cv2.boundingRect()` is always **axis-aligned** (not rotated). Use `cv2.minAreaRect()` for a tighter fit on rotated objects.
- Aspect ratio, extent, and solidity are **dimensionless ratios** — they don't change with scale, making them useful for shape classification.
- Always filter contours by minimum area first to avoid division-by-zero errors and skip noise.
- `cv2.boxPoints()` returns floating-point coordinates. Convert to `np.int32` before drawing.

## Starter Code

```python
import cv2
import numpy as np

# Create image with different shapes
img = np.zeros((450, 700, 3), dtype=np.uint8)
img[:] = (40, 40, 40)

# Draw various shapes
# 1. Rectangle
cv2.rectangle(img, (30, 30), (160, 130), (255, 255, 255), -1)

# 2. Circle
cv2.circle(img, (270, 80), 60, (255, 255, 255), -1)

# 3. Triangle
tri_pts = np.array([[400, 130], [470, 30], [540, 130]], np.int32)
cv2.fillPoly(img, [tri_pts], (255, 255, 255))

# 4. Ellipse (rotated)
cv2.ellipse(img, (640, 80), (50, 30), 30, 0, 360, (255, 255, 255), -1)

# 5. Star shape (low solidity)
star_pts = []
cx_star, cy_star = 100, 300
for i in range(10):
    angle = i * 36 * np.pi / 180 - np.pi / 2
    r = 55 if i % 2 == 0 else 25
    star_pts.append([int(cx_star + r * np.cos(angle)), int(cy_star + r * np.sin(angle))])
star_pts = np.array(star_pts, np.int32)
cv2.fillPoly(img, [star_pts], (255, 255, 255))

# 6. L-shape (low extent)
l_shape = np.array([[250, 200], [320, 200], [320, 280], [290, 280],
                     [290, 230], [250, 230]], np.int32)
cv2.fillPoly(img, [l_shape], (255, 255, 255))

# 7. Irregular blob
blob_pts = np.array([[420, 250], [500, 220], [560, 260], [580, 340],
                      [520, 380], [440, 360], [400, 310]], np.int32)
cv2.fillPoly(img, [blob_pts], (255, 255, 255))

# Convert to grayscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Analyze each contour
result = img.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

print(f'{"#":<4} {"Area":<8} {"Perim":<8} {"Aspect":<8} {"Extent":<8} {"Solidity":<8}')
print('-' * 48)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 50:
        continue

    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = float(area) / (w * h) if w * h > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    print(f'{i:<4} {area:<8.0f} {perimeter:<8.1f} {aspect_ratio:<8.2f} {extent:<8.3f} {solidity:<8.3f}')

    # Draw bounding rectangle (green)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Draw minimum enclosing circle (blue)
    (mcx, mcy), radius = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(mcx), int(mcy)), int(radius), (255, 0, 0), 1)

    # Draw rotated bounding rectangle (red)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(result, [box], 0, (0, 0, 255), 1)

    # Draw contour outline (yellow)
    cv2.drawContours(result, [cnt], 0, (0, 255, 255), 2)

    # Draw centroid
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        mcx_m = int(M['m10'] / M['m00'])
        mcy_m = int(M['m01'] / M['m00'])
        cv2.circle(result, (mcx_m, mcy_m), 4, (0, 0, 255), -1)

    # Label with properties
    label = f'A={area:.0f} E={extent:.2f} S={solidity:.2f}'
    cv2.putText(result, label, (x, y - 8), font, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

# Add legend
cv2.putText(result, 'Yellow=contour Green=bbox Red=rotated Blue=enclosing', (10, 440),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imshow('Contour Properties', result)
```
