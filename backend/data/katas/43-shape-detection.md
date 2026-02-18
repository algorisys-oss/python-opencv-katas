---
slug: 43-shape-detection
title: Shape Detection
level: intermediate
concepts: [shape classification, vertex counting, circularity]
prerequisites: [42-contour-approximation, 39-contour-properties]
---

## What Problem Are We Solving?

You have a binary image with various shapes, and you need to identify what each shape **is** — triangle, rectangle, pentagon, or circle. This is a fundamental building block for tasks like document analysis (detecting arrows, checkboxes, icons), robotics (identifying objects by shape), and quality inspection (verifying part geometry). The approach combines **contour approximation** (to count vertices) with **circularity** (to distinguish circles from polygons).

## The Shape Classification Strategy

The core idea is simple:

1. Find contours in the image.
2. Approximate each contour with `cv2.approxPolyDP()` to reduce it to its essential vertices.
3. Count the number of vertices in the approximation.
4. Use the vertex count to classify the shape.

```python
perimeter = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
vertices = len(approx)
```

The mapping from vertex count to shape:

| Vertices | Shape |
|---|---|
| 3 | Triangle |
| 4 | Rectangle or Square |
| 5 | Pentagon |
| 6 | Hexagon |
| 7+ | Circle (or complex polygon) |

## Distinguishing Rectangles from Squares

When the vertex count is 4, you have a quadrilateral. To distinguish a square from a rectangle, check the aspect ratio of its bounding box:

```python
if vertices == 4:
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    if 0.9 < aspect_ratio < 1.1:
        shape = 'Square'
    else:
        shape = 'Rectangle'
```

## Using Circularity to Detect Circles

Counting vertices alone isn't reliable for circles — a circle approximated at 4% epsilon might yield 8-12 vertices, which overlaps with an octagon or decagon. **Circularity** provides a more robust test:

```python
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)
circularity = (4 * np.pi * area) / (perimeter * perimeter)
```

Circularity is 1.0 for a perfect circle and decreases for less circular shapes:

| Shape | Typical Circularity |
|---|---|
| Circle | 0.85 - 1.0 |
| Square | ~0.78 |
| Triangle | ~0.60 |
| Star | < 0.40 |

So the check becomes:

```python
if circularity > 0.80 and vertices > 6:
    shape = 'Circle'
```

## A Complete Shape Classifier

Putting it all together into a reusable function:

```python
def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)

    # Circularity check
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    if vertices == 3:
        return 'Triangle'
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(w) / h
        return 'Square' if 0.9 < ar < 1.1 else 'Rectangle'
    elif vertices == 5:
        return 'Pentagon'
    elif circularity > 0.80:
        return 'Circle'
    else:
        return f'Polygon ({vertices})'
```

## Why 4% Epsilon?

The epsilon percentage for `approxPolyDP` matters a lot for shape detection:

- **Too low (1%)**: A rectangle might keep 6-8 points instead of 4, misclassified as a hexagon.
- **Too high (10%)**: A pentagon might collapse to 4 points, misclassified as a rectangle.
- **4% is the sweet spot**: Polygons reduce to their true vertex count, while circles retain enough points to be distinguished by circularity.

```python
# This is the standard epsilon for shape detection
epsilon = 0.04 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

## Handling Rotated Shapes

The classification is rotation-invariant because both vertex counting and circularity are properties of the shape geometry, not its orientation. A rotated triangle still has 3 vertices after approximation. A tilted rectangle still has 4 vertices and an aspect ratio check on its **minimum area bounding rectangle** is more accurate than using the upright bounding box:

```python
if vertices == 4:
    rect = cv2.minAreaRect(approx)
    w, h = rect[1]
    ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    shape = 'Square' if ar < 1.15 else 'Rectangle'
```

## Tips & Common Mistakes

- Always filter out tiny contours (by area) before classifying shapes. Noise contours will produce meaningless results.
- The 4% epsilon works for clean, well-separated shapes. For noisy images, you may need to adjust or add smoothing.
- Circularity can exceed 1.0 for very small contours due to discretization. Clamp or handle this case.
- Do not rely solely on vertex count for circles. A regular octagon also has 8 vertices. Use circularity as the primary circle test.
- Squares are a special case of rectangles. Always check aspect ratio when you get 4 vertices.
- For real-world applications, add a minimum area threshold to filter out noise contours before classification.
- `cv2.approxPolyDP` with `closed=True` is essential for closed shape detection.

## Starter Code

```python
import cv2
import numpy as np

# Create canvas with various shapes
canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# Triangle
tri_pts = np.array([[100, 50], [50, 180], [150, 180]], dtype=np.int32)
cv2.fillPoly(canvas, [tri_pts], (255, 255, 255))

# Square
cv2.rectangle(canvas, (220, 60), (340, 180), (255, 255, 255), -1)

# Rectangle
cv2.rectangle(canvas, (400, 70), (580, 170), (255, 255, 255), -1)

# Pentagon
pent_pts = []
cx, cy, r = 680, 120, 70
for i in range(5):
    angle = i * 72 - 90
    x = int(cx + r * np.cos(np.radians(angle)))
    y = int(cy + r * np.sin(np.radians(angle)))
    pent_pts.append([x, y])
pent_pts = np.array(pent_pts, dtype=np.int32)
cv2.fillPoly(canvas, [pent_pts], (255, 255, 255))

# Circle
cv2.circle(canvas, (100, 350), 70, (255, 255, 255), -1)

# Ellipse (will test circularity)
cv2.ellipse(canvas, (280, 350), (90, 50), 0, 0, 360, (255, 255, 255), -1)

# Rotated square (diamond)
diamond_pts = np.array([[480, 270], [550, 350], [480, 430], [410, 350]], dtype=np.int32)
cv2.fillPoly(canvas, [diamond_pts], (255, 255, 255))

# Small circle
cv2.circle(canvas, (670, 350), 50, (255, 255, 255), -1)

# --- Shape classification function ---
def classify_shape(cnt):
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(cnt)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    if vertices == 3:
        return 'Triangle', vertices, circularity
    elif vertices == 4:
        rect = cv2.minAreaRect(approx)
        w, h = rect[1]
        ar = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if ar < 1.15:
            return 'Square', vertices, circularity
        else:
            return 'Rectangle', vertices, circularity
    elif vertices == 5:
        return 'Pentagon', vertices, circularity
    elif circularity > 0.80:
        return 'Circle', vertices, circularity
    else:
        return f'Polygon-{vertices}', vertices, circularity

# Find contours
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Classify and annotate each shape
result = canvas.copy()
colors = {
    'Triangle': (0, 255, 0),
    'Square': (255, 0, 0),
    'Rectangle': (255, 150, 0),
    'Pentagon': (0, 200, 255),
    'Circle': (255, 0, 255),
}

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue

    shape_name, verts, circ = classify_shape(cnt)
    color = colors.get(shape_name, (200, 200, 200))

    # Draw contour
    cv2.drawContours(result, [cnt], 0, color, 2)

    # Draw approximated vertices
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    for pt in approx:
        cv2.circle(result, tuple(pt[0]), 4, (0, 0, 255), -1)

    # Label the shape
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(result, shape_name, (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(result, f'v={verts} c={circ:.2f}', (cx - 40, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    print(f'{shape_name:12s} | vertices={verts:2d} | circularity={circ:.3f} | area={area:.0f}')

cv2.imshow('Shape Detection', result)
```
