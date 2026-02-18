---
slug: 42-contour-approximation
title: Contour Approximation
level: intermediate
concepts: [cv2.approxPolyDP, epsilon, shape simplification]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

Contours extracted from images often have far more points than necessary. A circle might be represented by hundreds of closely spaced points, and a rectangle might have jagged edges with dozens of tiny segments. **Contour approximation** simplifies a contour by reducing it to fewer points while preserving its essential shape. This is critical for shape recognition (a rectangle should have 4 vertices, not 400) and for reducing computational cost in downstream processing.

## The Douglas-Peucker Algorithm

OpenCV uses the **Douglas-Peucker** (also called Ramer-Douglas-Peucker) algorithm for contour approximation. The idea is intuitive:

1. Draw a straight line from the first point to the last point of the contour.
2. Find the contour point that is farthest from this line.
3. If that distance is greater than a threshold (**epsilon**), keep the point and recursively simplify each sub-segment.
4. If the distance is less than epsilon, all the points between can be discarded — the straight line is "close enough."

The result is a polygon with far fewer vertices that still approximates the original shape within the epsilon tolerance.

## Using cv2.approxPolyDP

```python
approx = cv2.approxPolyDP(contour, epsilon, closed)
```

| Parameter | Meaning |
|---|---|
| `contour` | Input contour (array of points) |
| `epsilon` | Maximum distance between the original contour and the approximation |
| `closed` | `True` if the contour is closed (which it usually is) |

The returned `approx` is a simplified contour with fewer points.

```python
epsilon = 10.0
approx = cv2.approxPolyDP(contour, epsilon, True)
print(f'Original: {len(contour)} points -> Approximated: {len(approx)} points')
```

## Choosing Epsilon Relative to Arc Length

The key challenge is choosing a good epsilon value. Too small, and the approximation keeps too many points. Too large, and the shape gets distorted beyond recognition.

The standard approach is to set epsilon as a **percentage of the contour's arc length** (perimeter):

```python
perimeter = cv2.arcLength(contour, True)
epsilon = 0.02 * perimeter  # 2% of perimeter
approx = cv2.approxPolyDP(contour, epsilon, True)
```

Common epsilon percentages and their effects:

| Epsilon (% of perimeter) | Effect |
|---|---|
| `0.01` (1%) | Very close to original, many points retained |
| `0.02` (2%) | Good default — preserves shape, removes noise |
| `0.04` (4%) | Aggressive simplification — may lose subtle features |
| `0.10` (10%) | Very aggressive — only the coarsest shape remains |

For shape detection (triangle, rectangle, pentagon), 2-4% typically works well.

## How Point Reduction Works in Practice

Consider a circle contour with 200 points:

```python
# Original circle contour: ~200 points
perimeter = cv2.arcLength(circle_contour, True)

# 1% epsilon: maybe 30 points (still looks like a circle)
approx1 = cv2.approxPolyDP(circle_contour, 0.01 * perimeter, True)

# 5% epsilon: maybe 8-10 points (looks like an octagon)
approx5 = cv2.approxPolyDP(circle_contour, 0.05 * perimeter, True)

# 15% epsilon: maybe 4 points (looks like a diamond/square)
approx15 = cv2.approxPolyDP(circle_contour, 0.15 * perimeter, True)
```

For a rectangle contour, even 2-4% epsilon will reduce it to exactly 4 points — the corners.

## Visualizing the Approximation

Drawing the approximated contour alongside the original makes the effect clear:

```python
# Draw original in green
cv2.drawContours(img, [contour], 0, (0, 255, 0), 1)

# Draw approximation in red with vertex dots
cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
for pt in approx:
    cv2.circle(img, tuple(pt[0]), 4, (255, 0, 0), -1)
```

## Tips & Common Mistakes

- Always set `closed=True` for contours from `cv2.findContours()` — they are closed by definition.
- Use `cv2.arcLength(contour, True)` to compute the perimeter. The second argument `True` indicates the contour is closed.
- Start with `epsilon = 0.02 * perimeter` and adjust. For shape classification, 2-4% is the sweet spot.
- A circle approximated with 2% epsilon might yield 10-15 vertices, not 0. Circles are detected by high vertex count combined with circularity checks, not by approximation alone.
- `cv2.approxPolyDP` returns the same array format as contours, so you can pass the result directly to `cv2.drawContours`, `cv2.contourArea`, etc.
- The number of vertices in the approximated contour is the basis for shape detection: 3 = triangle, 4 = rectangle, 5 = pentagon, many = circle.
- Epsilon is in **pixel units**, not a percentage. You compute the percentage yourself by multiplying by the arc length.

## Starter Code

```python
import cv2
import numpy as np

# Create a canvas with different shapes
canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# Draw a star (complex shape with many points)
star_pts = []
cx, cy, outer_r, inner_r = 150, 150, 100, 40
for i in range(10):
    angle = i * 36 - 90  # 36 degrees apart, start from top
    r = outer_r if i % 2 == 0 else inner_r
    x = int(cx + r * np.cos(np.radians(angle)))
    y = int(cy + r * np.sin(np.radians(angle)))
    star_pts.append([x, y])
star_pts = np.array(star_pts, dtype=np.int32)
cv2.fillPoly(canvas, [star_pts], (255, 255, 255))

# Draw a circle
cv2.circle(canvas, (400, 150), 90, (255, 255, 255), -1)

# Draw a rounded rectangle (many contour points)
cv2.rectangle(canvas, (560, 60), (750, 240), (255, 255, 255), -1)

# Draw an irregular blob
blob_pts = np.array([
    [100, 350], [150, 310], [220, 330], [280, 300],
    [330, 340], [350, 400], [320, 450], [250, 470],
    [180, 460], [120, 430], [80, 390]
], dtype=np.int32)
cv2.fillPoly(canvas, [blob_pts], (255, 255, 255))

# Draw a triangle
tri_pts = np.array([[550, 430], [650, 300], [750, 430]], dtype=np.int32)
cv2.fillPoly(canvas, [tri_pts], (255, 255, 255))

# Find contours
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Compare different epsilon values ---
epsilon_pcts = [0.01, 0.02, 0.05, 0.10]
results = []

for pct in epsilon_pcts:
    display = canvas.copy()
    total_original = 0
    total_approx = 0

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = pct * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        total_original += len(cnt)
        total_approx += len(approx)

        # Draw approximated contour in red
        cv2.drawContours(display, [approx], 0, (0, 0, 255), 2)

        # Draw vertices as blue dots
        for pt in approx:
            cv2.circle(display, tuple(pt[0]), 4, (255, 100, 0), -1)

    cv2.putText(display, f'eps={pct:.0%} ({total_approx} pts)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    results.append(display)

    print(f'Epsilon = {pct:.0%} of perimeter: {total_original} -> {total_approx} points')

# Print per-contour details for the 2% case
print('\nDetailed breakdown (epsilon = 2%):')
for i, cnt in enumerate(contours):
    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    area = cv2.contourArea(cnt)
    print(f'  Contour {i}: area={area:.0f}, perimeter={perimeter:.0f}, '
          f'{len(cnt)} pts -> {len(approx)} vertices')

# Stack results: 2x2 grid
top_row = np.hstack([results[0], results[1]])
bot_row = np.hstack([results[2], results[3]])
result = np.vstack([top_row, bot_row])

cv2.imshow('Contour Approximation', result)
```
