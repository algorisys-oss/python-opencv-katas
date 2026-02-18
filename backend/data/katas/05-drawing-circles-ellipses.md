---
slug: 05-drawing-circles-ellipses
title: Drawing Circles & Ellipses
level: beginner
concepts: [cv2.circle, cv2.ellipse, filled shapes, angle parameters]
prerequisites: [04-drawing-lines-rectangles]
---

## What Problem Are We Solving?

Lines and rectangles are all straight edges. To draw **round shapes** — circles, ellipses, arcs — you need different functions. These are essential for highlighting detected objects (faces, eyes, coins) and creating annotations.

## Drawing a Circle

```python
cv2.circle(img, center, radius, color, thickness)
```

| Parameter | Meaning |
|---|---|
| `center` | Center point as `(x, y)` |
| `radius` | Radius in pixels |
| `color` | BGR tuple |
| `thickness` | Line width, or `-1` for filled |

```python
# Outlined circle
cv2.circle(img, (200, 150), 80, (0, 255, 0), 2)

# Filled circle
cv2.circle(img, (200, 150), 40, (0, 0, 255), -1)
```

> **When would you use this?** Face detection returns a bounding box — you might draw a circle around the face center. Coin detection uses HoughCircles, which returns center + radius directly.

## Drawing an Ellipse

An ellipse is a stretched circle. It needs more parameters:

```python
cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
```

| Parameter | Meaning |
|---|---|
| `center` | Center point `(x, y)` |
| `axes` | Half-widths as `(half_w, half_h)` — **not** full width/height |
| `angle` | Rotation of the ellipse in degrees (clockwise) |
| `startAngle` | Arc start in degrees (0 = right, 90 = bottom) |
| `endAngle` | Arc end in degrees (360 = full ellipse) |
| `color` | BGR tuple |
| `thickness` | Line width, or `-1` for filled |

```python
# Full ellipse (0 to 360)
cv2.ellipse(img, (250, 200), (100, 50), 0, 0, 360, (255, 0, 0), 2)

# Rotated ellipse (tilted 45 degrees)
cv2.ellipse(img, (250, 200), (100, 50), 45, 0, 360, (0, 255, 255), 2)
```

## Drawing Arcs (Partial Ellipses)

By changing `startAngle` and `endAngle`, you draw arcs instead of full ellipses:

```python
# Top half of an ellipse (0° to 180°)
cv2.ellipse(img, (200, 200), (80, 40), 0, 0, 180, (255, 255, 0), 2)

# Quarter arc (0° to 90°)
cv2.ellipse(img, (200, 200), (80, 40), 0, 0, 90, (0, 255, 0), 3)
```

The angle system: **0°** is the 3 o'clock position (right). Angles increase **clockwise** (because y-axis points downward in images).

```
        270°
         |
  180° --+-- 0°
         |
        90°
```

## Axes vs Diameter

A common mistake: `axes` is `(half_width, half_height)`, not `(width, height)`.

```python
# An ellipse that spans 200px wide and 100px tall:
cv2.ellipse(img, center, (100, 50), ...)  # axes = half of each dimension
```

## Tips & Common Mistakes

- `center` and all points use `(x, y)` — same as other drawing functions.
- `axes` is **half-size**, not full size. An axis of `(100, 50)` draws 200px wide, 100px tall.
- `thickness=-1` fills the shape, just like rectangles.
- `angle` rotates the entire ellipse. `startAngle`/`endAngle` control which arc segment to draw.
- Angles are in **degrees** (not radians), and go **clockwise** from the 3 o'clock position.
- A circle is just an ellipse where both axes are equal: `cv2.ellipse(img, center, (r, r), 0, 0, 360, ...)`.

## Starter Code

```python
import cv2
import numpy as np

# Create a dark canvas
img = np.zeros((400, 500, 3), dtype=np.uint8)
img[:] = (30, 30, 30)

# --- Circles ---
# Outlined circle (green, 2px)
cv2.circle(img, (100, 100), 60, (0, 255, 0), 2)

# Filled circle (red)
cv2.circle(img, (100, 100), 25, (0, 0, 255), -1)

# Concentric circles
for r in range(20, 100, 15):
    cv2.circle(img, (300, 100), r, (255, 200, 0), 1, cv2.LINE_AA)

# --- Ellipses ---
# Horizontal ellipse (wide)
cv2.ellipse(img, (100, 280), (80, 40), 0, 0, 360, (255, 100, 100), 2)

# Rotated ellipse (tilted 30°)
cv2.ellipse(img, (300, 250), (80, 40), 30, 0, 360, (100, 255, 100), 2)

# Filled ellipse
cv2.ellipse(img, (300, 350), (60, 30), -20, 0, 360, (100, 100, 255), -1)

# --- Arcs (partial ellipses) ---
# Top half arc
cv2.ellipse(img, (100, 380), (70, 35), 0, 180, 360, (255, 255, 0), 2)

# Pac-Man shape (filled, mouth open from 30° to 330°)
cv2.ellipse(img, (420, 200), (50, 50), 0, 30, 330, (0, 255, 255), -1)

print(f'Canvas: {img.shape[1]}x{img.shape[0]}')
print('Drew circles, ellipses, concentric rings, arcs, and Pac-Man')

cv2.imshow('Circles & Ellipses', img)
```
