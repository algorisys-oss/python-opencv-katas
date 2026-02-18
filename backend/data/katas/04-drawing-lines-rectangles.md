---
slug: 04-drawing-lines-rectangles
title: Drawing Lines & Rectangles
level: beginner
concepts: [cv2.line, cv2.rectangle, thickness, color tuples]
prerequisites: [01-image-loading, 03-pixel-access]
---

## What Problem Are We Solving?

In the previous kata, you drew rectangles by setting pixel regions directly (`img[50:150, 50:200] = color`). That works, but it's limited — you can't draw **lines**, **outlined shapes**, or control **line thickness**.

OpenCV provides dedicated drawing functions that handle all of this. They draw directly onto the image (they **modify it in place**).

## Drawing a Line

```python
cv2.line(img, pt1, pt2, color, thickness)
```

| Parameter | Meaning |
|---|---|
| `img` | The image to draw on (modified in place) |
| `pt1` | Start point as `(x, y)` — **not** `(y, x)`! |
| `pt2` | End point as `(x, y)` |
| `color` | BGR tuple, e.g. `(0, 0, 255)` for red |
| `thickness` | Line width in pixels (default: 1) |

> **Coordinate trap:** NumPy indexing uses `[y, x]` (row, col), but all OpenCV drawing functions use `(x, y)` (column, row). This is the #1 source of confusion for beginners. Drawing = `(x, y)`. Array access = `[y, x]`.

## Drawing a Rectangle

```python
cv2.rectangle(img, pt1, pt2, color, thickness)
```

- `pt1` = top-left corner `(x, y)`
- `pt2` = bottom-right corner `(x, y)`
- `thickness = -1` fills the rectangle solid

```python
# Outlined rectangle (2px border)
cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), 2)

# Filled rectangle
cv2.rectangle(img, (250, 50), (350, 150), (255, 0, 0), -1)
```

## Understanding Thickness

| Value | Effect |
|---|---|
| `1` | Thin line (default) |
| `2`, `3`, ... | Thicker lines |
| `-1` | **Filled** shape (only for closed shapes like rectangles, circles) |

Thickness only applies to **outlines**. For `cv2.line`, there's no concept of "filled" — it's always a stroke.

## Line Types (Anti-Aliasing)

OpenCV drawing functions accept an optional `lineType` parameter:

```python
cv2.line(img, (0, 0), (300, 200), (255, 255, 255), 2, cv2.LINE_AA)
```

| Line Type | Effect |
|---|---|
| `cv2.LINE_8` | 8-connected line (default, slightly jagged) |
| `cv2.LINE_4` | 4-connected line (more jagged) |
| `cv2.LINE_AA` | Anti-aliased line (smooth, but slower) |

Use `cv2.LINE_AA` when visual quality matters (e.g., annotations). Use the default when speed matters (e.g., real-time processing).

## In-Place Modification

All OpenCV drawing functions **modify the image directly**. They do not return a new image.

```python
# This modifies img — there's no return value to capture
cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 2)
```

If you want to keep the original, draw on a copy:

```python
annotated = img.copy()
cv2.line(annotated, (0, 0), (100, 100), (255, 0, 0), 2)
```

## Tips & Common Mistakes

- Drawing coordinates are `(x, y)`, but array indexing is `[y, x]`. Never mix them up.
- `thickness=-1` fills the shape. This only works for closed shapes (rectangles, circles), not lines.
- Colors are **BGR** tuples: `(Blue, Green, Red)`. `(0, 0, 255)` = red, not blue.
- Drawing modifies the image **in place**. Use `.copy()` if you need the original.
- Coordinates must be **integers**. If you compute floating-point positions, cast with `int()`.

## Starter Code

```python
import cv2
import numpy as np

# Create a dark canvas
img = np.zeros((400, 500, 3), dtype=np.uint8)
img[:] = (30, 30, 30)  # Dark gray background

# --- Lines ---
# Diagonal line: top-left to bottom-right (white, 2px)
cv2.line(img, (20, 20), (200, 180), (255, 255, 255), 2)

# Thick red line
cv2.line(img, (20, 50), (200, 50), (0, 0, 255), 4)

# Anti-aliased green diagonal
cv2.line(img, (20, 80), (200, 200), (0, 255, 0), 2, cv2.LINE_AA)

# --- Outlined Rectangles ---
# Blue rectangle (2px outline)
cv2.rectangle(img, (250, 20), (400, 120), (255, 100, 0), 2)

# Green rectangle (1px outline)
cv2.rectangle(img, (250, 140), (400, 200), (0, 255, 0), 1)

# --- Filled Rectangles ---
# Solid yellow rectangle (thickness = -1)
cv2.rectangle(img, (250, 220), (400, 300), (0, 255, 255), -1)

# --- Combine: filled background + outlined border ---
cv2.rectangle(img, (50, 250), (200, 350), (80, 50, 20), -1)   # Fill
cv2.rectangle(img, (50, 250), (200, 350), (255, 150, 50), 2)  # Border

# --- Grid lines ---
for x in range(0, 500, 50):
    cv2.line(img, (x, 370), (x, 400), (80, 80, 80), 1)

print(f'Canvas size: {img.shape[1]}x{img.shape[0]}')
print(f'Drew lines, outlined rects, filled rects, and a grid')

cv2.imshow('Lines & Rectangles', img)
```
