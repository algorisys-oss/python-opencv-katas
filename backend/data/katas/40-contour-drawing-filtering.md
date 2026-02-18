---
slug: 40-contour-drawing-filtering
title: Contour Drawing & Filtering
level: intermediate
concepts: [cv2.drawContours, contour filtering, area threshold]
prerequisites: [39-contour-properties]
---

## What Problem Are We Solving?

After finding contours with `cv2.findContours()`, you often end up with dozens or hundreds of them — many are tiny noise contours, irrelevant edges, or fragments you don't care about. You need a way to **draw** the contours you want and **filter out** the ones you don't. `cv2.drawContours()` handles the drawing, and filtering by properties like area, aspect ratio, or perimeter lets you keep only the meaningful contours.

## Drawing Contours with cv2.drawContours

The `cv2.drawContours()` function draws contour outlines (or filled contours) onto an image:

```python
cv2.drawContours(image, contours, contourIdx, color, thickness)
```

| Parameter | Meaning |
|---|---|
| `image` | The image to draw on (modified in place) |
| `contours` | List of contours (from `cv2.findContours`) |
| `contourIdx` | Index of the contour to draw, or `-1` for all |
| `color` | Color as `(B, G, R)` tuple |
| `thickness` | Line thickness, or `-1` (cv2.FILLED) for filled |

```python
# Draw ALL contours in green, 2px thick
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Draw only the first contour in red, filled
cv2.drawContours(img, contours, 0, (0, 0, 255), -1)
```

The `contourIdx` parameter is especially useful when you want to draw contours one at a time, for example with different colors:

```python
for i, cnt in enumerate(contours):
    color = (0, (i * 60) % 256, 255)
    cv2.drawContours(img, contours, i, color, 2)
```

## Filtering Contours by Area

The most common filter is **area**. Tiny contours (a few pixels) are usually noise, and very large contours might be the entire image border:

```python
min_area = 500
filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
```

You can also set an upper bound:

```python
min_area = 500
max_area = 50000
filtered = [cnt for cnt in contours
            if min_area < cv2.contourArea(cnt) < max_area]
```

## Filtering Contours by Aspect Ratio

Aspect ratio helps you find contours of a specific shape category. A near-square object has an aspect ratio close to 1.0, while a long thin object has a high ratio:

```python
def aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return float(w) / h if h > 0 else 0

# Keep only roughly square contours (aspect ratio between 0.7 and 1.3)
square_contours = [cnt for cnt in contours
                   if 0.7 < aspect_ratio(cnt) < 1.3]
```

## Filtering by Multiple Criteria

In practice, you combine multiple filters:

```python
def is_valid_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < 500:
        return False
    x, y, w, h = cv2.boundingRect(cnt)
    ar = float(w) / h if h > 0 else 0
    if ar < 0.3 or ar > 3.0:
        return False
    return True

valid = [cnt for cnt in contours if is_valid_contour(cnt)]
```

## Drawing Specific Contours from a Filtered List

After filtering, you pass the filtered list to `drawContours` with index `-1`:

```python
filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
cv2.drawContours(img, filtered, -1, (0, 255, 0), 2)
```

Or draw each one individually with a unique color:

```python
import random
for cnt in filtered:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(img, [cnt], 0, color, 2)
```

Note that when you pass a single contour wrapped in a list `[cnt]`, you use index `0` to refer to it.

## Tips & Common Mistakes

- `cv2.drawContours` modifies the image **in place**. If you want to keep the original, draw on a copy: `canvas = img.copy()`.
- Setting `contourIdx=-1` draws **all** contours in the list. To draw one, pass its index or wrap it in a list.
- `thickness=-1` fills the contour. This is useful for creating masks from contours.
- `cv2.contourArea()` returns 0 for contours with fewer than 3 points — always check for minimum area, not just non-zero.
- When filtering by aspect ratio, guard against division by zero when `h == 0`.
- Contours found with `RETR_TREE` include hierarchy info. Filtering by area alone may keep child contours (holes) you don't want — consider using hierarchy to filter parent/child relationships.
- Always convert to grayscale and threshold before finding contours, not after.

## Starter Code

```python
import cv2
import numpy as np

# Create a black canvas with various shapes of different sizes
canvas = np.zeros((500, 700, 3), dtype=np.uint8)

# Large shapes (should pass area filter)
cv2.rectangle(canvas, (50, 50), (200, 180), (255, 255, 255), -1)
cv2.circle(canvas, (350, 120), 80, (255, 255, 255), -1)
cv2.ellipse(canvas, (550, 120), (80, 50), 0, 0, 360, (255, 255, 255), -1)

# Medium shapes
cv2.rectangle(canvas, (50, 300), (130, 380), (255, 255, 255), -1)
cv2.circle(canvas, (250, 340), 40, (255, 255, 255), -1)

# Small noise shapes (should be filtered out)
cv2.circle(canvas, (400, 350), 8, (255, 255, 255), -1)
cv2.circle(canvas, (450, 370), 5, (255, 255, 255), -1)
cv2.rectangle(canvas, (500, 340), (515, 355), (255, 255, 255), -1)
cv2.circle(canvas, (550, 380), 3, (255, 255, 255), -1)
cv2.rectangle(canvas, (600, 300), (605, 305), (255, 255, 255), -1)

# Convert to grayscale and threshold
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find all contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'Total contours found: {len(contours)}')

# --- Draw ALL contours (before filtering) ---
all_drawn = canvas.copy()
cv2.drawContours(all_drawn, contours, -1, (0, 255, 0), 2)
cv2.putText(all_drawn, f'All: {len(contours)}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# --- Filter by area (keep only contours > 1000 pixels) ---
min_area = 1000
area_filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
area_drawn = canvas.copy()
cv2.drawContours(area_drawn, area_filtered, -1, (0, 200, 255), 2)
cv2.putText(area_drawn, f'Area > {min_area}: {len(area_filtered)}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# --- Filter by aspect ratio (keep roughly square shapes) ---
def get_aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return float(w) / h if h > 0 else 0

square_filtered = [cnt for cnt in area_filtered
                   if 0.6 < get_aspect_ratio(cnt) < 1.5]
square_drawn = canvas.copy()
for i, cnt in enumerate(square_filtered):
    color = (255, 0, 255) if i % 2 == 0 else (255, 255, 0)
    cv2.drawContours(square_drawn, [cnt], 0, color, 3)
cv2.putText(square_drawn, f'Square-ish: {len(square_filtered)}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Print info about each contour
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    ar = get_aspect_ratio(cnt)
    label = 'KEPT' if area > min_area else 'filtered out'
    print(f'Contour {i}: area={area:.0f}, aspect_ratio={ar:.2f} -> {label}')

# Stack results for comparison
top_row = np.hstack([all_drawn, area_drawn])
# Pad square_drawn to match width
pad = np.zeros_like(square_drawn)
cv2.putText(pad, 'Original', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
pad[:] = canvas
cv2.putText(pad, 'Original', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
bot_row = np.hstack([square_drawn, pad])
result = np.vstack([top_row, bot_row])

cv2.imshow('Contour Drawing & Filtering', result)
```
