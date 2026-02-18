---
slug: 45-bounding-shapes
title: Bounding Shapes
level: intermediate
concepts: [cv2.boundingRect, cv2.minAreaRect, cv2.minEnclosingCircle, cv2.fitEllipse]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

Once you have a contour, you often need to enclose it in a simple geometric shape — a rectangle for cropping, a circle for distance calculations, or an ellipse for orientation analysis. OpenCV provides several bounding shape functions, each suited to different use cases. The **upright bounding rectangle** is fast but wastes space on rotated objects. The **minimum area rotated rectangle** fits tightly. The **minimum enclosing circle** gives a rotation-invariant bound. The **fitted ellipse** captures orientation and eccentricity.

## Upright Bounding Rectangle

`cv2.boundingRect()` returns the smallest axis-aligned (upright) rectangle that contains the contour:

```python
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

| Return Value | Meaning |
|---|---|
| `x, y` | Top-left corner of the rectangle |
| `w, h` | Width and height |

This is the fastest bounding shape to compute and is commonly used for:
- Cropping a region of interest: `roi = img[y:y+h, x:x+w]`
- Computing aspect ratio: `ar = w / h`
- Quick overlap checks between objects

The downside is that for rotated objects, the upright rectangle is much larger than necessary.

## Minimum Area Rotated Rectangle

`cv2.minAreaRect()` finds the smallest rectangle (at any rotation) that encloses the contour:

```python
rect = cv2.minAreaRect(contour)
# rect = ((center_x, center_y), (width, height), angle)
```

To draw it, convert to integer corner points:

```python
box = cv2.boxPoints(rect)
box = np.int32(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
```

The returned `rect` is a tuple: `((cx, cy), (w, h), angle)`.
- `(cx, cy)` — center of the rectangle
- `(w, h)` — width and height (note: which is "width" vs "height" depends on the angle)
- `angle` — rotation angle in degrees

This is ideal for measuring the true extent of rotated objects and for computing the rotation angle.

## Minimum Enclosing Circle

`cv2.minEnclosingCircle()` finds the smallest circle that fully contains the contour:

```python
(cx, cy), radius = cv2.minEnclosingCircle(contour)
center = (int(cx), int(cy))
radius = int(radius)
cv2.circle(img, center, radius, (255, 0, 0), 2)
```

This is useful when you need a rotation-invariant bounding shape, or when computing distances between objects (the radius gives a natural "size" measure).

## Fitted Ellipse

`cv2.fitEllipse()` fits an ellipse to the contour using a least-squares approach. The contour must have at least 5 points:

```python
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(img, ellipse, (255, 255, 0), 2)
```

The returned `ellipse` is: `((cx, cy), (major_axis, minor_axis), angle)`.

Unlike the other bounding shapes, the fitted ellipse does **not** necessarily enclose the entire contour — it's a best-fit approximation. This makes it useful for:
- Determining the orientation of elongated objects
- Computing eccentricity: `ecc = minor / major`
- Fitting to noisy or irregular contours

## Comparing All Four Methods

Each bounding shape has different characteristics:

| Method | Rotation-aware? | Tight fit? | Points needed | Use case |
|---|---|---|---|---|
| `boundingRect` | No (axis-aligned) | Loose for rotated shapes | Any | Cropping, ROI |
| `minAreaRect` | Yes | Tight | Any | Rotation measurement |
| `minEnclosingCircle` | N/A (rotation-invariant) | Moderate | Any | Distance/size checks |
| `fitEllipse` | Yes | Best-fit (not enclosing) | >= 5 | Orientation, eccentricity |

## Extracting Useful Measurements

```python
# From upright rect: aspect ratio and extent
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = float(w) / h
extent = cv2.contourArea(contour) / (w * h)  # How much of the rect is filled

# From rotated rect: true dimensions and rotation
rect = cv2.minAreaRect(contour)
true_w, true_h = rect[1]
rotation = rect[2]

# From enclosing circle: equivalent radius
(cx, cy), radius = cv2.minEnclosingCircle(contour)

# From ellipse: eccentricity and orientation
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    major = max(ellipse[1])
    minor = min(ellipse[1])
    eccentricity = minor / major if major > 0 else 0
```

## Tips & Common Mistakes

- `cv2.boundingRect` returns `(x, y, w, h)` as integers. The other functions return floats that you need to convert with `int()` for drawing.
- `cv2.minAreaRect` returns `(width, height)` where width is not necessarily the longer side. Use `max()` and `min()` if you need the major/minor axis.
- `cv2.boxPoints()` (or `cv2.cv.BoxPoints` in older versions) converts the rotated rect to 4 corner points for drawing.
- `cv2.fitEllipse` requires at least **5 points** in the contour. Always check `len(contour) >= 5` before calling it.
- The fitted ellipse may **not** enclose the contour — it's a best-fit, not a bounding shape. Don't use it for containment checks.
- The angle returned by `cv2.minAreaRect` is relative to the x-axis and can be confusing — it ranges from -90 to 0 degrees in some OpenCV versions.
- For axis-aligned objects, `boundingRect` and `minAreaRect` give nearly identical results. The difference becomes significant with rotation.

## Starter Code

```python
import cv2
import numpy as np

# Create canvas with rotated and irregular shapes
canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# Rotated rectangle (drawn as a polygon)
center = (150, 150)
size = (180, 80)
angle = 35
rect_pts = cv2.boxPoints(((center), size, angle))
rect_pts = np.int32(rect_pts)
cv2.fillPoly(canvas, [rect_pts], (255, 255, 255))

# Irregular blob
blob_pts = np.array([
    [400, 60], [480, 80], [520, 140], [500, 200],
    [450, 230], [380, 210], [340, 160], [350, 100]
], dtype=np.int32)
cv2.fillPoly(canvas, [blob_pts], (255, 255, 255))

# Elongated ellipse
cv2.ellipse(canvas, (680, 150), (90, 35), -25, 0, 360, (255, 255, 255), -1)

# L-shape (non-convex)
l_pts = np.array([
    [50, 300], [170, 300], [170, 350],
    [100, 350], [100, 470], [50, 470]
], dtype=np.int32)
cv2.fillPoly(canvas, [l_pts], (255, 255, 255))

# Circle
cv2.circle(canvas, (320, 390), 70, (255, 255, 255), -1)

# Star shape
star_pts = []
scx, scy = 550, 390
for i in range(10):
    a = i * 36 - 90
    r = 80 if i % 2 == 0 else 35
    sx = int(scx + r * np.cos(np.radians(a)))
    sy = int(scy + r * np.sin(np.radians(a)))
    star_pts.append([sx, sy])
cv2.fillPoly(canvas, [np.array(star_pts, dtype=np.int32)], (255, 255, 255))

# Find contours
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Draw all four bounding shapes for each contour ---
result = canvas.copy()

colors = {
    'boundingRect': (0, 255, 0),       # Green
    'minAreaRect': (0, 0, 255),         # Red
    'minEnclosingCircle': (255, 0, 0),  # Blue
    'fitEllipse': (0, 255, 255),        # Yellow
}

print('=== Bounding Shapes Analysis ===')
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 200:
        continue

    # 1. Upright bounding rectangle (green)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(result, (x, y), (x + w, y + h), colors['boundingRect'], 2)

    # 2. Minimum area rotated rectangle (red)
    min_rect = cv2.minAreaRect(cnt)
    box = np.int32(cv2.boxPoints(min_rect))
    cv2.drawContours(result, [box], 0, colors['minAreaRect'], 2)

    # 3. Minimum enclosing circle (blue)
    (mcx, mcy), mradius = cv2.minEnclosingCircle(cnt)
    cv2.circle(result, (int(mcx), int(mcy)), int(mradius), colors['minEnclosingCircle'], 2)

    # 4. Fitted ellipse (yellow) - needs at least 5 points
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(result, ellipse, colors['fitEllipse'], 2)

    # Print measurements
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0
    rot_w, rot_h = min_rect[1]
    rot_angle = min_rect[2]

    print(f'\nContour {i}:')
    print(f'  Upright rect:  ({x},{y}) {w}x{h}, AR={aspect_ratio:.2f}, extent={extent:.2f}')
    print(f'  Rotated rect:  {rot_w:.0f}x{rot_h:.0f}, angle={rot_angle:.1f} deg')
    print(f'  Encl. circle:  center=({mcx:.0f},{mcy:.0f}), radius={mradius:.0f}')
    if len(cnt) >= 5:
        ecx, ecy = ellipse[0]
        emaj, emin = max(ellipse[1]), min(ellipse[1])
        ecc = emin / emaj if emaj > 0 else 0
        print(f'  Fitted ellipse: center=({ecx:.0f},{ecy:.0f}), '
              f'axes=({emaj:.0f},{emin:.0f}), eccentricity={ecc:.2f}')

# Add legend
legend_y = 15
for name, color in colors.items():
    cv2.putText(result, name, (610, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    legend_y += 18

cv2.imshow('Bounding Shapes', result)
```
