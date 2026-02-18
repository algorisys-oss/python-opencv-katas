---
slug: 61-shi-tomasi-corners
title: Shi-Tomasi Corners
level: advanced
concepts: [cv2.goodFeaturesToTrack, quality level, min distance]
prerequisites: [60-harris-corner-detection]
---

## What Problem Are We Solving?

The Harris corner detector gives a response map that you have to threshold manually, and its corner score formula can sometimes be sensitive to the `k` parameter. **Shi-Tomasi** proposed a simpler and often more effective corner quality measure: instead of computing `det(M) - k * trace(M)^2`, just use the **minimum eigenvalue** of the structure tensor. If the smaller eigenvalue is large, both eigenvalues are large, which guarantees a strong corner. OpenCV provides this as `cv2.goodFeaturesToTrack()`, which directly returns a list of the best corner coordinates.

## Shi-Tomasi vs Harris

Harris computes the corner response as:

```
R = lambda1 * lambda2 - k * (lambda1 + lambda2)^2
```

Shi-Tomasi simplifies this to:

```
R = min(lambda1, lambda2)
```

This is more intuitive — a corner exists when both eigenvalues exceed a threshold. In practice, Shi-Tomasi tends to select corners that are more **uniformly strong** across the image, and the quality parameter is easier to reason about than Harris's `k`.

## Using cv2.goodFeaturesToTrack()

```python
corners = cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
```

| Parameter | Meaning |
|---|---|
| `image` | Input single-channel grayscale image (`uint8` or `float32`) |
| `maxCorners` | Maximum number of corners to return (0 = no limit) |
| `qualityLevel` | Minimum accepted quality as a fraction of the best corner's quality (0.0 to 1.0) |
| `minDistance` | Minimum Euclidean distance (in pixels) between returned corners |

The function returns an array of shape `(N, 1, 2)` containing the (x, y) coordinates of the N strongest corners, or `None` if no corners are found.

## The qualityLevel Parameter

The `qualityLevel` sets the minimum acceptable corner quality relative to the strongest corner. If the best corner has quality Q, then only corners with quality >= `qualityLevel * Q` are kept:

```python
# Low quality level: more corners (accepts weaker corners)
corners_many = cv2.goodFeaturesToTrack(gray, 100, qualityLevel=0.01, minDistance=10)

# High quality level: fewer, stronger corners
corners_few = cv2.goodFeaturesToTrack(gray, 100, qualityLevel=0.3, minDistance=10)
```

A typical starting value is 0.01 (accept corners that are at least 1% as strong as the best). Increase it to be more selective.

## The minDistance Parameter

After corners are sorted by quality, `minDistance` enforces a minimum spacing. Starting from the strongest corner, any corner within `minDistance` pixels of an already-accepted corner is rejected:

```python
# Corners can be very close together
corners_close = cv2.goodFeaturesToTrack(gray, 50, 0.01, minDistance=5)

# Corners are spread out
corners_spread = cv2.goodFeaturesToTrack(gray, 50, 0.01, minDistance=30)
```

This is critical for applications like tracking, where you want corners distributed across the image rather than clustered in one area.

## The maxCorners Parameter

`maxCorners` puts a hard cap on how many corners are returned. Corners are sorted by strength, so you always get the N best:

```python
# Get only the 4 strongest corners
top_4 = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10)

# Get up to 200 corners
many = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=5)

# No limit on count (set to 0)
all_corners = cv2.goodFeaturesToTrack(gray, maxCorners=0, qualityLevel=0.01, minDistance=5)
```

## Drawing the Detected Corners

The returned corners are floating-point coordinates. To draw them, convert to integers:

```python
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
if corners is not None:
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
```

## Tips & Common Mistakes

- Unlike `cv2.cornerHarris`, you do **not** need to convert the input to `float32` — `goodFeaturesToTrack` accepts `uint8` grayscale directly.
- The return value can be `None` if no corners meet the criteria. Always check before iterating.
- The returned coordinates are `float32` with shape `(N, 1, 2)`. Use `.ravel()` or indexing like `corner[0]` to extract (x, y).
- Setting `maxCorners=0` means unlimited, not zero corners.
- Very low `minDistance` values (e.g., 1 or 2) can produce clusters of corners at the same feature. Use at least 5-10 pixels for most applications.
- The `qualityLevel` is **relative** to the best corner in the image. This means different images can have very different absolute thresholds even with the same `qualityLevel`.
- You can optionally pass `useHarrisDetector=True` to make `goodFeaturesToTrack` use the Harris score instead of the minimum eigenvalue, along with a `k` parameter.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with various corner-producing shapes
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (50, 50, 50)

# Draw shapes with clear corners
cv2.rectangle(img, (30, 30), (170, 170), (255, 255, 255), 2)
cv2.rectangle(img, (200, 30), (350, 130), (200, 200, 200), -1)
pts_star = np.array([[480, 30], [500, 90], [560, 100], [515, 140],
                     [530, 200], [480, 160], [430, 200], [445, 140],
                     [400, 100], [460, 90]], dtype=np.int32)
cv2.polylines(img, [pts_star], True, (180, 220, 255), 2)
cv2.rectangle(img, (30, 230), (150, 370), (100, 200, 100), -1)
pts_tri = np.array([[250, 370], [350, 230], [450, 370]], dtype=np.int32)
cv2.polylines(img, [pts_tri], True, (255, 180, 100), 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Different quality levels ---
corners_low_q = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
corners_high_q = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.2, minDistance=10)

# --- Different min distances ---
corners_close = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=5)
corners_spread = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=30)

# --- Draw corners on separate copies ---
def draw_corners(image, corners, color=(0, 0, 255), radius=5):
    out = image.copy()
    if corners is not None:
        for c in corners:
            x, y = c.ravel()
            cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out

img_low_q = draw_corners(img, corners_low_q)
img_high_q = draw_corners(img, corners_high_q)
img_close = draw_corners(img, corners_close, color=(255, 0, 0))
img_spread = draw_corners(img, corners_spread, color=(255, 0, 0))

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
n_low = len(corners_low_q) if corners_low_q is not None else 0
n_high = len(corners_high_q) if corners_high_q is not None else 0
n_close = len(corners_close) if corners_close is not None else 0
n_spread = len(corners_spread) if corners_spread is not None else 0

cv2.putText(img_low_q, f'quality=0.01 ({n_low} pts)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_high_q, f'quality=0.2 ({n_high} pts)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_close, f'minDist=5 ({n_close} pts)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_spread, f'minDist=30 ({n_spread} pts)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Build comparison grid
top_row = np.hstack([img_low_q, img_high_q])
bottom_row = np.hstack([img_close, img_spread])
result = np.vstack([top_row, bottom_row])

print(f'Corners detected (quality=0.01): {n_low}')
print(f'Corners detected (quality=0.2): {n_high}')
print(f'Corners detected (minDist=5): {n_close}')
print(f'Corners detected (minDist=30): {n_spread}')

cv2.imshow('Shi-Tomasi Corners', result)
```
