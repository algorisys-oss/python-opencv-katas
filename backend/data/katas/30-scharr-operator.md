---
slug: 30-scharr-operator
title: Scharr Operator
level: intermediate
concepts: [cv2.Scharr, gradient accuracy]
prerequisites: [29-sobel-edge-detection]
---

## What Problem Are We Solving?

The Sobel operator with a 3x3 kernel has **limited accuracy** when computing image gradients. The angles and magnitudes it produces can be noticeably wrong, especially for diagonal edges. The **Scharr operator** uses a different set of kernel weights that give **more accurate gradient approximations** for the same 3x3 size.

## How Scharr Differs from Sobel

The Sobel 3x3 kernel for the x-direction looks like:

```
[-1  0  1]
[-2  0  2]
[-1  0  1]
```

The Scharr kernel for the x-direction uses larger center weights:

```
[-3   0   3]
[-10  0  10]
[-3   0   3]
```

These weights are **optimized to minimize angular error** in gradient estimation. Where Sobel 3x3 can have angular errors up to several degrees, Scharr is nearly perfect.

## Using cv2.Scharr()

```python
scharr_x = cv2.Scharr(src, ddepth, dx, dy)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image (typically grayscale) |
| `ddepth` | Output depth — use `cv2.CV_64F` to capture negative gradients |
| `dx` | Order of x derivative (0 or 1) |
| `dy` | Order of y derivative (0 or 1) |

> **Important:** Unlike Sobel, Scharr does **not** take a `ksize` parameter. It always uses its fixed 3x3 kernel. You set either `dx=1, dy=0` (horizontal gradient) or `dx=0, dy=1` (vertical gradient) — never both at once.

## Scharr via Sobel's ksize=-1

You can also get the Scharr kernel by passing `ksize=-1` (or `cv2.FILTER_SCHARR`) to `cv2.Sobel()`:

```python
# These two are equivalent:
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=-1)
```

## When to Use Scharr vs Sobel

- **Use Scharr** when you need a 3x3 gradient with the best possible accuracy (e.g., computing edge orientation for feature descriptors).
- **Use Sobel with larger kernels** (5x5, 7x7) when you need more smoothing to suppress noise — Scharr is only available as 3x3.
- **Use Scharr** when gradient direction matters (e.g., computing the angle of an edge). Sobel 3x3 can produce noticeable angular error; Scharr virtually eliminates it.

## Computing Gradient Magnitude

Just like with Sobel, combine x and y gradients to get the full edge strength:

```python
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
magnitude = cv2.magnitude(scharr_x, scharr_y)
```

## Tips & Common Mistakes

- Always use `ddepth=cv2.CV_64F` (or `cv2.CV_32F`) so negative gradients are preserved. Using `cv2.CV_8U` clips negatives to zero and you lose half the edges.
- Convert to absolute value with `cv2.convertScaleAbs()` for display.
- Scharr is **only 3x3** — if you need a larger kernel, use Sobel with `ksize=5` or `ksize=7`.
- Don't set both `dx=1` and `dy=1` simultaneously — compute them separately and combine with `cv2.magnitude()`.
- Scharr is most beneficial when gradient **direction** accuracy matters. For simple edge presence detection, the difference from Sobel 3x3 is subtle.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with edges at various angles
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (40, 40, 40)

# Draw shapes with different edge orientations
cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), 2)
cv2.line(img, (250, 50), (400, 200), (255, 255, 255), 2)
cv2.circle(img, (500, 130), 80, (255, 255, 255), 2)
cv2.line(img, (50, 280), (550, 350), (255, 255, 255), 2)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Scharr gradients ---
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)  # Horizontal gradient
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)  # Vertical gradient

# Convert to absolute for display
abs_scharr_x = cv2.convertScaleAbs(scharr_x)
abs_scharr_y = cv2.convertScaleAbs(scharr_y)

# Combined magnitude
magnitude = cv2.magnitude(scharr_x, scharr_y)
magnitude = np.uint8(np.clip(magnitude, 0, 255))

# --- Compare with Sobel 3x3 ---
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

# --- Build comparison display ---
# Top row: Scharr X, Scharr Y, Scharr Magnitude
# Bottom row: Original, Sobel Magnitude, Difference
scharr_x_color = cv2.cvtColor(abs_scharr_x, cv2.COLOR_GRAY2BGR)
scharr_y_color = cv2.cvtColor(abs_scharr_y, cv2.COLOR_GRAY2BGR)
scharr_mag_color = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
sobel_mag_color = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)

# Amplified difference between Scharr and Sobel
diff = cv2.absdiff(magnitude, sobel_mag)
diff_amplified = np.uint8(np.clip(diff * 5, 0, 255))
diff_color = cv2.cvtColor(diff_amplified, cv2.COLOR_GRAY2BGR)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(scharr_x_color, 'Scharr X', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(scharr_y_color, 'Scharr Y', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(scharr_mag_color, 'Scharr Magnitude', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img, 'Original', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(sobel_mag_color, 'Sobel 3x3 Magnitude', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(diff_color, 'Difference (5x)', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

top_row = np.hstack([scharr_x_color, scharr_y_color, scharr_mag_color])
bottom_row = np.hstack([img, sobel_mag_color, diff_color])
result = np.vstack([top_row, bottom_row])

print(f'Scharr magnitude range: {magnitude.min()} - {magnitude.max()}')
print(f'Sobel magnitude range: {sobel_mag.min()} - {sobel_mag.max()}')

cv2.imshow('Scharr Operator', result)
```
