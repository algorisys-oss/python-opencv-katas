---
slug: 11-image-translation
title: Image Translation
level: beginner
concepts: [cv2.warpAffine, translation matrix, shifting images]
prerequisites: [09-image-rotation]
---

## What Problem Are We Solving?

Translation means **shifting** an image — moving every pixel a fixed number of pixels in the x and/or y direction. You might need this to center an object in a frame, align two images, create a sliding animation, or as part of data augmentation. OpenCV handles translation through the same `cv2.warpAffine()` function you already know from rotation, but with a simpler matrix.

## The 2x3 Translation Matrix

An affine transformation is defined by a 2x3 matrix. For pure translation (no rotation, no scaling), the matrix looks like this:

```
| 1  0  tx |
| 0  1  ty |
```

Where:
- `tx` = number of pixels to shift **horizontally** (positive = shift right, negative = shift left)
- `ty` = number of pixels to shift **vertically** (positive = shift down, negative = shift up)

In code:

```python
import numpy as np

tx, ty = 100, 50  # Shift 100px right, 50px down
M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])
```

> **Why float32?** `cv2.warpAffine()` expects the matrix to be a 32-bit float array. Using `np.float32()` or `np.float64()` works; plain Python lists or integer arrays will cause an error.

## Applying the Translation

Just like rotation, you apply the matrix with `cv2.warpAffine()`:

```python
translated = cv2.warpAffine(img, M, (width, height))
```

```python
h, w = img.shape[:2]
tx, ty = 75, 30
M = np.float32([[1, 0, tx], [0, 1, ty]])
shifted = cv2.warpAffine(img, M, (w, h))
```

The output size is `(width, height)`. Pixels that shift **outside** this boundary are lost. New areas that were not in the original are filled with **black** (zeros) by default.

## Shifting in All Four Directions

```python
h, w = img.shape[:2]

# Shift right and down
M_right_down = np.float32([[1, 0, 50], [0, 1, 30]])
rd = cv2.warpAffine(img, M_right_down, (w, h))

# Shift left and up (negative values)
M_left_up = np.float32([[1, 0, -50], [0, 1, -30]])
lu = cv2.warpAffine(img, M_left_up, (w, h))

# Shift right only (horizontal only)
M_right = np.float32([[1, 0, 80], [0, 1, 0]])
r = cv2.warpAffine(img, M_right, (w, h))

# Shift down only (vertical only)
M_down = np.float32([[1, 0, 0], [0, 1, 60]])
d = cv2.warpAffine(img, M_down, (w, h))
```

## What Happens to Pixels That Shift Out of Bounds

When you translate an image, pixels that move beyond the image boundary are **permanently lost** (clipped). The vacated area is filled with a default value:

```python
# Black fill (default)
shifted = cv2.warpAffine(img, M, (w, h))

# Custom fill color (e.g., gray)
shifted = cv2.warpAffine(img, M, (w, h),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(128, 128, 128))

# Replicate edge pixels to fill the border
shifted = cv2.warpAffine(img, M, (w, h),
                         borderMode=cv2.BORDER_REPLICATE)

# Wrap around (pixels that exit one side appear on the other)
shifted = cv2.warpAffine(img, M, (w, h),
                         borderMode=cv2.BORDER_WRAP)
```

| Border Mode | Effect |
|---|---|
| `cv2.BORDER_CONSTANT` | Fill with a solid color (default: black) |
| `cv2.BORDER_REPLICATE` | Repeat the nearest edge pixel |
| `cv2.BORDER_REFLECT` | Mirror the edge pixels |
| `cv2.BORDER_WRAP` | Wrap around to the opposite edge |

## Expanding the Canvas to Avoid Losing Pixels

If you want to shift the image without losing any content, make the output canvas larger:

```python
tx, ty = 100, 50
M = np.float32([[1, 0, tx], [0, 1, ty]])

# Expand canvas to accommodate the shift
new_w = w + abs(tx)
new_h = h + abs(ty)
shifted = cv2.warpAffine(img, M, (new_w, new_h))
```

For negative shifts (left or up), you need to adjust the translation values so the content shifts into the expanded area rather than out of bounds:

```python
tx, ty = -100, -50
adj_tx = max(0, -tx)  # If shifting left, add offset to keep content visible
adj_ty = max(0, -ty)
M = np.float32([[1, 0, adj_tx], [0, 1, adj_ty]])
new_w = w + abs(tx)
new_h = h + abs(ty)
shifted = cv2.warpAffine(img, M, (new_w, new_h))
```

## Understanding the Matrix Intuitively

The 2x3 affine matrix transforms each output pixel coordinate `(x', y')` back to the source:

```
source_x = 1 * x' + 0 * y' + tx  -->  x' - tx in the source
source_y = 0 * x' + 1 * y' + ty  -->  y' - ty in the source
```

Wait — that looks like the **inverse** mapping. And it is! `cv2.warpAffine()` uses an inverse mapping: "for each pixel in the output, where do I sample from the input?" This is why a positive `tx` moves the image to the **right** — the output pixel at `(100, 0)` samples from source `(0, 0)`.

## Tips & Common Mistakes

- The translation matrix must be `np.float32` (or `np.float64`). Integer arrays will fail.
- The matrix shape must be exactly `(2, 3)`. A common mistake is creating a `(3, 3)` matrix.
- Positive `tx` shifts **right**, positive `ty` shifts **down**. This matches the image coordinate system where `(0, 0)` is the top-left.
- Pixels shifted outside the output boundary are **lost**. Expand the canvas if you need to preserve all content.
- Translation is the simplest affine transform. The same `cv2.warpAffine()` function handles rotation, scaling, shearing, and any combination — translation is just the special case where the 2x2 sub-matrix is the identity.
- For very large shifts, most of the output will be the border fill value. Make sure this is intentional.
- `cv2.warpAffine()` returns a new image — the original is not modified.

## Starter Code

```python
import cv2
import numpy as np

# Create a source image with visible content and orientation markers
img = np.zeros((250, 350, 3), dtype=np.uint8)
img[:] = (40, 35, 30)

# Draw a colored rectangle with text
cv2.rectangle(img, (30, 30), (320, 220), (180, 120, 0), 2)
cv2.rectangle(img, (40, 40), (160, 120), (0, 160, 200), -1)
cv2.putText(img, 'SHIFT', (55, 95), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2, cv2.LINE_AA)

# Orientation marker: red dot in top-left area
cv2.circle(img, (50, 180), 12, (0, 0, 255), -1)
# Green dot in bottom-right area
cv2.circle(img, (280, 180), 12, (0, 255, 0), -1)

h, w = img.shape[:2]

# --- Translation 1: shift right and down ---
tx1, ty1 = 80, 50
M1 = np.float32([[1, 0, tx1], [0, 1, ty1]])
shifted_rd = cv2.warpAffine(img, M1, (w, h))

# --- Translation 2: shift left and up ---
tx2, ty2 = -60, -40
M2 = np.float32([[1, 0, tx2], [0, 1, ty2]])
shifted_lu = cv2.warpAffine(img, M2, (w, h))

# --- Translation 3: shift with replicated border ---
tx3, ty3 = 80, 50
M3 = np.float32([[1, 0, tx3], [0, 1, ty3]])
shifted_rep = cv2.warpAffine(img, M3, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

# --- Translation 4: shift with wrap-around ---
tx4, ty4 = 80, 50
M4 = np.float32([[1, 0, tx4], [0, 1, ty4]])
shifted_wrap = cv2.warpAffine(img, M4, (w, h),
                               borderMode=cv2.BORDER_WRAP)

# --- Label each image ---
cv2.putText(img, 'Original', (5, 245), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(shifted_rd, f'tx={tx1} ty={ty1}', (5, 245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(shifted_lu, f'tx={tx2} ty={ty2}', (5, 245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(shifted_rep, 'Replicate border', (5, 245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(shifted_wrap, 'Wrap border', (5, 245),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

# --- Arrange display ---
# Print the translation matrix for educational purposes
print('Translation matrix (tx=80, ty=50):')
print(M1)
print()

# 2x3 grid: top row = original, right+down, left+up
# bottom row = replicate, wrap, (blank)
top_row = np.hstack([img, shifted_rd, shifted_lu])
bottom_row = np.hstack([shifted_rep, shifted_wrap,
                        np.zeros_like(img)])

# Add labels to the blank panel
cv2.putText(bottom_row, 'Matrix:', (w * 2 + 20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
cv2.putText(bottom_row, '|1  0  tx|', (w * 2 + 20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)
cv2.putText(bottom_row, '|0  1  ty|', (w * 2 + 20, 155),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1, cv2.LINE_AA)

grid = np.vstack([top_row, bottom_row])

print(f'Original size: {w}x{h}')
print(f'All outputs:   {w}x{h} (same canvas, content clipped at edges)')

cv2.imshow('Image Translation', grid)
```
