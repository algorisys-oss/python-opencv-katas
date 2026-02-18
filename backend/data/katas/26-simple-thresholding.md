---
slug: 26-simple-thresholding
title: Simple Thresholding
level: intermediate
concepts: [cv2.threshold, THRESH_BINARY, THRESH_TRUNC, THRESH_TOZERO]
prerequisites: [02-color-spaces]
---

## What Problem Are We Solving?

Many computer vision tasks require separating an image into **foreground and background** — detecting objects, reading text, analyzing shapes. **Thresholding** is the simplest form of image segmentation: it converts a grayscale image into a binary (black and white) image by comparing each pixel to a fixed value. Pixels above the threshold become white (or a set value), pixels below become black (or zero).

## How Simple Thresholding Works

The basic idea is straightforward:

```
For each pixel in the image:
    if pixel_value > threshold:
        output = maxval (typically 255)
    else:
        output = 0
```

This creates a **binary mask** where bright regions become white and dark regions become black. The key challenge is choosing the right threshold value.

## Using cv2.threshold()

```python
ret, binary = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
```

| Parameter | Meaning |
|---|---|
| `gray` | Input image (must be **single-channel/grayscale**) |
| `thresh` | Threshold value (0-255 for uint8 images) |
| `maxval` | Value assigned to pixels that pass the threshold test |
| `type` | Thresholding method |

Returns:
- `ret` — the threshold value used (same as `thresh` for simple thresholding; useful for Otsu's method)
- `binary` — the thresholded output image

## Threshold Types

OpenCV provides five threshold types. Given threshold `T` and max value `M`:

### THRESH_BINARY
```python
ret, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# pixel > T  →  M (255)
# pixel <= T →  0
```
The most common type. Produces a clean binary image.

### THRESH_BINARY_INV
```python
ret, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# pixel > T  →  0
# pixel <= T →  M (255)
```
Inverted binary — dark objects on light background become white.

### THRESH_TRUNC
```python
ret, out = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
# pixel > T  →  T (capped at threshold)
# pixel <= T →  pixel (unchanged)
```
Caps bright values at the threshold. Useful for limiting intensity range.

### THRESH_TOZERO
```python
ret, out = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
# pixel > T  →  pixel (unchanged)
# pixel <= T →  0
```
Keeps bright pixels, zeros out dark ones. Good for isolating bright features.

### THRESH_TOZERO_INV
```python
ret, out = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
# pixel > T  →  0
# pixel <= T →  pixel (unchanged)
```
Inverse of TOZERO — keeps dark pixels, zeros out bright ones.

## Choosing the Right Threshold Value

The threshold value is the most critical parameter. Too low and you include noise; too high and you lose detail:

```python
# Low threshold — captures more detail but also noise
_, low = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

# Medium threshold — typical starting point
_, mid = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# High threshold — only the brightest features remain
_, high = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
```

> **Practical approach:** Look at the image histogram to find natural gaps between foreground and background intensities. Place the threshold in that gap.

## Binary Segmentation

Thresholding is the first step in many segmentation pipelines:

```python
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to create binary mask
_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Use mask to extract foreground from original color image
foreground = cv2.bitwise_and(img, img, mask=mask)
```

This pattern — threshold to create a mask, then apply the mask — is fundamental in OpenCV.

## Pre-processing Before Thresholding

Noise in the image can create speckled results. Always smooth before thresholding:

```python
# Without smoothing: noisy binary output
_, noisy_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# With smoothing: cleaner binary output
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, clean_binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
```

## Tips & Common Mistakes

- Input must be **grayscale** (single-channel). Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` first.
- `maxval` is only used by `THRESH_BINARY` and `THRESH_BINARY_INV`. The other types ignore it.
- The return value `ret` seems useless for simple thresholding (it just returns the threshold you passed in), but becomes important with Otsu's method.
- A threshold of 127 is not magic — it's just the midpoint of 0-255. The best threshold depends entirely on your image.
- Always visualize your threshold result. Small changes in the threshold value can dramatically change the output.
- For images with uneven lighting, simple thresholding often fails. Use adaptive thresholding instead.
- Smooth (Gaussian blur) before thresholding to reduce noise-induced speckle in the binary output.

## Starter Code

```python
import cv2
import numpy as np

# Create a grayscale test image with a gradient and shapes
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Horizontal gradient background (0 to 255)
for x in range(400):
    val = int(255 * x / 399)
    img[:, x] = (val, val, val)

# Draw shapes with known intensities
cv2.rectangle(img, (30, 40), (130, 140), (220, 220, 220), -1)
cv2.circle(img, (250, 90), 50, (60, 60, 60), -1)
cv2.rectangle(img, (50, 170), (180, 260), (40, 40, 40), -1)
cv2.circle(img, (300, 220), 40, (200, 200, 200), -1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Apply all five threshold types ---
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

# --- Different threshold values on THRESH_BINARY ---
_, low_t = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
_, mid_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, high_t = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Label helper
def label(image, text):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 255), 1, cv2.LINE_AA)
    return out

# Row 1: All threshold types
row1 = np.hstack([
    label(gray, 'Original'),
    label(binary, 'BINARY'),
    label(binary_inv, 'BINARY_INV'),
    label(trunc, 'TRUNC'),
])

# Row 2: TOZERO variants + threshold value comparison
row2 = np.hstack([
    label(tozero, 'TOZERO'),
    label(low_t, 'thresh=80'),
    label(mid_t, 'thresh=127'),
    label(high_t, 'thresh=200'),
])

result = np.vstack([row1, row2])

# Count white pixels at different thresholds
print(f'White pixels (thresh=80):  {np.sum(low_t == 255)}')
print(f'White pixels (thresh=127): {np.sum(mid_t == 255)}')
print(f'White pixels (thresh=200): {np.sum(high_t == 255)}')
print(f'Total pixels: {gray.size}')

cv2.imshow('Simple Thresholding', result)
```
