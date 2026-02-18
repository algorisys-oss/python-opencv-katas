---
slug: 27-adaptive-thresholding
title: Adaptive Thresholding
level: intermediate
concepts: [cv2.adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C]
prerequisites: [26-simple-thresholding]
---

## What Problem Are We Solving?

Simple thresholding uses a **single global value** for the entire image. This fails badly when lighting is uneven — for example, a page photographed under a desk lamp where one side is bright and the other is in shadow. With a global threshold, either the bright side washes out or the dark side turns completely black. **Adaptive thresholding** solves this by computing a **different threshold for every pixel** based on its local neighborhood.

## Why Simple Thresholding Fails with Uneven Lighting

Consider a document with text under uneven illumination:

```
Left side (shadow):  background=80,  text=30   → good threshold ~55
Right side (bright): background=220, text=170  → good threshold ~195

Global threshold=127: left text disappears, right background included
```

No single threshold value works for both regions. Adaptive thresholding computes a local threshold for each pixel, so each region gets the threshold it needs.

## Using cv2.adaptiveThreshold()

```python
binary = cv2.adaptiveThreshold(
    gray,                           # Input: must be 8-bit single-channel
    maxValue=255,                   # Value for pixels that pass
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,   # Method
    thresholdType=cv2.THRESH_BINARY,              # Binary or Binary Inverse
    blockSize=11,                   # Size of local neighborhood
    C=2                             # Constant subtracted from mean
)
```

| Parameter | Meaning |
|---|---|
| `gray` | Input image — **must be 8-bit grayscale** |
| `maxValue` | Output value for "white" pixels (typically 255) |
| `adaptiveMethod` | How the local threshold is computed |
| `thresholdType` | `THRESH_BINARY` or `THRESH_BINARY_INV` only |
| `blockSize` | Size of the local neighborhood — **must be odd** (3, 5, 7, 11, ...) |
| `C` | Constant subtracted from the computed local threshold |

## ADAPTIVE_THRESH_MEAN_C

The threshold for each pixel is the **mean** of its block neighborhood minus the constant C:

```python
# threshold(x,y) = mean(neighborhood) - C
binary_mean = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
```

For a pixel at position (x, y) with blockSize=11:
1. Take the 11x11 neighborhood centered on (x, y)
2. Compute the mean of those 121 pixels
3. Subtract C from the mean — that's the threshold for this pixel
4. If pixel value > threshold: output = 255, else output = 0

## ADAPTIVE_THRESH_GAUSSIAN_C

The threshold is a **Gaussian-weighted mean** of the neighborhood minus C:

```python
# threshold(x,y) = gaussian_weighted_mean(neighborhood) - C
binary_gauss = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
```

The Gaussian-weighted mean gives more importance to pixels closer to the center, producing slightly smoother threshold boundaries. In practice, Gaussian adaptive thresholding tends to produce **cleaner results** than mean adaptive thresholding.

## Understanding Block Size

The `blockSize` parameter determines how "local" the threshold computation is:

```python
# Small block: very local threshold, captures fine detail
# But can be noisy if block is too small
small_block = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

# Medium block: good balance between local adaptation and smoothness
medium_block = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Large block: broader adaptation, approaches global behavior
# Good for large-scale lighting variations
large_block = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
```

> **Choosing blockSize:** The block should be large enough to cover both foreground and background pixels. For text on a page, 11-31 works well. For larger features, try 51-101. **Must be odd.**

## Understanding the C Constant

The constant C is subtracted from the computed mean, effectively adjusting sensitivity:

```python
# C = 0: threshold is exactly the local mean
# Flat regions (where pixel ≈ mean) will be noisy
_, c0 = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

# C = 2: threshold is slightly below the local mean
# Pixels must be noticeably darker than their neighborhood to be "black"
# This is the typical starting value
c2 = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# C = 10: very conservative — only significantly darker pixels are "black"
# Good for noisy images or when you want less detail
c10 = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
```

- **Positive C** → fewer pixels marked as foreground (more conservative)
- **Negative C** → more pixels marked as foreground (more aggressive)
- **C = 0** → threshold equals the local mean exactly

## Comparing Global vs Adaptive Thresholding

```python
# Global: one threshold for entire image
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Mean: local mean per pixel
adaptive_mean = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian: local Gaussian-weighted mean per pixel
adaptive_gauss = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

## Tips & Common Mistakes

- Input **must** be 8-bit single-channel (grayscale). Color images will cause an error.
- `blockSize` **must be odd** and greater than 1. Even values cause an error.
- Only `THRESH_BINARY` and `THRESH_BINARY_INV` are supported as `thresholdType`. Other threshold types (TRUNC, TOZERO) are not available with adaptive thresholding.
- Pre-processing with Gaussian blur before adaptive thresholding reduces noise artifacts.
- If the result looks too speckled, increase C (less sensitive) or increase blockSize (broader averaging).
- If you're losing fine detail, decrease C or decrease blockSize.
- Gaussian adaptive typically produces better results than mean adaptive — use it as your default.
- Adaptive thresholding is ideal for document scanning, OCR preprocessing, and any task where lighting is uncontrolled.

## Starter Code

```python
import cv2
import numpy as np

# Create an image with uneven "lighting" (gradient brightness)
img = np.zeros((300, 400), dtype=np.uint8)

# Simulate uneven illumination with a gradient
for x in range(400):
    for y in range(300):
        # Background brightness varies across the image
        brightness = int(80 + 150 * x / 399 + 30 * np.sin(y / 50))
        img[y, x] = np.clip(brightness, 0, 255)

# Draw dark "text-like" features at fixed contrast below background
features = img.copy()
cv2.rectangle(features, (30, 30), (120, 80), 0, -1)
cv2.rectangle(features, (250, 30), (370, 80), 0, -1)
cv2.circle(features, (80, 160), 35, 0, -1)
cv2.circle(features, (300, 160), 35, 0, -1)
cv2.putText(features, 'ABC', (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
cv2.putText(features, 'XYZ', (240, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

# Blend: features are darker than local background
gray = np.where(features == 0, np.clip(img.astype(int) - 60, 0, 255), img).astype(np.uint8)

# Add mild noise
noise = np.random.randint(-10, 10, gray.shape, dtype=np.int16)
gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Global thresholding (fails with uneven lighting) ---
_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --- Adaptive Mean thresholding ---
adapt_mean = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 5)

# --- Adaptive Gaussian thresholding ---
adapt_gauss = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

# --- Block size comparison ---
block_small = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
block_large = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

# --- C constant comparison ---
c_low = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
c_high = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

# Label helper
def label(image, text):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
    return out

# Row 1: Global vs Adaptive methods
row1 = np.hstack([
    label(gray, 'Uneven input'),
    label(global_thresh, 'Global 127'),
    label(adapt_mean, 'Adapt Mean'),
    label(adapt_gauss, 'Adapt Gauss'),
])

# Row 2: Parameter variations
row2 = np.hstack([
    label(block_small, 'block=7'),
    label(block_large, 'block=51'),
    label(c_low, 'C=0'),
    label(c_high, 'C=15'),
])

result = np.vstack([row1, row2])

print(f'Global threshold white pixels: {np.sum(global_thresh == 255)}')
print(f'Adaptive Gaussian white pixels: {np.sum(adapt_gauss == 255)}')
print(f'Image intensity range: [{gray.min()}, {gray.max()}]')
print('Notice: global fails on the dark side, adaptive works everywhere')

cv2.imshow('Adaptive Thresholding', result)
```
