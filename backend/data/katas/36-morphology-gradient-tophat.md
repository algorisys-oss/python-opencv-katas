---
slug: 36-morphology-gradient-tophat
title: "Morphology: Gradient, TopHat, BlackHat"
level: intermediate
concepts: [MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT]
prerequisites: [35-morphology-opening-closing]
---

## What Problem Are We Solving?

Beyond noise removal (opening/closing), morphological operations can extract **structural information** from images. The **morphological gradient** extracts object outlines. **TopHat** isolates small bright features on a dark or uneven background. **BlackHat** isolates small dark features on a bright background. These three operations are powerful tools for feature extraction and image analysis.

## Morphological Gradient

The morphological gradient is the **difference between dilation and erosion**:

```python
gradient = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel)
```

This is equivalent to:

```python
gradient = cv2.dilate(src, kernel) - cv2.erode(src, kernel)
```

Dilation expands the object outward; erosion shrinks it inward. The difference is the **boundary region** — giving you a clean outline of every object. It works on both binary and grayscale images.

## TopHat (White Hat)

TopHat is the **difference between the original image and its opening**:

```python
tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
```

This is equivalent to:

```python
tophat = src - cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
```

Since opening removes small bright features, subtracting the opened image from the original **isolates those small bright features**. Use cases:

- Extracting bright text on a dark background with uneven lighting
- Finding small bright spots (stars, particles, defects)
- Correcting uneven illumination before thresholding

## BlackHat

BlackHat is the **difference between the closing and the original image**:

```python
blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
```

This is equivalent to:

```python
blackhat = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel) - src
```

Since closing fills small dark holes, subtracting the original from the closed image **isolates those small dark features**. Use cases:

- Finding dark text or scratches on a bright surface
- Detecting dark defects in a bright material
- Extracting blood vessels in medical imaging

## Kernel Size Matters

The kernel size determines what counts as "small":

```python
# Small kernel: extracts fine details
small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Large kernel: extracts larger features
large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
```

Features **smaller than the kernel** are extracted by TopHat/BlackHat. Features **larger than the kernel** are ignored.

## Practical Example: Uneven Illumination

TopHat is especially useful when lighting is uneven:

```python
# Original image has bright text but uneven background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, large_kernel)
# tophat now contains just the text, with uniform background
_, binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
```

## Tips & Common Mistakes

- The morphological gradient produces **thick outlines** — the thickness equals the kernel size. Use a small kernel (3x3 or 5x5) for thin outlines.
- TopHat and BlackHat output **difference images** — they're usually dim. You may need to increase contrast or threshold the result.
- For TopHat/BlackHat, the kernel must be **larger** than the features you want to extract. If the kernel is too small, nothing will be extracted.
- TopHat extracts **bright-on-dark** features; BlackHat extracts **dark-on-bright** features. Don't mix them up.
- These operations work on **grayscale** images too, not just binary. On grayscale, they're particularly useful for illumination correction.
- For the morphological gradient, use `MORPH_ELLIPSE` for rounded outlines that look natural.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a grayscale image with various features ---
img = np.zeros((400, 600), dtype=np.uint8)

# Uneven background (gradient illumination)
for x in range(600):
    img[:, x] = int(40 + 60 * (x / 600))

# Large objects
cv2.rectangle(img, (30, 30), (170, 170), 200, -1)
cv2.circle(img, (300, 110), 75, 220, -1)
cv2.rectangle(img, (430, 40), (560, 160), 190, -1)

# Small bright features (TopHat will extract these)
for pos in [(50, 250), (120, 280), (190, 260), (260, 290), (330, 250)]:
    cv2.circle(img, pos, 5, 230, -1)

# Small dark features inside bright objects (BlackHat will extract these)
cv2.circle(img, (100, 100), 6, 50, -1)
cv2.circle(img, (130, 80), 5, 40, -1)
cv2.circle(img, (300, 100), 6, 50, -1)
cv2.circle(img, (490, 90), 5, 40, -1)

# Text (bright on varying background)
cv2.putText(img, 'HELLO', (150, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 220, 2)

# --- Apply morphological operations ---
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

# Morphological gradient (outlines)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_small)

# TopHat (bright features on dark/uneven background)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_large)

# BlackHat (dark features on bright background)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_large)

# --- Practical use: threshold TopHat for clean extraction ---
tophat_thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)[1]

# --- Build comparison display ---
def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
panels = [
    (to_bgr(img), 'Original (uneven light)'),
    (to_bgr(gradient), 'Morph Gradient (outlines)'),
    (to_bgr(tophat), 'TopHat (bright features)'),
    (to_bgr(blackhat), 'BlackHat (dark features)'),
    (to_bgr(tophat_thresh), 'TopHat + Threshold'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

blank = np.zeros_like(panels[0][0])
cv2.putText(blank, '(blank)', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
top_row = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
bottom_row = np.hstack([panels[3][0], panels[4][0], blank])
result = np.vstack([top_row, bottom_row])

print(f'Gradient - max value: {gradient.max()}, non-zero pixels: {np.count_nonzero(gradient)}')
print(f'TopHat - max value: {tophat.max()}, reveals bright features despite uneven lighting')
print(f'BlackHat - max value: {blackhat.max()}, reveals dark defects inside bright objects')

cv2.imshow('Morphology: Gradient, TopHat, BlackHat', result)
```
