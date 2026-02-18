---
slug: 60-harris-corner-detection
title: Harris Corner Detection
level: advanced
concepts: [cv2.cornerHarris, corner response, non-max suppression]
prerequisites: [29-sobel-edge-detection, 26-simple-thresholding]
---

## What Problem Are We Solving?

Edges tell us where intensity changes in one direction, but **corners** are where intensity changes in **two directions** simultaneously. Corners are far more useful than edges for tasks like image matching, tracking, and 3D reconstruction because they are **unique landmarks** — you can pinpoint a corner precisely, whereas a point on an edge could be anywhere along that edge. The **Harris Corner Detector** is a classic algorithm that identifies these corner points reliably.

## The Harris Corner Concept

The Harris detector works by examining how the image intensity changes when you shift a small window in any direction. At a **flat region**, shifting the window in any direction produces no change. Along an **edge**, shifting parallel to the edge produces no change, but shifting perpendicular does. At a **corner**, shifting in **any** direction produces a significant change.

Mathematically, Harris computes a **corner response** R for each pixel based on the eigenvalues of the structure tensor (the matrix of squared gradients). The response is:

```
R = det(M) - k * (trace(M))^2
```

Where M is the structure tensor, `det` is the determinant, `trace` is the sum of eigenvalues, and `k` is a sensitivity parameter (typically 0.04 to 0.06). Large positive R indicates a corner, large negative R indicates an edge, and small |R| indicates a flat region.

## Using cv2.cornerHarris()

```python
dst = cv2.cornerHarris(src, blockSize, ksize, k)
```

| Parameter | Meaning |
|---|---|
| `src` | Input single-channel image (grayscale), `float32` type |
| `blockSize` | Size of the neighborhood for the structure tensor (e.g., 2 or 3) |
| `ksize` | Aperture size for the Sobel derivative (e.g., 3, 5, 7) |
| `k` | Harris detector free parameter, typically 0.04 to 0.06 |

The function returns a **response map** the same size as the input, where each pixel contains its corner response value R.

## The blockSize Parameter

The `blockSize` controls how large a neighborhood is used to compute the structure tensor at each pixel:

```python
# Small blockSize: detects sharp, small-scale corners
dst_small = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Larger blockSize: detects larger-scale corner structures
dst_large = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.04)
```

A smaller blockSize is more sensitive to fine details but also more sensitive to noise. A larger blockSize is more robust but may miss small corners.

## The k Parameter

The `k` parameter controls the sensitivity of the detector. It balances the relationship between the determinant and trace of the structure tensor:

```python
# Lower k: more corners detected (more sensitive)
dst_sensitive = cv2.cornerHarris(gray, 2, 3, k=0.02)

# Higher k: fewer corners detected (more selective)
dst_selective = cv2.cornerHarris(gray, 2, 3, k=0.06)
```

## Thresholding the Response

The raw corner response map contains floating-point values. To identify corners, you **threshold** the response — only pixels with R above a certain fraction of the maximum response are considered corners:

```python
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Threshold at 1% of the maximum response value
threshold = 0.01 * dst.max()
corner_mask = dst > threshold

# Mark corners on the original image
img[corner_mask] = [0, 0, 255]  # Red corners
```

## Non-Maximum Suppression with Dilation

To get cleaner corner points (instead of blobs), you can apply **dilation** to the response map before thresholding. This expands the response peaks, and when you compare each pixel to the dilated version, only the local maxima survive:

```python
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate to find local maxima
dst_dilated = cv2.dilate(dst, None)

# Only keep pixels that are local maxima AND above threshold
corners = (dst == dst_dilated) & (dst > 0.01 * dst.max())
```

## Tips & Common Mistakes

- The input image **must** be `float32`. Convert with `gray = np.float32(gray)` before calling `cornerHarris`. Passing `uint8` will either error or produce garbage results.
- The output is a response map, not a list of corner points. You need to threshold it yourself.
- A threshold of `0.01 * dst.max()` is a common starting point, but the best value depends on your image. Experiment with values from 0.001 to 0.1.
- `ksize` must be odd (3, 5, 7, ...) and refers to the Sobel kernel size, not the block size.
- Harris corners are **not scale-invariant** — corners detected at one image scale may not be detected at another. For scale invariance, look into SIFT or ORB.
- The `k` parameter has a subtle but important effect. Values that are too low produce many false corners; values too high miss real corners.
- Apply Gaussian blur before corner detection if the image is noisy, otherwise noise can produce spurious corners.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with clear corners
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (50, 50, 50)

# Draw shapes with sharp corners
cv2.rectangle(img, (40, 40), (180, 180), (255, 255, 255), 2)
cv2.rectangle(img, (220, 40), (360, 140), (200, 200, 200), -1)
pts = np.array([[450, 40], [550, 100], [500, 190], [400, 150]], dtype=np.int32)
cv2.polylines(img, [pts], True, (180, 220, 255), 2)
cv2.rectangle(img, (40, 230), (200, 370), (100, 200, 100), -1)
cv2.line(img, (250, 250), (380, 250), (255, 255, 255), 2)
cv2.line(img, (380, 250), (310, 370), (255, 255, 255), 2)
cv2.line(img, (310, 370), (250, 250), (255, 255, 255), 2)

# Convert to grayscale and float32
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f32 = np.float32(gray)

# --- Harris corner detection with different parameters ---
dst_default = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
dst_large_block = cv2.cornerHarris(gray_f32, blockSize=5, ksize=3, k=0.04)
dst_low_k = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.02)

# --- Non-maximum suppression via dilation ---
dst_dilated = cv2.dilate(dst_default, None)

# --- Mark corners on copies of the image ---
img_default = img.copy()
img_large_block = img.copy()
img_low_k = img.copy()
img_nms = img.copy()

# Threshold and mark: default
thresh = 0.01 * dst_default.max()
img_default[dst_default > thresh] = [0, 0, 255]

# Threshold and mark: larger blockSize
thresh2 = 0.01 * dst_large_block.max()
img_large_block[dst_large_block > thresh2] = [0, 0, 255]

# Threshold and mark: lower k
thresh3 = 0.01 * dst_low_k.max()
img_low_k[dst_low_k > thresh3] = [0, 0, 255]

# Threshold and mark: with NMS
nms_mask = (dst_default == dst_dilated) & (dst_default > thresh)
img_nms[nms_mask] = [0, 0, 255]

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_default, 'Default (block=2, k=0.04)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_large_block, 'blockSize=5', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_low_k, 'k=0.02 (more sensitive)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_nms, 'With NMS', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Build comparison grid
top_row = np.hstack([img_default, img_large_block])
bottom_row = np.hstack([img_low_k, img_nms])
result = np.vstack([top_row, bottom_row])

# Print corner response statistics
print(f'Corner response range: {dst_default.min():.6f} to {dst_default.max():.6f}')
print(f'Threshold value: {thresh:.6f}')
print(f'Corners detected (default): {np.sum(dst_default > thresh)}')
print(f'Corners detected (NMS): {np.sum(nms_mask)}')

cv2.imshow('Harris Corner Detection', result)
```
