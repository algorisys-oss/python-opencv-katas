---
slug: 21-gaussian-blur
title: Gaussian Blur
level: intermediate
concepts: [cv2.GaussianBlur, sigma, noise reduction]
prerequisites: [20-averaging-box-filter]
---

## What Problem Are We Solving?

The averaging filter from the previous kata treats all pixels in the kernel **equally**. This creates a somewhat unnatural blur because distant pixels influence the center just as much as close neighbors. **Gaussian blur** solves this by weighting pixels based on their distance from the center — nearby pixels contribute more, far pixels contribute less. This produces a smoother, more natural-looking blur and is the most widely used smoothing filter in computer vision.

## The Gaussian Kernel Concept

A Gaussian kernel is shaped like a **bell curve** (the normal distribution). In 2D, it looks like a dome — highest in the center, tapering off toward the edges:

```
Approximate 3x3 Gaussian kernel:

| 1  2  1 |
| 2  4  2 |  * (1/16)
| 1  2  1 |
```

The center pixel has weight 4, immediate neighbors have weight 2, and corner pixels have weight 1. This weighted averaging produces blur that looks much more natural than the flat averaging filter.

## Using cv2.GaussianBlur()

```python
blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `ksize` | Kernel size as `(width, height)` — **must be odd** |
| `sigmaX` | Standard deviation in X direction. `0` = auto-calculate from kernel size |
| `sigmaY` | Standard deviation in Y direction (optional, defaults to `sigmaX`) |

## Understanding Sigma

**Sigma** (standard deviation) controls the shape of the bell curve — how quickly the weights drop off from center to edge:

```python
# Small sigma: sharp bell curve, less blurring
mild = cv2.GaussianBlur(img, (7, 7), sigmaX=1)

# Large sigma: wide bell curve, more blurring
strong = cv2.GaussianBlur(img, (7, 7), sigmaX=3)

# sigmaX=0: OpenCV calculates sigma from kernel size
# Formula: sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
auto = cv2.GaussianBlur(img, (7, 7), sigmaX=0)
```

> **Key insight:** Two knobs control Gaussian blur — kernel size and sigma. Kernel size determines how many pixels are considered; sigma determines how their weights fall off. For most uses, set `sigmaX=0` and just vary the kernel size.

## Gaussian Blur vs Box Blur

```python
box_blur = cv2.blur(img, (7, 7))
gauss_blur = cv2.GaussianBlur(img, (7, 7), 0)
```

| Property | Box Blur (`cv2.blur`) | Gaussian Blur |
|---|---|---|
| **Weights** | All equal | Bell-curve weighted |
| **Visual quality** | Can look "blocky" | Smooth and natural |
| **Edge handling** | Smears edges uniformly | Slightly better edge preservation |
| **Speed** | Marginally faster | Very fast (separable filter) |
| **Use case** | Quick-and-dirty smoothing | General-purpose smoothing |

In practice, Gaussian blur is preferred for almost all smoothing tasks because it produces fewer visual artifacts.

## Noise Reduction with Gaussian Blur

Gaussian blur is the standard first step for noise reduction in image processing pipelines:

```python
# Simulate a noisy image
noise = np.random.randn(*img.shape) * 25
noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# Denoise with Gaussian blur
denoised = cv2.GaussianBlur(noisy, (5, 5), 0)
```

The key tradeoff: **more blur = less noise, but also less detail**. Gaussian blur is often applied before edge detection or thresholding to prevent noise from creating false edges or threshold artifacts.

## Separability: Why Gaussian Blur Is Fast

A 2D Gaussian kernel can be decomposed into two 1D passes — one horizontal, one vertical. This is called **separability**:

```python
# These produce the same result:
result_2d = cv2.GaussianBlur(img, (15, 15), 0)

# But internally, OpenCV does:
# 1. Blur each row with a 1D kernel of size 15
# 2. Blur each column with a 1D kernel of size 15
# This is O(n) per pixel instead of O(n^2)
```

A 15x15 kernel normally requires 225 operations per pixel, but separability reduces it to just 30 (15 + 15). This makes Gaussian blur surprisingly fast even with large kernels.

## Tips & Common Mistakes

- Kernel size **must be odd** for Gaussian blur — `(3,3)`, `(5,5)`, `(7,7)`, etc. Even sizes will raise an error.
- Setting `sigmaX=0` is the most common approach — OpenCV picks a reasonable sigma based on the kernel size.
- If you specify sigma but the kernel is too small to represent the bell curve, you'll get a poor approximation. Rule of thumb: `ksize >= 6 * sigma + 1`.
- Gaussian blur is the recommended pre-processing step before Canny edge detection and other gradient-based operations.
- For color images, each channel is blurred independently. This generally works fine.
- Don't confuse `GaussianBlur` with `bilateralFilter` — Gaussian blur smooths edges too, while bilateral filtering preserves them.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with sharp features
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (220, 220, 220)

# Draw shapes with sharp edges
cv2.rectangle(img, (20, 20), (130, 130), (200, 50, 50), -1)
cv2.circle(img, (250, 75), 55, (50, 180, 50), -1)
cv2.putText(img, 'Sharp', (290, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Add Gaussian noise
noise = np.random.randn(*img.shape).astype(np.float64) * 30
noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# --- Gaussian blur with different kernel sizes ---
gauss_3 = cv2.GaussianBlur(noisy, (3, 3), 0)
gauss_7 = cv2.GaussianBlur(noisy, (7, 7), 0)
gauss_15 = cv2.GaussianBlur(noisy, (15, 15), 0)

# --- Sigma comparison (same kernel, different sigma) ---
sigma_1 = cv2.GaussianBlur(noisy, (11, 11), sigmaX=1)
sigma_5 = cv2.GaussianBlur(noisy, (11, 11), sigmaX=5)

# --- Box blur vs Gaussian blur comparison ---
box_7 = cv2.blur(noisy, (7, 7))

# Add labels
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Row 1: kernel size comparison
row1 = np.hstack([
    label(noisy, 'Noisy'),
    label(gauss_3, 'Gauss 3x3'),
    label(gauss_7, 'Gauss 7x7'),
    label(gauss_15, 'Gauss 15x15'),
])

# Row 2: sigma comparison + box vs gauss
row2 = np.hstack([
    label(img, 'Original'),
    label(sigma_1, 'sigma=1'),
    label(sigma_5, 'sigma=5'),
    label(box_7, 'Box 7x7'),
])

result = np.vstack([row1, row2])

print(f'Noisy image noise level (std): {np.std(noisy.astype(int) - img.astype(int)):.1f}')
print(f'After Gauss 7x7 noise level: {np.std(gauss_7.astype(int) - img.astype(int)):.1f}')

cv2.imshow('Gaussian Blur', result)
```
