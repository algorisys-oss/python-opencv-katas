---
slug: 23-bilateral-filter
title: Bilateral Filter
level: intermediate
concepts: [cv2.bilateralFilter, edge-preserving smoothing]
prerequisites: [21-gaussian-blur]
---

## What Problem Are We Solving?

Gaussian blur smooths noise effectively, but it also **blurs edges**. In many applications — face smoothing, noise reduction in photography, pre-processing for segmentation — you want to smooth flat regions while keeping sharp boundaries intact. The **bilateral filter** achieves this by considering not just spatial distance (like Gaussian) but also **intensity difference** between pixels. Pixels that are far away in value (across an edge) get low weight, so edges are preserved.

## How the Bilateral Filter Works

A standard Gaussian filter weights pixels based only on **spatial distance** — how far a neighbor is from the center pixel. The bilateral filter adds a second weighting factor: **intensity similarity**.

For each pixel, the bilateral filter computes:

1. **Spatial weight** — how close is the neighbor? (Same as Gaussian blur)
2. **Range weight** — how similar in color/intensity is the neighbor?
3. **Combined weight** = spatial weight x range weight

```
Gaussian blur:     weight = f(spatial_distance)
Bilateral filter:  weight = f(spatial_distance) * f(intensity_difference)
```

If a neighbor pixel has a very different intensity (e.g., across an edge), its range weight drops to near zero, so it doesn't contribute to the blur. This is why edges remain sharp.

## Using cv2.bilateralFilter()

```python
smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image (8-bit or floating-point, 1 or 3 channels) |
| `d` | Diameter of the pixel neighborhood. Use `-1` to auto-compute from `sigmaSpace` |
| `sigmaColor` | Range sigma — how much intensity difference is tolerated |
| `sigmaSpace` | Spatial sigma — same as Gaussian sigma, controls spatial extent |

## Understanding sigmaColor

`sigmaColor` controls the **intensity sensitivity** — how different two pixel values can be while still being blurred together:

```python
# Small sigmaColor: only very similar pixels are blurred together
# Edges are strongly preserved, less smoothing in flat areas
sharp_edges = cv2.bilateralFilter(img, d=9, sigmaColor=25, sigmaSpace=75)

# Large sigmaColor: pixels with bigger intensity gaps are still blurred
# More smoothing, but edges start to blur too
smooth = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=75)
```

> **Rule of thumb:** A `sigmaColor` of 50-100 works well for most denoising tasks. Below 25 preserves almost every small gradient. Above 150, the filter starts behaving like a regular Gaussian blur.

## Understanding sigmaSpace

`sigmaSpace` controls the **spatial extent** of the filter — exactly like sigma in Gaussian blur:

```python
# Small sigmaSpace: narrow spatial neighborhood, subtle smoothing
local = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=20)

# Large sigmaSpace: wide spatial neighborhood, broader smoothing
broad = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=150)
```

When `d` (diameter) is set to `-1`, OpenCV automatically computes the diameter from `sigmaSpace`. This is the recommended approach:

```python
# Let OpenCV choose the neighborhood size
auto = cv2.bilateralFilter(img, d=-1, sigmaColor=75, sigmaSpace=75)
```

## Why It Preserves Edges While Smoothing

Consider a pixel right at the border between a dark region (value ~50) and a bright region (value ~200):

```
With Gaussian blur (sigmaColor = infinity):
  All neighbors averaged equally → edge pixel becomes ~125 (smeared)

With bilateral filter (sigmaColor = 50):
  Dark neighbors: similar intensity → high weight → contribute normally
  Bright neighbors: very different → near-zero weight → ignored
  Result: pixel stays ~50 → edge preserved!
```

The bilateral filter effectively creates a **data-dependent kernel** that adapts its shape at every pixel location.

## The Speed Tradeoff

The bilateral filter is significantly **slower** than Gaussian blur because:

1. It cannot be separated into 1D passes (it's not separable)
2. Weights must be recomputed at every pixel (they depend on local intensities)

```python
import time

# Typical timing comparison
start = time.time()
gauss = cv2.GaussianBlur(img, (9, 9), 0)
gauss_time = time.time() - start

start = time.time()
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
bilateral_time = time.time() - start

# Bilateral is typically 10-50x slower than Gaussian
```

For real-time applications, consider:
- Using a smaller `d` value (5 instead of 9)
- Using `cv2.edgePreservingFilter()` as a faster alternative
- Applying bilateral filter only to regions of interest, not the full image

## Tips & Common Mistakes

- Bilateral filter is **much slower** than Gaussian blur — 10x to 50x for typical parameters. Don't use it in real-time pipelines without profiling first.
- Start with `d=9, sigmaColor=75, sigmaSpace=75` as a baseline and adjust from there.
- Use `d=-1` to auto-compute diameter from `sigmaSpace` — this avoids inconsistent parameter combinations.
- Very large `sigmaColor` values (200+) make the bilateral filter behave almost identically to a Gaussian blur, wasting computation.
- The bilateral filter works on color images — the intensity comparison uses all three channels, which produces better edge detection than per-channel processing.
- For portrait/face smoothing, bilateral filtering is the classic technique: `cv2.bilateralFilter(face, 9, 75, 75)` smooths skin while keeping eyes, nose, and mouth edges crisp.
- Multiple passes of bilateral filtering with small parameters often produce better results than a single pass with large parameters.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with distinct regions and edges
img = np.zeros((250, 350, 3), dtype=np.uint8)

# Gradient background (smooth region that should stay smooth)
for x in range(350):
    img[:, x] = (int(180 * x / 350), int(100 + 80 * x / 350), int(200 - 100 * x / 350))

# Sharp-edged shapes (edges that should be preserved)
cv2.rectangle(img, (40, 30), (150, 110), (220, 60, 40), -1)
cv2.circle(img, (250, 70), 50, (40, 180, 60), -1)
cv2.rectangle(img, (60, 150), (290, 220), (50, 50, 200), -1)

# Add Gaussian noise
noise = np.random.randn(*img.shape).astype(np.float64) * 25
noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# --- Compare Gaussian blur vs bilateral filter ---
gauss = cv2.GaussianBlur(noisy, (9, 9), 0)
bilateral = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

# --- Vary sigmaColor ---
low_color = cv2.bilateralFilter(noisy, d=9, sigmaColor=25, sigmaSpace=75)
high_color = cv2.bilateralFilter(noisy, d=9, sigmaColor=150, sigmaSpace=75)

# --- Multiple passes for stronger effect ---
multi_pass = noisy.copy()
for _ in range(3):
    multi_pass = cv2.bilateralFilter(multi_pass, d=9, sigmaColor=50, sigmaSpace=50)

# Label helper
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Row 1: Gaussian vs Bilateral
row1 = np.hstack([
    label(noisy, 'Noisy'),
    label(gauss, 'Gaussian 9x9'),
    label(bilateral, 'Bilateral'),
])

# Row 2: sigmaColor effect + multi-pass
row2 = np.hstack([
    label(low_color, 'sigmaColor=25'),
    label(high_color, 'sigmaColor=150'),
    label(multi_pass, '3-pass bilateral'),
])

result = np.vstack([row1, row2])

# Measure edge sharpness difference
gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
gray_gauss = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY).astype(float)
gray_bilat = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY).astype(float)
print(f'Edge loss (Gaussian): {np.std(gray_gauss - gray_orig):.1f}')
print(f'Edge loss (Bilateral): {np.std(gray_bilat - gray_orig):.1f}')
print('Lower = closer to original (bilateral should be lower)')

cv2.imshow('Bilateral Filter', result)
```
