---
slug: 53-color-quantization
title: Color Quantization
level: intermediate
concepts: [K-means clustering, cv2.kmeans, color reduction]
prerequisites: [03-pixel-access]
---

## What Problem Are We Solving?

A typical color image uses up to 16.7 million possible colors (256 x 256 x 256). But what if you want to reduce it to just 8, 16, or 64 colors? This is **color quantization** -- replacing the full color palette with a small set of representative colors. It's useful for creating poster-like effects, reducing file size, simplifying images for analysis, or generating color palettes from photographs.

## How K-Means Clustering Works for Images

K-means is an algorithm that groups data points into **K clusters**, where each point belongs to the cluster with the nearest center. For images:

1. Treat every pixel as a 3D point (B, G, R).
2. K-means finds K cluster centers (the "representative colors").
3. Every pixel gets replaced by the color of its nearest cluster center.

The result: an image with exactly K distinct colors.

## Reshaping the Image for K-Means

`cv2.kmeans()` expects a 2D array of float32 data points -- one row per pixel. An image of shape `(H, W, 3)` needs to be reshaped to `(H*W, 3)`:

```python
pixel_data = img.reshape((-1, 3)).astype(np.float32)
```

| Original Shape | Reshaped Shape | Meaning |
|---|---|---|
| `(300, 400, 3)` | `(120000, 3)` | 120,000 pixels, each with 3 color values |

The `-1` tells NumPy to compute the first dimension automatically. The `float32` conversion is required because `cv2.kmeans()` only works with floating-point data.

## Using cv2.kmeans()

The function signature has several parameters:

```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = 8
_, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
```

| Parameter | Meaning |
|---|---|
| `pixel_data` | Input data as `float32` array of shape `(N, 3)` |
| `K` | Number of clusters (colors in the output) |
| `None` | Best labels output (pass `None` to let OpenCV allocate) |
| `criteria` | When to stop iterating |
| `10` | Number of attempts with different initial centers |
| `cv2.KMEANS_RANDOM_CENTERS` | How to initialize cluster centers |

## Understanding the Criteria Parameter

The criteria tuple controls when K-means stops iterating:

```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
```

- `cv2.TERM_CRITERIA_EPS` -- stop when cluster centers move less than `epsilon`.
- `cv2.TERM_CRITERIA_MAX_ITER` -- stop after `max_iter` iterations.
- Combining both (with `+`) means "stop when either condition is met."

Typical values: `max_iter=20`, `epsilon=1.0`. More iterations give better results but take longer.

## Understanding the Attempts Parameter

K-means results depend on the initial random placement of cluster centers. The `attempts` parameter runs the algorithm multiple times with different random starts and returns the best result:

```python
# Run 10 times, keep the best clustering
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
```

More attempts = better results but slower. For most images, 10 attempts is sufficient.

## Reconstructing the Quantized Image

After K-means, you have `labels` (which cluster each pixel belongs to) and `centers` (the representative colors). Reconstruct the image:

```python
centers = np.uint8(centers)                  # Convert centers back to uint8
quantized = centers[labels.flatten()]        # Map each pixel to its cluster color
quantized = quantized.reshape(img.shape)     # Reshape back to image dimensions
```

The key line `centers[labels.flatten()]` uses the label of each pixel as an index into the centers array, replacing every pixel with its cluster's representative color.

## Effect of Different K Values

```python
# Few colors: strong poster effect, loses detail
K = 4   # Very few colors -- abstract look

# Moderate: recognizable but stylized
K = 16  # Good balance of simplification and detail

# Many colors: subtle reduction, close to original
K = 64  # Hard to tell apart from the original
```

## Tips & Common Mistakes

- Always convert to `float32` before passing to `cv2.kmeans()`. Passing `uint8` data causes an error.
- Remember to reshape the image to `(N, 3)` and reshape back to `(H, W, 3)` after processing.
- Convert `centers` back to `uint8` before reconstructing the image. They come back as `float32`.
- Lower K values produce more dramatic effects but can lose important details. Start with K=8 for visible but reasonable quantization.
- The `attempts` parameter significantly affects quality for small K values. Use at least 10 attempts.
- K-means on large images can be slow. Consider resizing the image first, computing clusters, then applying to the full-resolution image.
- `labels` comes back as a column vector of shape `(N, 1)` -- use `.flatten()` before indexing.
- This technique also works in other color spaces (LAB, HSV) which may give perceptually better results.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful gradient image with shapes
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Create a smooth gradient background
for y in range(300):
    for x in range(400):
        img[y, x] = (
            int(255 * x / 400),           # Blue increases left to right
            int(255 * y / 300),            # Green increases top to bottom
            int(255 * (1 - x / 400))       # Red decreases left to right
        )

# Add some shapes for contrast
cv2.circle(img, (200, 150), 80, (0, 200, 255), -1)
cv2.rectangle(img, (30, 200), (150, 290), (255, 100, 0), -1)
cv2.ellipse(img, (330, 230), (60, 40), 30, 0, 360, (100, 255, 100), -1)

# --- Reshape image for K-means ---
pixel_data = img.reshape((-1, 3)).astype(np.float32)

# --- K-means criteria ---
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
attempts = 10
flags = cv2.KMEANS_RANDOM_CENTERS

# --- Quantize with different K values ---
results = []
for K in [4, 8, 16]:
    _, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, attempts, flags)

    # Reconstruct the quantized image
    centers_uint8 = np.uint8(centers)
    quantized = centers_uint8[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    # Label the result
    cv2.putText(quantized, f'K={K}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    results.append(quantized)

# Label the original
original = img.copy()
cv2.putText(original, 'Original', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Build display: original + three quantized versions
top_row = np.hstack([original, results[0]])
bottom_row = np.hstack([results[1], results[2]])
display = np.vstack([top_row, bottom_row])

print(f'Original image shape: {img.shape}')
print(f'Pixel data shape for K-means: {pixel_data.shape}')
print(f'Original unique colors: {len(np.unique(img.reshape(-1, 3), axis=0))}')
for K, res in zip([4, 8, 16], results):
    unique = len(np.unique(res.reshape(-1, 3), axis=0))
    print(f'K={K} unique colors: {unique}')

cv2.imshow('Color Quantization', display)
```
