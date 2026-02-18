---
slug: 31-laplacian-edge-detection
title: Laplacian Edge Detection
level: intermediate
concepts: [cv2.Laplacian, second-order derivatives]
prerequisites: [29-sobel-edge-detection]
---

## What Problem Are We Solving?

Sobel and Scharr compute **first-order derivatives** — they detect edges by finding where intensity is changing rapidly. The **Laplacian** computes a **second-order derivative**, which detects edges by finding where the rate of change itself changes — the **zero-crossings** of the second derivative correspond to edge locations. The advantage: it finds edges in **all directions at once** without needing separate x and y passes.

## What Is the Laplacian?

Mathematically, the Laplacian is the sum of second derivatives in x and y:

```
Laplacian(f) = d²f/dx² + d²f/dy²
```

The simplest discrete kernel looks like:

```
[0   1  0]
[1  -4  1]
[0   1  0]
```

A pixel is highlighted when it differs strongly from its neighbors — exactly what happens at an edge.

## Using cv2.Laplacian()

```python
laplacian = cv2.Laplacian(src, ddepth, ksize=3)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image (typically grayscale) |
| `ddepth` | Output depth — use `cv2.CV_64F` to capture both positive and negative values |
| `ksize` | Kernel size (1, 3, 5, or 7). `ksize=1` uses a simple 3x3 kernel; larger sizes add Gaussian smoothing |

> **Key insight:** Since the Laplacian uses second derivatives, its output contains **both positive and negative** values. A dark-to-bright edge produces a positive response; a bright-to-dark edge produces negative. Use `cv2.CV_64F` to preserve both, then take the absolute value for display.

## Why Laplacian Is Noise-Sensitive

Second derivatives amplify noise much more than first derivatives. A tiny random fluctuation in pixel values creates a large second-derivative spike. This is why you almost always need to **blur first**:

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
```

This two-step process is known as the **Laplacian of Gaussian (LoG)**.

## Laplacian of Gaussian (LoG)

The LoG approach combines smoothing and edge detection:

1. Apply Gaussian blur to suppress noise
2. Apply Laplacian to detect edges

```python
# Manual LoG
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
```

Increasing the Gaussian kernel size detects **coarser** edges while ignoring finer details and noise.

## Choosing ksize

- `ksize=1` — uses the basic `[[0,1,0],[1,-4,1],[0,1,0]]` kernel. Very sensitive to noise.
- `ksize=3` — applies some internal Gaussian smoothing. Good default.
- `ksize=5` or `ksize=7` — more smoothing built in. Detects broader edges, less noise.

## Tips & Common Mistakes

- Always use `ddepth=cv2.CV_64F` to avoid losing negative values. Then convert with `cv2.convertScaleAbs()` for display.
- The Laplacian is **very sensitive to noise**. Always apply Gaussian blur before using it on real images.
- Unlike Sobel, you don't need to compute x and y separately — the Laplacian captures all directions in one pass.
- The output shows **both sides** of an edge (positive and negative response). For thin edges, you'd look for zero-crossings, but `convertScaleAbs` gives you thick, visible edges.
- For a cleaner edge map, consider Canny instead. Laplacian is better for understanding image structure than for binary edge detection.
- Larger `ksize` values produce smoother results but thicker edges.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with shapes
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (50, 50, 50)

# Draw shapes for edge detection
cv2.rectangle(img, (40, 40), (200, 180), (200, 200, 200), -1)
cv2.circle(img, (350, 110), 80, (220, 220, 220), -1)
cv2.ellipse(img, (500, 300), (80, 50), 30, 0, 360, (180, 180, 180), -1)
cv2.rectangle(img, (40, 240), (250, 370), (160, 160, 160), -1)

# Add some text to create fine detail
cv2.putText(img, 'Edge', (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Direct Laplacian (no blur) ---
lap_noisy = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
lap_noisy_abs = cv2.convertScaleAbs(lap_noisy)

# --- Laplacian of Gaussian (blur first) ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
lap_clean = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
lap_clean_abs = cv2.convertScaleAbs(lap_clean)

# --- Different kernel sizes ---
lap_k1 = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=1))
lap_k3 = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3))
lap_k5 = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=5))

# --- Build comparison display ---
def to_bgr(gray_img):
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX

# Label each image
panels = [
    (img, 'Original'),
    (to_bgr(lap_noisy_abs), 'Laplacian (no blur)'),
    (to_bgr(lap_clean_abs), 'LoG (blur + Laplacian)'),
    (to_bgr(lap_k1), 'ksize=1'),
    (to_bgr(lap_k3), 'ksize=3'),
    (to_bgr(lap_k5), 'ksize=5'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

top_row = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
bottom_row = np.hstack([panels[3][0], panels[4][0], panels[5][0]])
result = np.vstack([top_row, bottom_row])

print(f'Laplacian value range (raw): {lap_clean.min():.1f} to {lap_clean.max():.1f}')
print(f'Notice both positive and negative values (edges from both sides)')

cv2.imshow('Laplacian Edge Detection', result)
```
