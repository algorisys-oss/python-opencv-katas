---
slug: 20-averaging-box-filter
title: "Smoothing: Averaging & Box Filter"
level: intermediate
concepts: [cv2.blur, cv2.boxFilter, kernel size]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

Images captured from cameras or sensors often contain **noise** — random pixel-level variations that don't belong to the actual scene. Smoothing (also called blurring) reduces this noise by averaging neighboring pixel values. The simplest forms of smoothing are the **averaging filter** and the **box filter**, which replace each pixel with the mean of its surrounding neighborhood.

## How Averaging Works

Imagine sliding a small window (called a **kernel**) across every pixel in the image. At each position, you compute the **average** of all pixel values under the kernel and replace the center pixel with that average. For a 3x3 kernel, you're averaging 9 pixels; for a 5x5, you're averaging 25 pixels.

The kernel for a 3x3 averaging filter looks like this mathematically:

```
K = (1/9) * | 1 1 1 |
             | 1 1 1 |
             | 1 1 1 |
```

Every pixel contributes **equally** to the result — that's what makes it an "averaging" filter.

## Using cv2.blur()

`cv2.blur()` is the simplest smoothing function in OpenCV. It applies a normalized box filter:

```python
blurred = cv2.blur(img, ksize=(5, 5))
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `ksize` | Kernel size as `(width, height)` — must be positive and odd is recommended |

The kernel size controls **how much smoothing** you get:

```python
light  = cv2.blur(img, (3, 3))    # Mild smoothing
medium = cv2.blur(img, (7, 7))    # Moderate smoothing
heavy  = cv2.blur(img, (15, 15))  # Strong smoothing — image gets very blurry
```

Larger kernels = more blur = more noise removed, but also more detail lost.

## Using cv2.boxFilter()

`cv2.boxFilter()` gives you more control than `cv2.blur()`. The key extra parameter is `normalize`:

```python
# With normalize=True (default) — identical to cv2.blur()
blurred = cv2.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=True)

# With normalize=False — sums pixel values without dividing
summed = cv2.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=False)
```

| Parameter | Meaning |
|---|---|
| `ddepth` | Output depth. Use `-1` to keep the same depth as input |
| `ksize` | Kernel size as `(width, height)` |
| `normalize` | If `True`, divides by kernel area (averaging). If `False`, just sums. |

When `normalize=False`, pixel values are **summed** instead of averaged. This can easily exceed 255 and saturate to white, which is useful for specific tasks like computing local sums but not for typical smoothing.

## Kernel Size Effects

The kernel size has a dramatic impact on the output:

```python
# Small kernel: subtle smoothing, preserves most detail
subtle = cv2.blur(img, (3, 3))

# Medium kernel: noticeable blur, good for noise reduction
moderate = cv2.blur(img, (9, 9))

# Large kernel: heavy blur, useful for background extraction
heavy = cv2.blur(img, (25, 25))
```

> **Rule of thumb:** Start with a 3x3 or 5x5 kernel and increase only if you need more smoothing. Every doubling of the kernel size significantly increases computation time and detail loss.

## Non-Square Kernels

You can use rectangular kernels for directional blurring:

```python
# Horizontal blur only — creates motion-blur effect
h_blur = cv2.blur(img, (15, 1))

# Vertical blur only
v_blur = cv2.blur(img, (1, 15))
```

This is useful when you want to blur in one direction while preserving sharpness in the other.

## Tips & Common Mistakes

- Kernel size should use **odd numbers** (3, 5, 7, ...) so the kernel has a clear center pixel. Even sizes work but are less intuitive.
- `cv2.blur()` is just `cv2.boxFilter()` with `normalize=True` and `ddepth=-1`.
- Averaging blurs **everything equally** — it doesn't distinguish between edges and flat regions. This means edges get smeared. For edge-preserving smoothing, look at bilateral filtering.
- With `normalize=False`, output values can overflow for `uint8` images. OpenCV clips them to 255, which usually makes the image appear mostly white.
- `ksize` is `(width, height)` not `(height, width)` — but for square kernels this doesn't matter.
- Smoothing a color image works on each channel independently — colors can shift slightly in transition regions.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with shapes and added noise
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (200, 200, 200)

# Draw some shapes for visual reference
cv2.rectangle(img, (30, 30), (120, 120), (255, 0, 0), -1)
cv2.circle(img, (250, 80), 50, (0, 180, 0), -1)
cv2.line(img, (320, 30), (380, 130), (0, 0, 255), 3)
cv2.putText(img, 'OpenCV', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)

# Add random noise to simulate a noisy capture
noise = np.random.randint(-40, 40, img.shape, dtype=np.int16)
noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Averaging with cv2.blur() at different kernel sizes ---
blur_3 = cv2.blur(noisy, (3, 3))
blur_7 = cv2.blur(noisy, (7, 7))
blur_15 = cv2.blur(noisy, (15, 15))

# --- Box filter with normalize on vs off ---
box_norm = cv2.boxFilter(noisy, ddepth=-1, ksize=(7, 7), normalize=True)
box_sum = cv2.boxFilter(noisy, ddepth=-1, ksize=(7, 7), normalize=False)

# --- Directional blur ---
h_blur = cv2.blur(noisy, (15, 1))

# Add labels to each image
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Build comparison: top row = blur sizes, bottom row = box filter variants
row1 = np.hstack([
    label(noisy, 'Noisy'),
    label(blur_3, 'blur 3x3'),
    label(blur_7, 'blur 7x7'),
    label(blur_15, 'blur 15x15'),
])

row2 = np.hstack([
    label(img, 'Original'),
    label(box_norm, 'boxFilter norm'),
    label(box_sum, 'boxFilter sum'),
    label(h_blur, 'Horizontal blur'),
])

result = np.vstack([row1, row2])

print(f'Original shape: {img.shape}')
print(f'Kernel 3x3 blur shape: {blur_3.shape}')
print(f'Box filter (normalize=False) max value: {box_sum.max()}')

cv2.imshow('Averaging & Box Filter', result)
```
