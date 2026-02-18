---
slug: 07-image-resizing
title: Image Resizing
level: beginner
concepts: [cv2.resize, interpolation, INTER_LINEAR, INTER_AREA, INTER_CUBIC]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

Images come in all sizes. You might need to shrink a 4000x3000 photo to fit a display, enlarge a thumbnail for inspection, or standardize all images to the same dimensions before feeding them to a machine learning model. `cv2.resize()` handles all of these cases, and the **interpolation method** you choose affects both quality and speed.

## Resizing to an Explicit Size

The simplest form of `cv2.resize()` takes an image and a target `(width, height)` tuple:

```python
resized = cv2.resize(img, (new_width, new_height))
```

| Parameter | Meaning |
|---|---|
| `img` | Source image |
| `(new_width, new_height)` | Target size as `(width, height)` — **not** `(height, width)` |

> **Critical trap:** The size tuple is `(width, height)`, which is the **opposite** of NumPy's `shape` order `(height, width, channels)`. This is the most common mistake beginners make with `cv2.resize()`.

```python
img = np.zeros((300, 400, 3), dtype=np.uint8)  # 400 wide, 300 tall
resized = cv2.resize(img, (200, 150))            # 200 wide, 150 tall
print(resized.shape)  # (150, 200, 3) — note shape is (H, W, C)
```

## Resizing with Scale Factors (fx, fy)

Instead of specifying exact pixel dimensions, you can scale by a factor. Pass `(0, 0)` as the size and use `fx` and `fy`:

```python
# Double the width and height
bigger = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)

# Shrink to 50% in both directions
smaller = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
```

- `fx` — scale factor for the **width** (x-axis)
- `fy` — scale factor for the **height** (y-axis)

This is handy when you want to scale proportionally without computing exact pixel sizes.

## Preserving Aspect Ratio

If you resize to an arbitrary width and height, the image may get distorted (stretched or squished). To preserve the aspect ratio, compute one dimension from the other:

```python
# Resize to a target width, keeping aspect ratio
target_width = 300
h, w = img.shape[:2]
scale = target_width / w
new_h = int(h * scale)
resized = cv2.resize(img, (target_width, new_h))
```

```python
# Resize to a target height, keeping aspect ratio
target_height = 200
h, w = img.shape[:2]
scale = target_height / h
new_w = int(w * scale)
resized = cv2.resize(img, (new_w, target_height))
```

## Interpolation Methods

Interpolation determines how pixel values are calculated when the image is scaled. Different methods offer different trade-offs:

| Method | Best For | Quality | Speed |
|---|---|---|---|
| `cv2.INTER_NEAREST` | Pixel art, masks | Lowest (blocky) | Fastest |
| `cv2.INTER_LINEAR` | General upscaling (default) | Good | Fast |
| `cv2.INTER_AREA` | **Shrinking** images | Best for downscale | Moderate |
| `cv2.INTER_CUBIC` | High-quality upscaling | Better than linear | Slower |
| `cv2.INTER_LANCZOS4` | Highest-quality upscaling | Best for upscale | Slowest |

```python
# Shrinking — use INTER_AREA to avoid aliasing artifacts
small = cv2.resize(img, (100, 75), interpolation=cv2.INTER_AREA)

# Enlarging — use INTER_CUBIC for quality, INTER_LINEAR for speed
big = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
```

**Rule of thumb:**
- **Making the image smaller?** Use `cv2.INTER_AREA`. It averages pixel neighborhoods and avoids moire patterns.
- **Making the image bigger?** Use `cv2.INTER_LINEAR` (fast, decent quality) or `cv2.INTER_CUBIC` (slower, better quality).

## Why Interpolation Matters

When you enlarge an image from 100x100 to 400x400, the new image has 16x more pixels than the original. Those new pixel values must be **invented** — interpolation is the algorithm that decides what those new values should be.

- `INTER_NEAREST` just copies the nearest original pixel (blocky, but preserves hard edges in masks).
- `INTER_LINEAR` averages the 4 nearest pixels (smooth, good default).
- `INTER_CUBIC` uses 16 nearby pixels with a cubic polynomial (smoother still).
- `INTER_AREA` uses pixel area relation (ideal for decimation/shrinking).

## Tips & Common Mistakes

- The size argument is `(width, height)`, but `img.shape` returns `(height, width, channels)`. Double-check the order every time.
- When shrinking, always prefer `cv2.INTER_AREA`. Using `INTER_LINEAR` or `INTER_CUBIC` for downscaling can produce aliasing artifacts (jagged edges, moire patterns).
- `cv2.resize()` returns a **new** image — it does not modify the original in place.
- If `fx` and `fy` are provided along with a non-zero size tuple, the size tuple takes precedence and the scale factors are ignored.
- Resizing loses information permanently. If you shrink an image and then enlarge it back, it will look blurry — you cannot recover the lost detail.
- Always use `int()` when computing dimensions from floating-point math. `cv2.resize()` requires integer sizes.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful source image with text for visual clarity
img = np.zeros((200, 300, 3), dtype=np.uint8)
img[:] = (50, 40, 30)

# Draw colored regions so resizing effects are visible
cv2.rectangle(img, (20, 20), (140, 90), (0, 180, 255), -1)   # Orange box
cv2.rectangle(img, (160, 20), (280, 90), (255, 100, 50), -1)  # Blue box
cv2.rectangle(img, (20, 110), (280, 180), (50, 200, 50), -1)  # Green bar

# Add text label
cv2.putText(img, 'Original', (80, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, f'{img.shape[1]}x{img.shape[0]}', (80, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# --- Resize to explicit size ---
resized_exact = cv2.resize(img, (150, 100))

# --- Resize with scale factors ---
resized_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# --- Resize preserving aspect ratio (target width = 150) ---
target_w = 150
h, w = img.shape[:2]
scale = target_w / w
resized_aspect = cv2.resize(img, (target_w, int(h * scale)))

# --- Compare interpolation methods (enlarge 2x) ---
up_nearest = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
up_linear = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
up_cubic = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

# --- Build a comparison display ---
# Resize all small variants to the same height for side-by-side display
display_h = 200
def fit_to_height(image, target_h):
    ih, iw = image.shape[:2]
    s = target_h / ih
    return cv2.resize(image, (int(iw * s), target_h), interpolation=cv2.INTER_LINEAR)

col1 = fit_to_height(img, display_h)
col2 = fit_to_height(resized_exact, display_h)
col3 = fit_to_height(resized_half, display_h)
col4 = fit_to_height(resized_aspect, display_h)

# Label each column
for label, col in [('Original', col1), ('Exact 150x100', col2),
                   ('fx=0.5 fy=0.5', col3), ('Aspect-fit', col4)]:
    cv2.putText(col, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 255), 1, cv2.LINE_AA)

row = np.hstack([col1, col2, col3, col4])

print(f'Original shape:       {img.shape}')
print(f'Exact resize shape:   {resized_exact.shape}')
print(f'Half scale shape:     {resized_half.shape}')
print(f'Aspect-fit shape:     {resized_aspect.shape}')
print(f'2x nearest shape:     {up_nearest.shape}')

cv2.imshow('Image Resizing', row)
```
