---
slug: 13-image-blending
title: Image Blending
level: beginner
concepts: [cv2.addWeighted, alpha blending, transparency]
prerequisites: [12-image-arithmetic]
---

## What Problem Are We Solving?

Sometimes you want to combine two images so that both are partially visible — like a crossfade transition in a video, a watermark overlay, or mixing two photos together. Simple addition (`cv2.add`) would just brighten pixels and clip at 255. What you actually need is **weighted blending**, where you control how much of each image contributes to the result.

## The cv2.addWeighted Formula

OpenCV gives you `cv2.addWeighted()` for this:

```python
result = cv2.addWeighted(src1, alpha, src2, beta, gamma)
```

This computes, **for every pixel**:

```
result = alpha * src1 + beta * src2 + gamma
```

| Parameter | Meaning |
|---|---|
| `src1` | First input image |
| `alpha` | Weight of the first image (0.0 to 1.0) |
| `src2` | Second input image (**must be same size and type** as src1) |
| `beta` | Weight of the second image (0.0 to 1.0) |
| `gamma` | Scalar added to every pixel (brightness offset) |

A typical use sets `alpha + beta = 1.0` so the overall brightness stays the same:

```python
# 70% of image1, 30% of image2
blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
```

If `alpha + beta > 1.0`, the result gets brighter. If `< 1.0`, it gets darker.

## How Alpha Blending Works

Think of `alpha` as **opacity**. When `alpha = 1.0`, you see only `src1`. When `alpha = 0.0`, you see only `src2`. Values in between give you a smooth mix:

```python
# Fade effect: gradually shift from img1 to img2
for alpha in [1.0, 0.75, 0.5, 0.25, 0.0]:
    beta = 1.0 - alpha
    blend = cv2.addWeighted(img1, alpha, img2, beta, 0)
```

This is exactly how **crossfade transitions** work in video editing.

## Important: Images Must Match

Both images **must** have the same shape (height, width, channels) and data type. If they don't, you'll get an error:

```python
# This FAILS if img1 and img2 have different sizes
result = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# Fix: resize to match
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
result = cv2.addWeighted(img1, 0.5, img2_resized, 0.5, 0)
```

## The Gamma Parameter

The `gamma` value is added to every pixel after blending. It's useful for brightness adjustment:

```python
# Blend and brighten by 50
result = cv2.addWeighted(img1, 0.5, img2, 0.5, 50)

# Blend and darken by 30
result = cv2.addWeighted(img1, 0.5, img2, 0.5, -30)
```

## Why Not Just Use Numpy Math?

You could do `result = (0.5 * img1 + 0.5 * img2).astype(np.uint8)`, but this has problems:

- No **saturation clipping** — values above 255 wrap around instead of capping at 255.
- Slower than OpenCV's optimized C++ implementation.
- `cv2.addWeighted` handles the `uint8` clipping automatically.

## Tips & Common Mistakes

- Both images **must be the same size and type** — use `cv2.resize()` if they differ.
- `alpha + beta = 1.0` keeps brightness consistent. Other sums change overall brightness.
- `gamma` is added **after** the weighted sum — it shifts all pixel values up or down.
- Values are clipped to `[0, 255]` automatically — no overflow or underflow.
- To create a fade-in effect, loop `alpha` from 0.0 to 1.0 in small steps.
- `cv2.addWeighted` works on all channels simultaneously — no need to process B, G, R separately.

## Starter Code

```python
import cv2
import numpy as np

# Create two colored images of the same size
h, w = 300, 400
img1 = np.zeros((h, w, 3), dtype=np.uint8)
img2 = np.zeros((h, w, 3), dtype=np.uint8)

# Image 1: blue-to-green gradient (left to right)
for x in range(w):
    ratio = x / w
    img1[:, x] = (int(255 * (1 - ratio)), int(255 * ratio), 0)  # Blue fades, Green grows

# Image 2: red rectangle on yellow background
img2[:] = (0, 255, 255)  # Yellow background
cv2.rectangle(img2, (80, 60), (320, 240), (0, 0, 255), -1)  # Red rectangle

# --- Blend with different alpha values ---
blend_75 = cv2.addWeighted(img1, 0.75, img2, 0.25, 0)  # 75% img1
blend_50 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)    # 50/50
blend_25 = cv2.addWeighted(img1, 0.25, img2, 0.75, 0)   # 75% img2

# --- Blend with gamma (brightness boost) ---
blend_gamma = cv2.addWeighted(img1, 0.5, img2, 0.5, 40)  # 50/50 + brightness

# Add labels to each image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1, 'Image 1', (10, 30), font, 0.8, (255, 255, 255), 2)
cv2.putText(img2, 'Image 2', (10, 30), font, 0.8, (0, 0, 0), 2)
cv2.putText(blend_75, 'alpha=0.75', (10, 30), font, 0.8, (255, 255, 255), 2)
cv2.putText(blend_50, 'alpha=0.50', (10, 30), font, 0.8, (255, 255, 255), 2)
cv2.putText(blend_25, 'alpha=0.25', (10, 30), font, 0.8, (255, 255, 255), 2)
cv2.putText(blend_gamma, 'a=0.5 g=40', (10, 30), font, 0.8, (255, 255, 255), 2)

# Stack results: top row = originals, middle = blends, bottom = gamma blend
top_row = np.hstack([img1, img2])
mid_row = np.hstack([blend_75, blend_50])
bot_row = np.hstack([blend_25, blend_gamma])
result = np.vstack([top_row, mid_row, bot_row])

print(f'Image 1 shape: {img1.shape}')
print(f'Image 2 shape: {img2.shape}')
print('Blended with alpha = 0.75, 0.50, 0.25, and gamma = 40')

cv2.imshow('Image Blending', result)
```
