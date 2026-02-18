---
slug: 24-image-sharpening
title: Image Sharpening
level: intermediate
concepts: [cv2.filter2D, sharpening kernel, unsharp mask]
prerequisites: [21-gaussian-blur]
---

## What Problem Are We Solving?

Images can appear **soft or blurry** due to camera focus issues, motion, compression artifacts, or intentional smoothing. **Sharpening** enhances edges and fine details, making the image appear crisper and more defined. Unlike blurring (which averages), sharpening **amplifies differences** between adjacent pixels — it boosts the high-frequency components of an image.

## How Sharpening Works

Sharpening works on a simple principle: **enhance the difference between a pixel and its neighbors**. Mathematically:

```
sharpened = original + strength * (original - blurred)
```

The term `(original - blurred)` extracts the **detail/edge information**. Adding it back amplifies those details. This can be done either through kernel convolution or the "unsharp mask" technique.

## Sharpening Kernels

A sharpening kernel is a small matrix that, when convolved with an image, emphasizes edges. The most common 3x3 sharpening kernel:

```
|  0  -1   0 |
| -1   5  -1 |
|  0  -1   0 |
```

Notice the center value (5) is larger than the sum of all values (which equals 1). This means:
- The center pixel is strongly weighted
- Neighboring pixels are subtracted
- The net effect amplifies local contrast (edges)

Other common sharpening kernels:

```python
# Mild sharpening (center = 5, sum = 1)
mild = np.array([[0, -1, 0],
                 [-1, 5, -1],
                 [0, -1, 0]], dtype=np.float32)

# Strong sharpening (includes diagonals, center = 9, sum = 1)
strong = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Extreme sharpening (sum > 1, will brighten overall)
extreme = np.array([[-1, -1, -1],
                    [-1, 10, -1],
                    [-1, -1, -1]], dtype=np.float32)
```

> **Key rule:** If the kernel values sum to 1, overall brightness is preserved. If greater than 1, the image brightens. If less than 1, it darkens.

## Using cv2.filter2D() for Custom Convolution

`cv2.filter2D()` applies any custom kernel to an image:

```python
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

sharpened = cv2.filter2D(img, ddepth=-1, kernel=kernel)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `ddepth` | Output depth. Use `-1` for same as input |
| `kernel` | The convolution kernel as a NumPy array |

The kernel must be a floating-point or integer NumPy array. OpenCV handles the convolution, boundary padding, and clipping automatically.

## The Unsharp Mask Technique

**Unsharp masking** is a classic sharpening technique borrowed from film photography. Despite the confusing name ("unsharp"), it **sharpens** images. The idea:

1. Create a blurred (unsharp) version of the image
2. Subtract the blur from the original to get the detail layer
3. Add the detail back to the original (scaled by a strength factor)

```python
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=1.5):
    # Step 1: Create blurred version
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Step 2 & 3: sharpened = original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return sharpened
```

`cv2.addWeighted()` computes `alpha * src1 + beta * src2 + gamma`, which gives us exactly:
```
(1 + strength) * original + (-strength) * blurred
= original + strength * (original - blurred)
```

## Controlling Sharpening Strength

The unsharp mask has three tunable parameters:

```python
# Subtle sharpening — good for most photos
subtle = unsharp_mask(img, kernel_size=(3, 3), sigma=1.0, strength=0.5)

# Medium sharpening — visible enhancement
medium = unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, strength=1.0)

# Aggressive sharpening — can introduce halos
aggressive = unsharp_mask(img, kernel_size=(7, 7), sigma=2.0, strength=2.0)
```

- **kernel_size/sigma:** Controls the radius of the detail extraction. Larger = coarser details are enhanced.
- **strength:** How much detail is added back. 0.5 = subtle, 1.0 = normal, 2.0+ = aggressive.

## Sharpening Artifacts

Over-sharpening creates visible **halos** — bright/dark outlines along edges. This happens when the strength is too high or the blur radius is too large:

```python
# This will create ugly halos around edges
over_sharpened = unsharp_mask(img, kernel_size=(15, 15), sigma=5.0, strength=3.0)
```

Always check your results at 100% zoom. What looks good zoomed-out may have severe halos when viewed at full resolution.

## Tips & Common Mistakes

- Sharpening **amplifies noise** along with detail. If your image is noisy, denoise first (Gaussian blur), then sharpen.
- Always use `dtype=np.float32` for kernels — integer kernels can cause unexpected rounding behavior.
- The unsharp mask is generally preferred over kernel-based sharpening because you can control the radius and strength independently.
- `cv2.addWeighted()` automatically clips to 0-255 for uint8 images, preventing overflow.
- Sharpening cannot recover truly lost detail (e.g., from severe defocus). It only enhances existing contrast differences.
- Apply sharpening as the **last step** in your processing pipeline, after resizing and color correction.
- For video, use conservative sharpening (strength < 1.0) to avoid flickering artifacts between frames.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with fine detail and edges
img = np.zeros((250, 350, 3), dtype=np.uint8)
img[:] = (200, 200, 200)

# Draw fine details
cv2.rectangle(img, (20, 20), (100, 100), (180, 60, 40), -1)
cv2.rectangle(img, (22, 22), (98, 98), (200, 80, 60), -1)
cv2.circle(img, (200, 60), 40, (40, 150, 60), -1)
cv2.circle(img, (200, 60), 30, (60, 180, 80), -1)
for y in range(130, 230, 8):
    cv2.line(img, (20, y), (330, y), (100, 100, 100), 1)
cv2.putText(img, 'Sharp', (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)

# Slightly blur the image to simulate soft focus
soft = cv2.GaussianBlur(img, (3, 3), 1.0)

# --- Method 1: Kernel-based sharpening ---
kernel_mild = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)

kernel_strong = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]], dtype=np.float32)

sharp_mild = cv2.filter2D(soft, -1, kernel_mild)
sharp_strong = cv2.filter2D(soft, -1, kernel_strong)

# --- Method 2: Unsharp mask ---
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

unsharp_subtle = unsharp_mask(soft, (3, 3), 1.0, 0.5)
unsharp_medium = unsharp_mask(soft, (5, 5), 1.0, 1.5)
unsharp_over = unsharp_mask(soft, (9, 9), 3.0, 3.0)

# Label helper
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Row 1: Original vs kernel sharpening
row1 = np.hstack([
    label(soft, 'Soft input'),
    label(sharp_mild, 'Kernel mild'),
    label(sharp_strong, 'Kernel strong'),
])

# Row 2: Unsharp mask variants
row2 = np.hstack([
    label(unsharp_subtle, 'Unsharp 0.5'),
    label(unsharp_medium, 'Unsharp 1.5'),
    label(unsharp_over, 'Over-sharpened'),
])

result = np.vstack([row1, row2])

print(f'Input image std (detail level): {np.std(soft):.1f}')
print(f'After mild kernel: {np.std(sharp_mild):.1f}')
print(f'After unsharp mask: {np.std(unsharp_medium):.1f}')
print('Higher std = more contrast/detail')

cv2.imshow('Image Sharpening', result)
```
