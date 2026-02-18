---
slug: 22-median-blur
title: Median Blur
level: intermediate
concepts: [cv2.medianBlur, salt-and-pepper noise]
prerequisites: [20-averaging-box-filter]
---

## What Problem Are We Solving?

Averaging and Gaussian filters smooth noise by computing a **weighted mean** of neighboring pixels. This works well for Gaussian noise (random variations), but fails badly against **salt-and-pepper noise** — isolated extreme-value pixels (pure black or pure white dots). A single outlier pixel drastically shifts the mean. The **median filter** solves this by replacing each pixel with the **median** (middle value) of its neighborhood, making it immune to outliers.

## Why Median Preserves Edges

Consider a 1D example — a row of pixels crossing an edge:

```
Pixel values: [50, 52, 48, 51, 200, 198, 201, 199, 203]
                            ^--- edge here
```

With a **mean filter** (size 5) at the edge pixel (200):
```
Mean of [48, 51, 200, 198, 201] = 139.6 → edge gets smeared
```

With a **median filter** (size 5) at the same pixel:
```
Sorted: [48, 51, 198, 200, 201] → median = 198 → edge stays sharp!
```

The median "votes" for the majority — if most pixels in the window are bright, the result is bright. There's no averaging that creates in-between values at edges.

## Salt-and-Pepper Noise

Salt-and-pepper noise consists of random pixels set to either 0 (pepper/black) or 255 (salt/white). It commonly appears due to sensor errors, transmission glitches, or dead pixels:

```python
# Simulate salt-and-pepper noise
def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white pixels)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    noisy[ys, xs] = 255

    # Pepper (black pixels)
    ys = np.random.randint(0, h, num_pixels)
    xs = np.random.randint(0, w, num_pixels)
    noisy[ys, xs] = 0

    return noisy
```

Applying a Gaussian blur to salt-and-pepper noise just **smears** the bright/dark dots into gray halos. The median filter **eliminates** them entirely because the outlier values never win the vote.

## Using cv2.medianBlur()

```python
denoised = cv2.medianBlur(img, ksize=5)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image (uint8 or float32) |
| `ksize` | Kernel size — **must be odd and positive**: 3, 5, 7, ... |

Note: unlike `cv2.blur()` and `cv2.GaussianBlur()`, the kernel size is a **single integer**, not a tuple. Median blur always uses a square kernel.

```python
# Different kernel sizes
mild   = cv2.medianBlur(img, 3)   # 3x3 neighborhood
medium = cv2.medianBlur(img, 5)   # 5x5 neighborhood
strong = cv2.medianBlur(img, 7)   # 7x7 neighborhood
```

## Kernel Size Must Be Odd

The kernel size for `cv2.medianBlur()` **must be an odd positive integer**. An even number will raise an error:

```python
# This will crash:
# result = cv2.medianBlur(img, 4)  # Error! ksize must be odd

# These are valid:
result = cv2.medianBlur(img, 3)
result = cv2.medianBlur(img, 5)
result = cv2.medianBlur(img, 9)
```

The reason is that an odd-sized kernel has a single, well-defined center pixel to replace.

## Median vs Gaussian for Different Noise Types

```python
# Gaussian noise: random variations around the true value
gaussian_noise = np.random.randn(*img.shape) * 25
noisy_gaussian = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)

# Salt-and-pepper noise: extreme outlier pixels
noisy_sp = add_salt_pepper(img, amount=0.05)

# Gaussian blur handles Gaussian noise well
gauss_fix = cv2.GaussianBlur(noisy_gaussian, (5, 5), 0)   # Good
gauss_fix_sp = cv2.GaussianBlur(noisy_sp, (5, 5), 0)      # Bad — smears dots

# Median blur handles salt-and-pepper perfectly
median_fix = cv2.medianBlur(noisy_sp, 5)                   # Excellent
median_fix_gaussian = cv2.medianBlur(noisy_gaussian, 5)    # Decent but can look "painterly"
```

> **Summary:** Use Gaussian blur for Gaussian noise. Use median blur for salt-and-pepper noise. For mixed noise, apply median first (to remove outliers), then Gaussian (to smooth remaining noise).

## Tips & Common Mistakes

- Kernel size is a **single integer**, not a tuple: `cv2.medianBlur(img, 5)` not `cv2.medianBlur(img, (5, 5))`.
- Kernel size **must be odd** — 3, 5, 7, 9, etc. Even values cause an error.
- Median blur is **slower** than Gaussian blur for large kernels because sorting is O(n log n) vs O(n). For kernel sizes beyond 7, consider whether you really need median.
- Large median kernels (9+) create a "painting-like" effect — this is sometimes used intentionally as an artistic filter.
- For color images, median blur processes all channels together (unlike Gaussian, which processes each independently). This generally produces better results for color images.
- Median blur perfectly removes salt-and-pepper noise as long as the noise density is below ~25% of pixels. Beyond that, even median blur struggles.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with clear edges and detail
img = np.zeros((250, 350, 3), dtype=np.uint8)
img[:] = (180, 180, 180)

# Draw shapes with sharp edges
cv2.rectangle(img, (20, 20), (120, 100), (200, 60, 40), -1)
cv2.circle(img, (230, 65), 45, (40, 160, 60), -1)
cv2.line(img, (20, 140), (330, 140), (0, 0, 0), 2)
cv2.putText(img, 'Median', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)

# --- Add salt-and-pepper noise ---
def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    h, w = image.shape[:2]
    n = int(amount * h * w)
    # Salt
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 255
    # Pepper
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 0
    return noisy

noisy = add_salt_pepper(img, amount=0.05)

# --- Apply different filters to salt-and-pepper noise ---
gauss_fix = cv2.GaussianBlur(noisy, (5, 5), 0)
median_3 = cv2.medianBlur(noisy, 3)
median_5 = cv2.medianBlur(noisy, 5)
median_9 = cv2.medianBlur(noisy, 9)

# Add labels
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Row 1: Gaussian vs Median on salt-and-pepper noise
row1 = np.hstack([
    label(noisy, 'Salt & Pepper'),
    label(gauss_fix, 'Gaussian 5x5'),
    label(median_5, 'Median 5'),
])

# Row 2: Median kernel size comparison
row2 = np.hstack([
    label(img, 'Original'),
    label(median_3, 'Median 3'),
    label(median_9, 'Median 9'),
])

result = np.vstack([row1, row2])

# Count remaining noisy pixels
diff = cv2.absdiff(median_5, img)
remaining_noise = np.sum(diff > 30)
print(f'Noisy pixels remaining after median 5: {remaining_noise}')
print(f'Salt-and-pepper noise is {("removed" if remaining_noise < 100 else "partially removed")}')

cv2.imshow('Median Blur', result)
```
