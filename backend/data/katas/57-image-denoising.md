---
slug: 57-image-denoising
title: Image Denoising
level: intermediate
concepts: [cv2.fastNlMeansDenoising, cv2.fastNlMeansDenoisingColored]
prerequisites: [21-gaussian-blur]
---

## What Problem Are We Solving?

Real-world images always contain **noise** -- random pixel-level variations caused by sensor imperfections, low light, high ISO settings, or signal interference. Simple blurring (like Gaussian blur) reduces noise but also destroys edges and fine details. **Non-Local Means denoising** is a smarter approach: it finds similar patches throughout the entire image and averages them, removing noise while preserving edges and textures far better than any blur filter.

## Why Non-Local Means is Different from Blurring

Gaussian blur averages a pixel with its **immediate neighbors**. If a pixel is on an edge, its neighbors include both sides of the edge, and the edge gets smeared.

Non-Local Means (NLM) takes a fundamentally different approach:

1. For each pixel, look at its surrounding **patch** (small neighborhood).
2. Search the image for **other patches that look similar**.
3. Average only the similar patches together.

Because similar patches can be anywhere in the image (not just nearby), the algorithm finds redundant information that local filters miss. Edges stay sharp because edge patches only get averaged with other edge patches.

## cv2.fastNlMeansDenoising() for Grayscale

For grayscale images, use `cv2.fastNlMeansDenoising()`:

```python
denoised = cv2.fastNlMeansDenoising(noisy_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
```

| Parameter | Meaning |
|---|---|
| `noisy_gray` | Input noisy grayscale image |
| `None` | Output image (pass `None` to create a new one) |
| `h` | Filter strength. Higher = more denoising but more detail loss |
| `templateWindowSize` | Size of the patch used for comparison (should be odd) |
| `searchWindowSize` | Size of the area searched for similar patches (should be odd) |

## cv2.fastNlMeansDenoisingColored() for Color Images

For BGR color images, use the colored variant:

```python
denoised = cv2.fastNlMeansDenoisingColored(noisy_color, None, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
```

| Parameter | Meaning |
|---|---|
| `noisy_color` | Input noisy BGR image |
| `h` | Filter strength for luminance component |
| `hForColorComponents` | Filter strength for color components (usually same as `h`) |
| `templateWindowSize` | Patch size for comparison |
| `searchWindowSize` | Search area size |

This function internally converts to LAB color space, denoises the luminance and color channels separately (since they have different noise characteristics), and converts back.

## The h Parameter (Filter Strength)

The `h` parameter is the most important one to tune. It controls how aggressively noise is removed:

```python
# Light denoising: preserves detail, some noise remains
light = cv2.fastNlMeansDenoising(noisy, None, h=5)

# Moderate denoising: good balance
moderate = cv2.fastNlMeansDenoising(noisy, None, h=10)

# Heavy denoising: very smooth, may lose fine detail
heavy = cv2.fastNlMeansDenoising(noisy, None, h=20)
```

- `h` should roughly match the **standard deviation of the noise**. If noise has sigma=15, try h=15.
- Too low: noise remains visible.
- Too high: the image looks "plastic" or "painted" -- fine textures disappear.

## Template and Search Window Sizes

The two window parameters control the trade-off between quality and speed:

```python
# templateWindowSize: patch used for matching
# Larger = more context for matching = better quality = slower
# Typical: 7 (default)

# searchWindowSize: area searched for similar patches
# Larger = more candidates found = better denoising = much slower
# Typical: 21 (default)
```

In practice, the defaults (7 and 21) work well for most images. Increasing `searchWindowSize` beyond 21 has diminishing returns and dramatically increases computation time.

## Comparison: NLM vs Gaussian Blur

```python
# Gaussian blur: fast but smears edges
blurred = cv2.GaussianBlur(noisy, (7, 7), 0)

# NLM: slower but preserves edges
denoised = cv2.fastNlMeansDenoising(noisy, None, h=10)
```

Gaussian blur treats every pixel the same way -- it's a local, fixed-weight operation. NLM is an adaptive, non-local operation that respects image structure. The difference is most visible around edges and fine textures.

## Performance Considerations

NLM denoising is significantly slower than Gaussian blur because it searches for similar patches across a large area:

```python
# Faster: smaller search window
fast = cv2.fastNlMeansDenoising(noisy, None, h=10, templateWindowSize=5, searchWindowSize=15)

# Slower but better: larger search window
better = cv2.fastNlMeansDenoising(noisy, None, h=10, templateWindowSize=7, searchWindowSize=21)
```

For real-time applications, Gaussian blur may be the practical choice. For offline processing where quality matters, NLM is clearly superior.

## Tips & Common Mistakes

- `cv2.fastNlMeansDenoising()` is for **grayscale** images only. For color images, always use `cv2.fastNlMeansDenoisingColored()`. Passing a color image to the grayscale function produces garbled results.
- Both window sizes must be **odd numbers**. Even sizes cause unexpected behavior or errors.
- The `h` parameter is not a radius or kernel size -- it's a noise-level estimate. Think of it as "how much noise to remove."
- NLM is **slow**. On a 1920x1080 image, it can take several seconds. For video, consider processing at reduced resolution.
- Setting `h` too high makes skin look like plastic and fabrics look painted. Start with h=10 and adjust.
- The second parameter (`None`) is the output destination. You can pass a pre-allocated array for efficiency, but `None` works fine.
- For video denoising, OpenCV also offers `cv2.fastNlMeansDenoisingMulti()` which uses temporal information from adjacent frames.

## Starter Code

```python
import cv2
import numpy as np

# Create a detailed test image with edges and textures
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Gradient background
for y in range(300):
    for x in range(400):
        img[y, x] = (
            int(180 + 40 * np.sin(x / 30)),
            int(160 + 30 * np.cos(y / 25)),
            int(140 + 20 * np.sin((x + y) / 40))
        )

# Sharp-edged shapes
cv2.rectangle(img, (20, 20), (120, 120), (255, 80, 0), -1)
cv2.circle(img, (250, 80), 50, (0, 200, 100), -1)
cv2.line(img, (300, 20), (380, 130), (0, 0, 255), 3)
cv2.putText(img, 'Detail', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Checkerboard texture region
for y in range(220, 290):
    for x in range(20, 180):
        if ((x // 8) + (y // 8)) % 2 == 0:
            img[y, x] = (220, 220, 220)
        else:
            img[y, x] = (40, 40, 40)

# --- Add Gaussian noise ---
noise = np.random.normal(0, 25, img.shape).astype(np.int16)
noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Denoise with different h values ---
denoised_h5 = cv2.fastNlMeansDenoisingColored(noisy, None, h=5, hForColorComponents=5)
denoised_h10 = cv2.fastNlMeansDenoisingColored(noisy, None, h=10, hForColorComponents=10)
denoised_h20 = cv2.fastNlMeansDenoisingColored(noisy, None, h=20, hForColorComponents=20)

# --- Compare with Gaussian blur ---
gaussian = cv2.GaussianBlur(noisy, (7, 7), 0)

# --- Grayscale denoising comparison ---
noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
denoised_gray = cv2.fastNlMeansDenoising(noisy_gray, None, h=10)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Original', (5, 295), font, 0.5, (255, 255, 255), 1)
cv2.putText(noisy, 'Noisy (s=25)', (5, 295), font, 0.5, (255, 255, 255), 1)
cv2.putText(denoised_h5, 'NLM h=5', (5, 295), font, 0.5, (255, 255, 255), 1)
cv2.putText(denoised_h10, 'NLM h=10', (5, 295), font, 0.5, (255, 255, 255), 1)
cv2.putText(denoised_h20, 'NLM h=20', (5, 295), font, 0.5, (255, 255, 255), 1)
cv2.putText(gaussian, 'Gaussian 7x7', (5, 295), font, 0.5, (255, 255, 255), 1)

# Build grid: 3 rows x 2 columns
row1 = np.hstack([img, noisy])
row2 = np.hstack([denoised_h5, denoised_h10])
row3 = np.hstack([denoised_h20, gaussian])
result = np.vstack([row1, row2, row3])

# Compute PSNR for quality comparison
psnr_noisy = cv2.PSNR(img, noisy)
psnr_h10 = cv2.PSNR(img, denoised_h10)
psnr_gauss = cv2.PSNR(img, gaussian)

print(f'Image shape: {img.shape}')
print(f'Noise standard deviation: 25')
print(f'PSNR noisy vs original: {psnr_noisy:.1f} dB')
print(f'PSNR NLM h=10 vs original: {psnr_h10:.1f} dB')
print(f'PSNR Gaussian vs original: {psnr_gauss:.1f} dB')

cv2.imshow('Image Denoising', result)
```
