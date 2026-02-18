---
slug: 96-image-similarity
title: Image Comparison & Similarity
level: advanced
concepts: [SSIM, histogram comparison, MSE]
prerequisites: [18-understanding-histograms]
---

## What Problem Are We Solving?

How do you measure whether two images are "similar"? This question arises everywhere: detecting duplicate photos, measuring image quality after compression, verifying that a rendered output matches an expected result, or finding the best match in a database. There is no single perfect metric -- different methods capture different aspects of similarity. This kata covers three complementary approaches: **Mean Squared Error (MSE)**, a **manual SSIM implementation**, and **histogram comparison**.

## Mean Squared Error (MSE)

MSE is the simplest similarity metric. It computes the average squared difference between corresponding pixels:

```python
def mse(img1, img2):
    err = np.sum((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err
```

| MSE Value | Meaning |
|---|---|
| 0 | Identical images |
| Small (< 100) | Very similar |
| Large (> 1000) | Quite different |

MSE is fast and intuitive but has a major limitation: it doesn't account for the **structure** of the difference. A slight shift of the entire image can produce a high MSE even though the images look identical to a human.

## Structural Similarity (SSIM) -- Manual Implementation

SSIM compares images based on three components: **luminance** (brightness), **contrast** (variance), and **structure** (correlation pattern). It better matches human perception than MSE:

```python
def ssim(img1, img2, C1=6.5025, C2=58.5225):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()
```

The constants `C1` and `C2` stabilize the division when means or variances are near zero. They are derived from `(K*L)**2` where `K` is a small constant (0.01 and 0.03) and `L` is the dynamic range (255).

| SSIM Value | Meaning |
|---|---|
| 1.0 | Identical images |
| > 0.95 | Excellent similarity |
| 0.8 - 0.95 | Good similarity with minor differences |
| < 0.5 | Substantially different |

## Histogram Comparison with cv2.compareHist

Histograms capture the **distribution** of pixel values, ignoring spatial arrangement. Two images can have very different content but similar histograms (e.g., two different forest photos), or similar content but different histograms (e.g., same photo with brightness change):

```python
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
cv2.normalize(hist1, hist1)
cv2.normalize(hist2, hist2)
score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

| Method | Range | Best Match |
|---|---|---|
| `HISTCMP_CORREL` | -1 to 1 | 1 (high correlation) |
| `HISTCMP_CHISQR` | 0 to inf | 0 (low chi-square distance) |
| `HISTCMP_INTERSECT` | 0 to N | High (large intersection) |
| `HISTCMP_BHATTACHARYYA` | 0 to 1 | 0 (low distance) |

## Template Matching for Localized Similarity

`cv2.matchTemplate` slides a template across an image and computes a similarity score at each position:

```python
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
```

This is useful when you want to find **where** a specific pattern appears in a larger image.

## The Complete Pipeline

1. **Input**: Two images to compare (original and modified)
2. **MSE**: Compute pixel-level squared difference
3. **SSIM**: Compute structural similarity (luminance + contrast + structure)
4. **Histogram Comparison**: Compare pixel value distributions
5. **Difference Visualization**: Show where the images differ
6. **Output**: Numerical scores and visual comparison

## Tips & Common Mistakes

- Always convert to the same data type before computing MSE or SSIM. Comparing `uint8` and `float32` images gives meaningless results.
- MSE is sensitive to global brightness changes. Two identical images at different brightness levels will have high MSE but should be considered similar.
- SSIM is computed per-channel for color images. Average the per-channel scores for a single number, or convert to grayscale first.
- Normalize histograms before comparison. Otherwise, images of different sizes produce incomparable histograms.
- `cv2.compareHist` works on single-channel histograms. For color comparison, compute histograms for each channel separately or convert to a different color space.
- Template matching finds the **best** match location but does not tell you if the match is good. Always check the max correlation value.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create an original image and modified versions
# =============================================================
img_h, img_w = 300, 400

# Original image: colorful scene with shapes
original = np.zeros((img_h, img_w, 3), dtype=np.uint8)
original[:] = (180, 170, 160)

# Draw shapes
cv2.rectangle(original, (30, 30), (150, 150), (200, 50, 50), -1)
cv2.circle(original, (280, 100), 60, (50, 180, 50), -1)
cv2.rectangle(original, (200, 180), (370, 270), (50, 50, 200), -1)
cv2.putText(original, 'OpenCV', (50, 250), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 2, cv2.LINE_AA)

# Add gradient texture
for y in range(img_h):
    for x in range(0, img_w, 5):
        val = int(10 * np.sin(x * 0.05 + y * 0.03))
        original[y, x:x+5] = np.clip(original[y, x:x+5].astype(int) + val, 0, 255).astype(np.uint8)

# Modified version 1: slight blur (simulating compression)
blurred = cv2.GaussianBlur(original, (7, 7), 2)

# Modified version 2: brightness shift
brighter = cv2.convertScaleAbs(original, alpha=1.0, beta=40)

# Modified version 3: added noise
noisy = original.copy().astype(np.float64)
noise = np.random.normal(0, 20, original.shape)
noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)

# Modified version 4: completely different image
different = np.zeros((img_h, img_w, 3), dtype=np.uint8)
different[:] = (40, 60, 80)
cv2.circle(different, (200, 150), 100, (0, 200, 200), -1)
cv2.rectangle(different, (50, 200), (350, 280), (200, 200, 0), -1)

print('Created original and 4 modified versions')

# =============================================================
# Step 2: Define similarity metrics
# =============================================================

def compute_mse(img1, img2):
    """Mean Squared Error between two images."""
    err = np.sum((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    err /= float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return err

def compute_ssim(img1, img2):
    """Simplified SSIM on grayscale images."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2  # 6.5025
    C2 = (0.03 * 255) ** 2  # 58.5225

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean()

def compare_histograms(img1, img2):
    """Compare histograms using correlation."""
    scores = []
    for ch in range(3):
        h1 = cv2.calcHist([img1], [ch], None, [256], [0, 256])
        h2 = cv2.calcHist([img2], [ch], None, [256], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        scores.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    return np.mean(scores)

# =============================================================
# Step 3: Compare original against each modified version
# =============================================================
test_images = {
    'Blurred': blurred,
    'Brighter': brighter,
    'Noisy': noisy,
    'Different': different,
}

print(f'\n{"Comparison":<20} {"MSE":>10} {"SSIM":>10} {"Hist Corr":>10}')
print('-' * 52)

results = {}
for name, modified in test_images.items():
    mse_val = compute_mse(original, modified)
    ssim_val = compute_ssim(original, modified)
    hist_val = compare_histograms(original, modified)
    results[name] = (mse_val, ssim_val, hist_val)
    print(f'{name:<20} {mse_val:>10.1f} {ssim_val:>10.4f} {hist_val:>10.4f}')

# Self-comparison (should be perfect)
mse_self = compute_mse(original, original)
ssim_self = compute_ssim(original, original)
hist_self = compare_histograms(original, original)
print(f'{"Self (identity)":<20} {mse_self:>10.1f} {ssim_self:>10.4f} {hist_self:>10.4f}')

# =============================================================
# Step 4: Compute visual difference maps
# =============================================================
def diff_map(img1, img2):
    """Amplified absolute difference for visualization."""
    diff = cv2.absdiff(img1, img2)
    # Amplify for visibility
    return cv2.convertScaleAbs(diff, alpha=3.0)

# =============================================================
# Step 5: Build comparison visualization
# =============================================================
font = cv2.FONT_HERSHEY_SIMPLEX
small_h, small_w = img_h // 2, img_w // 2

panels = []

# Original
orig_small = cv2.resize(original, (small_w, small_h))
cv2.putText(orig_small, 'Original', (5, 18), font, 0.45, (0, 255, 0), 1)
panels.append(orig_small)

# Each comparison: modified image with metrics overlay
for name, modified in test_images.items():
    mse_val, ssim_val, hist_val = results[name]
    mod_small = cv2.resize(modified, (small_w, small_h))
    cv2.putText(mod_small, name, (5, 18), font, 0.45, (0, 255, 0), 1)
    cv2.putText(mod_small, f'MSE:{mse_val:.0f}', (5, 38), font, 0.35, (0, 200, 255), 1)
    cv2.putText(mod_small, f'SSIM:{ssim_val:.3f}', (5, 55), font, 0.35, (0, 200, 255), 1)
    cv2.putText(mod_small, f'Hist:{hist_val:.3f}', (5, 72), font, 0.35, (0, 200, 255), 1)
    panels.append(mod_small)

# Difference maps
diff_panels = []
placeholder = np.zeros((small_h, small_w, 3), dtype=np.uint8)
cv2.putText(placeholder, 'Diff Maps:', (5, 18), font, 0.45, (200, 200, 200), 1)
diff_panels.append(placeholder)

for name, modified in test_images.items():
    d = diff_map(original, modified)
    d_small = cv2.resize(d, (small_w, small_h))
    cv2.putText(d_small, f'{name} diff', (5, 18), font, 0.4, (0, 255, 0), 1)
    diff_panels.append(d_small)

# Layout: 2 rows x 5 columns
top_row = np.hstack(panels)
bottom_row = np.hstack(diff_panels)
result = np.vstack([top_row, bottom_row])

cv2.imshow('Image Comparison & Similarity', result)
```
