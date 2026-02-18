---
slug: 28-otsu-thresholding
title: "Otsu's Thresholding"
level: intermediate
concepts: [cv2.threshold with THRESH_OTSU, automatic threshold]
prerequisites: [26-simple-thresholding, 18-understanding-histograms]
---

## What Problem Are We Solving?

With simple thresholding, you have to **manually choose** the threshold value. Pick it wrong and you get poor segmentation. But how do you find the "best" threshold for a given image? **Otsu's method** answers this question automatically — it analyzes the image histogram and computes the optimal threshold that **minimizes within-class variance** (or equivalently, maximizes between-class variance). It works best when the histogram has two distinct peaks (a **bimodal distribution**).

## What Is a Bimodal Histogram?

A bimodal histogram has **two peaks** — one for the background and one for the foreground:

```
Frequency
|   *                    *
|  * *                  * *
| *   *                *   *
|*     *              *     *
|       *            *
|        ****    ****
|            ****
+----------------------------------
0         T=127              255
          ↑ optimal threshold
```

The valley between the peaks is where Otsu places the threshold. If your histogram is **not** bimodal (e.g., it has three peaks or is flat), Otsu's method may give poor results.

## How Otsu's Method Works

Otsu's algorithm tries **every possible threshold** (0 to 255) and for each one computes:

1. **Class 1 (background):** all pixels <= threshold
2. **Class 2 (foreground):** all pixels > threshold
3. **Within-class variance:** weighted sum of each class's variance

The threshold that **minimizes** the combined within-class variance is chosen. This is equivalent to finding the threshold that **maximizes** the separation between the two classes.

```
For threshold T:
    w1 = fraction of pixels in class 1 (background)
    w2 = fraction of pixels in class 2 (foreground)
    σ²_within = w1 * var(class1) + w2 * var(class2)

Best T = argmin(σ²_within)
```

The beauty of Otsu's method is that it's completely **automatic** — no manual parameter tuning needed.

## Using Otsu's Method in OpenCV

Otsu's method is activated by adding the `cv2.THRESH_OTSU` flag to `cv2.threshold()`:

```python
# The thresh parameter (0 here) is IGNORED — Otsu computes its own
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f'Otsu selected threshold: {ret}')
```

| Important details |
|---|
| The `thresh` parameter (second argument) is **ignored** when using `THRESH_OTSU` |
| The return value `ret` contains the **automatically computed threshold** |
| Input must be **8-bit single-channel** (grayscale) |
| `THRESH_OTSU` is combined with `THRESH_BINARY` or `THRESH_BINARY_INV` using `+` |

## Comparing Manual vs Otsu Threshold

```python
# Manual: you guess a threshold
_, manual_low = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
_, manual_mid = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, manual_high = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Otsu: automatically finds the best threshold
otsu_val, otsu_result = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f'Otsu chose: {otsu_val}')  # Might be 145, 92, etc. — depends on image
```

Otsu typically finds a threshold you might not have guessed, and it adapts to each image individually.

## Combining Otsu with Gaussian Blur

Otsu's method works by analyzing the histogram. Noise adds extra peaks and smears the histogram, making the bimodal valley harder to find. **Pre-blurring** with Gaussian smoothing cleans up the histogram:

```python
# Without pre-blur: noise can confuse Otsu
otsu_val1, raw_otsu = cv2.threshold(noisy_gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# With Gaussian pre-blur: cleaner histogram, better threshold
blurred = cv2.GaussianBlur(noisy_gray, (5, 5), 0)
otsu_val2, blur_otsu = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f'Otsu without blur: {otsu_val1}')
print(f'Otsu with blur: {otsu_val2}')  # Usually a better result
```

> **Best practice:** Always apply `cv2.GaussianBlur()` before Otsu's thresholding. This is one of the most effective preprocessing combinations in OpenCV.

## When Otsu Fails

Otsu's method assumes the histogram is **bimodal**. It struggles when:

```python
# 1. Histogram is unimodal (one dominant class, no clear foreground)
# → Otsu picks an arbitrary threshold, results are poor

# 2. Histogram has three or more peaks
# → Otsu finds a compromise that may not separate what you want

# 3. Foreground and background overlap heavily in intensity
# → No clear valley to find

# 4. Very unequal class sizes (tiny object on large background)
# → The small class barely affects the variance calculation
```

For these cases, consider adaptive thresholding or more advanced segmentation methods.

## Visualizing the Histogram and Threshold

Understanding why Otsu picks a particular threshold:

```python
# Compute histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

# Get Otsu threshold
otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# The threshold should fall in the valley between two peaks
# You can verify by looking at hist[int(otsu_val)]
```

## Tips & Common Mistakes

- Always pass `0` as the `thresh` parameter when using `THRESH_OTSU` — it's ignored anyway, but `0` makes your intent clear.
- `THRESH_OTSU` is a **flag**, not a threshold type. Combine it: `cv2.THRESH_BINARY + cv2.THRESH_OTSU`.
- The return value `ret` is crucial — it tells you what threshold Otsu computed. Always check it.
- Otsu requires a **bimodal histogram** to work well. If your image doesn't have two clear intensity groups, the result may be poor.
- **Always Gaussian blur first.** This is the single most important tip for Otsu's method.
- Otsu operates on the **global** histogram — it doesn't handle uneven lighting. For that, use adaptive thresholding instead.
- Otsu's method only works with 8-bit images. For floating-point images, convert first.
- You can combine Otsu with `THRESH_BINARY_INV` too: `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`.

## Starter Code

```python
import cv2
import numpy as np

# Create a bimodal image (two distinct intensity groups)
img = np.zeros((300, 400), dtype=np.uint8)

# Background: bright region (peak around 180)
img[:] = 180
noise_bg = np.random.randint(-20, 20, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + noise_bg, 0, 255).astype(np.uint8)

# Foreground: dark shapes (peak around 60)
foreground = np.zeros_like(img)
cv2.rectangle(foreground, (30, 30), (150, 140), 255, -1)
cv2.circle(foreground, (280, 85), 55, 255, -1)
cv2.ellipse(foreground, (200, 220), (80, 40), 0, 0, 360, 255, -1)
cv2.putText(foreground, 'OTSU', (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

fg_mask = foreground > 0
fg_noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
img[fg_mask] = np.clip(60 + fg_noise[fg_mask], 0, 255).astype(np.uint8)

# --- Manual thresholding at different values ---
_, manual_80 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
_, manual_127 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
_, manual_180 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# --- Otsu's automatic thresholding ---
otsu_val, otsu_result = cv2.threshold(img, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Otsu with Gaussian pre-blur ---
blurred = cv2.GaussianBlur(img, (5, 5), 0)
otsu_blur_val, otsu_blur_result = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Otsu with inverse ---
_, otsu_inv = cv2.threshold(img, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Draw histogram visualization
hist_img = np.zeros((100, 256, 3), dtype=np.uint8)
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
hist_normalized = (hist / hist.max() * 95).astype(int)
for x in range(256):
    cv2.line(hist_img, (x, 99), (x, 99 - hist_normalized[x]), (200, 200, 200), 1)
# Mark Otsu threshold on histogram
cv2.line(hist_img, (int(otsu_val), 0), (int(otsu_val), 99), (0, 0, 255), 2)
cv2.putText(hist_img, f'T={int(otsu_val)}', (int(otsu_val) + 3, 15),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
hist_display = cv2.resize(hist_img, (400, 300))

# Label helper
def label(image, text):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA)
    return out

# Row 1: Manual thresholds vs Otsu
row1 = np.hstack([
    label(img, 'Original'),
    label(manual_80, 'Manual T=80'),
    label(manual_127, 'Manual T=127'),
    label(otsu_result, f'Otsu T={int(otsu_val)}'),
])

# Row 2: Otsu variants + histogram
row2 = np.hstack([
    label(hist_display, 'Histogram + Otsu T'),
    label(otsu_blur_result, f'Blur+Otsu T={int(otsu_blur_val)}'),
    label(otsu_inv, 'Otsu Inverted'),
    label(manual_180, 'Manual T=180'),
])

result = np.vstack([row1, row2])

print(f'Otsu threshold (no blur): {otsu_val:.0f}')
print(f'Otsu threshold (with blur): {otsu_blur_val:.0f}')
print(f'Image mean: {img.mean():.1f}')
print(f'Foreground pixels (Otsu): {np.sum(otsu_result == 0)} dark, {np.sum(otsu_result == 255)} bright')

cv2.imshow("Otsu's Thresholding", result)
```
