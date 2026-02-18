---
slug: 19-histogram-equalization
title: Histogram Equalization
level: beginner
concepts: [cv2.equalizeHist, CLAHE, contrast improvement]
prerequisites: [18-understanding-histograms]
---

## What Problem Are We Solving?

Some images look "flat" or "washed out" because their pixel intensities are concentrated in a narrow range — the histogram is squeezed into a small area. **Histogram equalization** stretches and redistributes pixel values so they spread across the entire 0-255 range, dramatically improving **contrast**.

This is essential for:
- Enhancing images taken in poor lighting conditions.
- Preprocessing images before feature detection (edges, corners).
- Medical imaging where subtle differences in intensity matter.

## What Does Equalization Do?

Equalization takes a histogram that's bunched up and **spreads it out** across the full intensity range:

```
Before:  [   ||||||||   ]    ← pixels clustered in the middle
After:   [|| ||| ||| || ]    ← pixels spread across full range
```

The algorithm uses the **cumulative distribution function (CDF)** to remap pixel values. Intensities that many pixels share get spread apart; rare intensities get compressed. The goal is a roughly **uniform distribution** — equal representation of all intensity levels.

## cv2.equalizeHist — Global Equalization

```python
equalized = cv2.equalizeHist(grayscale_image)
```

This function:
1. Computes the histogram of the input image.
2. Calculates the cumulative distribution function (CDF).
3. Normalizes the CDF to map values to [0, 255].
4. Remaps every pixel using the normalized CDF.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

**Important**: `cv2.equalizeHist` only works on **single-channel (grayscale)** images. For color images, you need to convert to a color space with a luminance channel (like YCrCb or LAB), equalize only the luminance, and merge back.

## The Problem with Global Equalization

Global equalization applies the **same transformation** to the entire image. This can cause problems:
- In images with both bright and dark regions, equalizing globally may **over-brighten** already-bright areas.
- Fine details in locally dark or bright regions may be lost.
- The result can look unnatural with harsh transitions.

## CLAHE — Adaptive Equalization

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** solves this by dividing the image into small tiles and equalizing each tile independently:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
result = clahe.apply(grayscale_image)
```

| Parameter | Meaning |
|---|---|
| `clipLimit` | Controls how much contrast is enhanced. Higher = more contrast. Default: `40.0`, but `2.0-3.0` is typical. |
| `tileGridSize` | Number of tiles in each dimension. `(8, 8)` divides the image into 64 tiles. |

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_clahe = clahe.apply(gray)
```

CLAHE produces more **natural-looking** results because it adapts to local regions rather than treating the whole image the same way.

## How clipLimit Works

The "CL" in CLAHE stands for **Contrast Limited**. The `clipLimit` parameter caps how much any histogram bin can be boosted:

```python
# Gentle enhancement
clahe_gentle = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

# Standard enhancement
clahe_standard = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Aggressive enhancement
clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
```

- Low `clipLimit` (1.0): Subtle enhancement, preserves the original look.
- Medium `clipLimit` (2.0-3.0): Good balance of enhanced contrast and natural appearance.
- High `clipLimit` (5.0+): Strong enhancement, can introduce noise amplification.

## Equalizing Color Images

Since `equalizeHist` works on single-channel images only, to equalize a color image you work with the **luminance** channel:

```python
# Convert BGR to YCrCb (Y = luminance)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Split channels
y, cr, cb = cv2.split(ycrcb)

# Equalize only the Y (luminance) channel
y_eq = cv2.equalizeHist(y)

# Merge back and convert to BGR
ycrcb_eq = cv2.merge([y_eq, cr, cb])
result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
```

Never equalize each B, G, R channel separately — this will distort the colors.

## Comparing Results: Before and After

A good way to evaluate equalization is to compare the histograms:

```python
hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])
```

After equalization, the histogram should look more **spread out** and more **uniform** across the full range.

## Tips & Common Mistakes

- `cv2.equalizeHist` only accepts **single-channel** (grayscale) images — it will error on 3-channel BGR.
- For color images, equalize the **luminance channel** (Y in YCrCb, or L in LAB), not individual B/G/R channels.
- CLAHE generally produces better results than global equalization for most real-world images.
- Start with `clipLimit=2.0` and `tileGridSize=(8,8)` as defaults, then adjust.
- High `clipLimit` values amplify noise — be careful with noisy images.
- Equalization is not always an improvement — well-exposed images with good contrast may look worse after equalization.
- CLAHE is a two-step process: first create the CLAHE object, then call `.apply()`.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a low-contrast grayscale image ---
h, w = 250, 300

# Simulate a low-contrast scene: values clustered between 80 and 170
low_contrast = np.zeros((h, w), dtype=np.uint8)

# Create a gradient pattern with limited range
for y in range(h):
    for x in range(w):
        val = int(80 + 90 * (x / w) * (y / h))  # Range: ~80 to ~170
        low_contrast[y, x] = val

# Add some shapes to show detail recovery
cv2.circle(low_contrast, (150, 125), 60, 130, -1)
cv2.rectangle(low_contrast, (50, 50), (120, 100), 100, -1)
cv2.rectangle(low_contrast, (200, 160), (270, 210), 155, -1)
cv2.putText(low_contrast, 'OpenCV', (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 140, 2)

# --- Apply global equalization ---
equalized = cv2.equalizeHist(low_contrast)

# --- Apply CLAHE with different clipLimit values ---
clahe_gentle = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
clahe_standard = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

result_gentle = clahe_gentle.apply(low_contrast)
result_standard = clahe_standard.apply(low_contrast)
result_strong = clahe_strong.apply(low_contrast)

# --- Draw histograms for comparison ---
def draw_hist(gray_img, hist_h=100, color=(200, 200, 200)):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_h - 5, cv2.NORM_MINMAX)
    hist_canvas = np.zeros((hist_h, 256, 3), dtype=np.uint8)
    for x in range(256):
        bar_h = int(hist[x])
        if bar_h > 0:
            cv2.line(hist_canvas, (x, hist_h), (x, hist_h - bar_h), color, 1)
    return hist_canvas

hist_orig = draw_hist(low_contrast, color=(100, 180, 255))
hist_eq = draw_hist(equalized, color=(100, 255, 180))
hist_clahe = draw_hist(result_standard, color=(180, 180, 255))

# --- Resize histograms to match image width ---
hist_orig = cv2.resize(hist_orig, (w, 100))
hist_eq = cv2.resize(hist_eq, (w, 100))
hist_clahe = cv2.resize(hist_clahe, (w, 100))

# Convert grayscale images to BGR for display
orig_bgr = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)
eq_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
cl_gentle_bgr = cv2.cvtColor(result_gentle, cv2.COLOR_GRAY2BGR)
cl_std_bgr = cv2.cvtColor(result_standard, cv2.COLOR_GRAY2BGR)
cl_strong_bgr = cv2.cvtColor(result_strong, cv2.COLOR_GRAY2BGR)

# Label images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(orig_bgr, 'Original', (5, 20), font, 0.6, (0, 200, 255), 1)
cv2.putText(eq_bgr, 'equalizeHist', (5, 20), font, 0.6, (0, 255, 200), 1)
cv2.putText(cl_gentle_bgr, 'CLAHE clip=1', (5, 20), font, 0.6, (200, 200, 255), 1)
cv2.putText(cl_std_bgr, 'CLAHE clip=2', (5, 20), font, 0.6, (200, 200, 255), 1)
cv2.putText(cl_strong_bgr, 'CLAHE clip=4', (5, 20), font, 0.6, (200, 200, 255), 1)

# Arrange layout:
# Row 1: Original + histogram | equalizeHist + histogram
# Row 2: CLAHE clip=1 | CLAHE clip=2 | CLAHE clip=4
# Resize CLAHE images to fit 3 across = same total width as 2 across
total_w = w * 2

col1_top = np.vstack([orig_bgr, hist_orig])
col2_top = np.vstack([eq_bgr, hist_eq])
top_section = np.hstack([col1_top, col2_top])

# Resize CLAHE results to fit 3 in the same total width
clahe_w = total_w // 3
cl_gentle_rsz = cv2.resize(cl_gentle_bgr, (clahe_w, h))
cl_std_rsz = cv2.resize(cl_std_bgr, (clahe_w, h))
# Last one takes remaining width to avoid off-by-one
cl_strong_rsz = cv2.resize(cl_strong_bgr, (total_w - 2 * clahe_w, h))
bot_section = np.hstack([cl_gentle_rsz, cl_std_rsz, cl_strong_rsz])

result = np.vstack([top_section, bot_section])

print(f'Original range: [{low_contrast.min()}, {low_contrast.max()}]')
print(f'Equalized range: [{equalized.min()}, {equalized.max()}]')
print(f'CLAHE range: [{result_standard.min()}, {result_standard.max()}]')
print('Top: Original (with histogram) | Global equalization (with histogram)')
print('Bottom: CLAHE clipLimit=1 | clipLimit=2 | clipLimit=4')

cv2.imshow('Histogram Equalization', result)
```
