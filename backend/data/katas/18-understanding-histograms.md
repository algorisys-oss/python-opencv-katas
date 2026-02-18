---
slug: 18-understanding-histograms
title: Understanding Histograms
level: beginner
concepts: [cv2.calcHist, pixel intensity distribution, histogram visualization]
prerequisites: [17-splitting-merging-channels]
---

## What Problem Are We Solving?

Looking at an image, can you tell if it's too dark? Too bright? Has good contrast? Your eyes can guess, but a **histogram** gives you the precise, numerical answer.

A histogram is a **bar chart** that shows how many pixels in an image have each intensity value (0-255). It's one of the most fundamental tools for understanding image quality — and it's essential for techniques like thresholding, equalization, and exposure correction.

## What Does a Histogram Show?

For a grayscale image, the histogram has:
- **X-axis**: Pixel intensity values (0 = black, 255 = white)
- **Y-axis**: How many pixels have that intensity

```
Dark image     → most bars are on the LEFT (low values)
Bright image   → most bars are on the RIGHT (high values)
Low contrast   → bars clustered in a NARROW range
High contrast  → bars SPREAD across the full range
```

## Computing a Histogram with cv2.calcHist

```python
hist = cv2.calcHist([image], [channel], mask, [histSize], [range])
```

| Parameter | Meaning |
|---|---|
| `[image]` | Input image **in a list** (yes, it must be a list) |
| `[channel]` | Which channel: `[0]` for Blue/Gray, `[1]` for Green, `[2]` for Red |
| `mask` | Optional mask — `None` to use the whole image |
| `[histSize]` | Number of bins — `[256]` for one bin per intensity value |
| `[range]` | Value range — `[0, 256]` for standard `uint8` images |

```python
# Grayscale histogram
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
```

The result `hist` is a NumPy array of shape `(256, 1)` — each entry is the **count** of pixels with that intensity.

## Histograms for Color Images

For a color image, you compute **three separate histograms** — one per channel:

```python
hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])  # Blue
hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])  # Green
hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])  # Red
```

## Drawing a Histogram on an Image

Since we're using only OpenCV (no matplotlib), we draw the histogram manually using `cv2.line()`:

```python
# Create a blank image for the histogram
hist_img = np.zeros((300, 256, 3), dtype=np.uint8)

# Normalize histogram values to fit in the image height
cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)

# Draw each bar as a vertical line
for x in range(256):
    cv2.line(hist_img,
             (x, 300),                    # bottom point
             (x, 300 - int(hist[x])),     # top point
             (255, 255, 255),             # color (white)
             1)                           # thickness
```

`cv2.normalize()` scales the histogram values so the tallest bar fits within the image height. Without normalization, a single peak could be thousands of pixels tall.

## Analyzing Histograms

What you can learn from the shape:

```python
# Find the most common intensity value (mode)
peak_intensity = np.argmax(hist)

# Find the mean intensity
mean_val = np.mean(gray)

# Check if image is dark (mean < 85), medium (85-170), or bright (> 170)
if mean_val < 85:
    print("Image is dark")
elif mean_val > 170:
    print("Image is bright")
else:
    print("Image has medium brightness")
```

## Using Masks with Histograms

You can compute a histogram for only a **specific region** of the image by providing a mask:

```python
# Create a mask for the center region
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)

# Histogram of just the masked region
hist_region = cv2.calcHist([gray], [0], mask, [256], [0, 256])
```

## Histogram Bins

Using 256 bins gives you one bin per intensity value. You can reduce the number of bins for a smoother histogram:

```python
# 32 bins instead of 256 — each bin covers 8 intensity values
hist_32 = cv2.calcHist([gray], [0], None, [32], [0, 256])
```

Fewer bins = smoother, less detailed. More bins = precise, more noisy.

## Tips & Common Mistakes

- The image parameter must be in a **list**: `[img]`, not just `img`.
- Channel indices are **zero-based**: `[0]` = Blue, `[1]` = Green, `[2]` = Red.
- The range is `[0, 256]` — the upper bound is **exclusive**, so this covers values 0 through 255.
- Always **normalize** the histogram before drawing, or the bars will overflow the canvas.
- A histogram with all bars clustered together means **low contrast** — the image looks "washed out."
- A histogram with a spike at 0 or 255 indicates **clipping** — details are lost in shadows or highlights.
- `cv2.calcHist` is much faster than computing a histogram with NumPy loops.

## Starter Code

```python
import cv2
import numpy as np

# --- Create test images with different brightness/contrast ---
h, w = 200, 256

# Dark image: values concentrated at low end
dark = np.random.randint(10, 80, (h, w), dtype=np.uint8)

# Bright image: values concentrated at high end
bright = np.random.randint(180, 250, (h, w), dtype=np.uint8)

# Low contrast: values in a narrow middle range
low_contrast = np.random.randint(100, 155, (h, w), dtype=np.uint8)

# High contrast: values spread across the full range
high_contrast = np.random.randint(0, 256, (h, w), dtype=np.uint8)

# --- Function to draw a histogram onto an image ---
def draw_histogram(gray_img, hist_h=200, color=(255, 255, 255)):
    """Calculate and draw histogram for a grayscale image."""
    # Calculate histogram
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # Normalize to fit in the display height
    cv2.normalize(hist, hist, 0, hist_h - 10, cv2.NORM_MINMAX)

    # Create black canvas for the histogram
    hist_img = np.zeros((hist_h, 256, 3), dtype=np.uint8)

    # Draw bars
    for x in range(256):
        bar_height = int(hist[x])
        if bar_height > 0:
            cv2.line(hist_img, (x, hist_h), (x, hist_h - bar_height), color, 1)

    return hist_img

# --- Draw histograms for each test image ---
hist_dark = draw_histogram(dark, color=(100, 180, 255))
hist_bright = draw_histogram(bright, color=(100, 255, 180))
hist_low = draw_histogram(low_contrast, color=(180, 180, 255))
hist_high = draw_histogram(high_contrast, color=(255, 200, 100))

# Convert grayscale images to BGR for stacking
dark_bgr = cv2.cvtColor(dark, cv2.COLOR_GRAY2BGR)
bright_bgr = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
low_bgr = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)
high_bgr = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(dark_bgr, 'Dark', (5, 25), font, 0.7, (0, 200, 255), 1)
cv2.putText(bright_bgr, 'Bright', (5, 25), font, 0.7, (0, 200, 255), 1)
cv2.putText(low_bgr, 'Low Contrast', (5, 25), font, 0.7, (0, 200, 255), 1)
cv2.putText(high_bgr, 'High Contrast', (5, 25), font, 0.7, (0, 200, 255), 1)
cv2.putText(hist_dark, f'mean={np.mean(dark):.0f}', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(hist_bright, f'mean={np.mean(bright):.0f}', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(hist_low, f'mean={np.mean(low_contrast):.0f}', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(hist_high, f'mean={np.mean(high_contrast):.0f}', (5, 20), font, 0.5, (255, 255, 255), 1)

# Arrange: each column = image on top, histogram below
col1 = np.vstack([dark_bgr, hist_dark])
col2 = np.vstack([bright_bgr, hist_bright])
col3 = np.vstack([low_bgr, hist_low])
col4 = np.vstack([high_bgr, hist_high])
result = np.hstack([col1, col2, col3, col4])

print('Dark image: histogram bars clustered on the LEFT')
print('Bright image: histogram bars clustered on the RIGHT')
print('Low contrast: histogram bars in a NARROW range')
print('High contrast: histogram bars SPREAD across full range')

cv2.imshow('Understanding Histograms', result)
```
