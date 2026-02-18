---
slug: 08-image-cropping
title: Image Cropping
level: beginner
concepts: [NumPy slicing, aspect ratio, ROI extraction]
prerequisites: [03-pixel-access]
---

## What Problem Are We Solving?

You often need to extract a portion of an image — maybe you detected a face and want just the face region, or you need to remove borders, or you want to create a square thumbnail from a rectangular photo. OpenCV does not have a dedicated "crop" function. Instead, you use **NumPy array slicing** to extract a rectangular sub-region, also called a **Region of Interest (ROI)**.

## Basic Cropping with NumPy Slicing

Since an OpenCV image is a NumPy array with shape `(height, width, channels)`, cropping is just slicing along the row and column axes:

```python
cropped = img[y_start:y_end, x_start:x_end]
```

| Slice | Meaning |
|---|---|
| `y_start:y_end` | Row range (top to bottom) |
| `x_start:x_end` | Column range (left to right) |

> **Remember:** NumPy indexing is `[row, col]` which is `[y, x]` — the opposite of OpenCV drawing functions which use `(x, y)`.

```python
# Crop a 100x150 region starting at (x=50, y=30)
roi = img[30:130, 50:200]  # rows 30-129, columns 50-199
print(roi.shape)  # (100, 150, 3)
```

The slicing syntax is `start:stop` where `start` is **inclusive** and `stop` is **exclusive** (just like standard Python slicing).

## Center Cropping

A center crop extracts a region from the middle of the image. This is commonly used to create square thumbnails or to remove border content:

```python
h, w = img.shape[:2]
crop_w, crop_h = 200, 200

# Calculate top-left corner of the centered crop
x_start = (w - crop_w) // 2
y_start = (h - crop_h) // 2

center_crop = img[y_start:y_start + crop_h, x_start:x_start + crop_w]
```

For a **square center crop** that takes the largest possible square:

```python
h, w = img.shape[:2]
side = min(h, w)
x_start = (w - side) // 2
y_start = (h - side) // 2
square_crop = img[y_start:y_start + side, x_start:x_start + side]
```

## Aspect-Ratio-Aware Cropping

Sometimes you need a crop with a specific aspect ratio (e.g., 16:9 or 4:3). The strategy is: compute the largest rectangle of the desired ratio that fits inside the image, then center it:

```python
def aspect_crop(img, target_w, target_h):
    """Crop the largest centered region with the given aspect ratio."""
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        # Image is wider than target ratio — crop width
        new_w = int(h * target_ratio)
        x_start = (w - new_w) // 2
        return img[:, x_start:x_start + new_w]
    else:
        # Image is taller than target ratio — crop height
        new_h = int(w / target_ratio)
        y_start = (h - new_h) // 2
        return img[y_start:y_start + new_h, :]
```

```python
# Get a 16:9 crop from any image
widescreen = aspect_crop(img, 16, 9)
```

## Crop and Resize

A very common pipeline: crop first, then resize to a standard size. This is how you create uniform thumbnails from images of varying dimensions:

```python
# Crop the center square, then resize to 128x128
h, w = img.shape[:2]
side = min(h, w)
x = (w - side) // 2
y = (h - side) // 2
square = img[y:y + side, x:x + side]
thumbnail = cv2.resize(square, (128, 128), interpolation=cv2.INTER_AREA)
```

## Crops Are Views, Not Copies

An important NumPy detail: slicing creates a **view** of the original array, not a copy. Modifying the cropped region will also modify the original image:

```python
roi = img[50:150, 50:200]
roi[:] = (0, 0, 255)  # This ALSO changes img!
```

If you need an independent copy of the cropped region:

```python
roi = img[50:150, 50:200].copy()
roi[:] = (0, 0, 255)  # Only changes roi, not img
```

## Tips & Common Mistakes

- Cropping uses `[y_start:y_end, x_start:x_end]` — remember it is `[row, col]` = `[y, x]`, not `[x, y]`.
- Slice indices must be **non-negative integers** within the image bounds. Going out of bounds will not raise an error — NumPy will silently return a smaller array, which can cause hard-to-debug shape mismatches later.
- Crops are **views** by default. Use `.copy()` if you need an independent copy.
- When computing center crops, use integer division `//` to avoid float errors.
- An empty slice (e.g., `img[100:100, :]`) returns a zero-height array — check that your `y_end > y_start` and `x_end > x_start`.
- If you need to crop a non-rectangular region (e.g., a circular area), you will need masking techniques instead of simple slicing.

## Starter Code

```python
import cv2
import numpy as np

# Create a source image with distinct colored regions
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (40, 35, 30)

# Draw colored quadrants so crops are visually obvious
img[0:150, 0:200] = (0, 100, 180)     # Top-left: orange-brown
img[0:150, 200:400] = (180, 80, 0)    # Top-right: blue
img[150:300, 0:200] = (0, 160, 0)     # Bottom-left: green
img[150:300, 200:400] = (0, 0, 180)   # Bottom-right: red

# Draw grid lines at the midpoints
cv2.line(img, (200, 0), (200, 300), (255, 255, 255), 1)
cv2.line(img, (0, 150), (400, 150), (255, 255, 255), 1)

# Label the original
cv2.putText(img, '400x300', (150, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA)

# --- Basic crop: top-left quadrant ---
top_left = img[0:150, 0:200].copy()

# --- Center crop: 150x150 square from the middle ---
h, w = img.shape[:2]
side = 150
cx = (w - side) // 2
cy = (h - side) // 2
center = img[cy:cy + side, cx:cx + side].copy()

# --- Aspect-ratio crop: 16:9 from the full image ---
target_ratio = 16 / 9
img_ratio = w / h
if img_ratio > target_ratio:
    new_w = int(h * target_ratio)
    xs = (w - new_w) // 2
    widescreen = img[:, xs:xs + new_w].copy()
else:
    new_h = int(w / target_ratio)
    ys = (h - new_h) // 2
    widescreen = img[ys:ys + new_h, :].copy()

# --- Crop and resize to thumbnail ---
sq_side = min(h, w)
sx = (w - sq_side) // 2
sy = (h - sq_side) // 2
thumbnail = cv2.resize(img[sy:sy + sq_side, sx:sx + sq_side],
                       (100, 100), interpolation=cv2.INTER_AREA)

# --- Build comparison display ---
# Resize all crops to the same height for display
display_h = 150
def fit_height(image, th):
    ih, iw = image.shape[:2]
    s = th / ih
    return cv2.resize(image, (int(iw * s), th), interpolation=cv2.INTER_LINEAR)

col1 = fit_height(img, display_h)
col2 = fit_height(top_left, display_h)
col3 = fit_height(center, display_h)
col4 = fit_height(widescreen, display_h)
col5 = fit_height(thumbnail, display_h)

for label, col in [('Original', col1), ('Top-Left', col2), ('Center', col3),
                   ('16:9', col4), ('Thumbnail', col5)]:
    cv2.putText(col, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 255), 1, cv2.LINE_AA)

row = np.hstack([col1, col2, col3, col4, col5])

print(f'Original shape:    {img.shape}')
print(f'Top-left crop:     {top_left.shape}')
print(f'Center crop:       {center.shape}')
print(f'16:9 crop:         {widescreen.shape}')
print(f'Thumbnail:         {thumbnail.shape}')

cv2.imshow('Image Cropping', row)
```
