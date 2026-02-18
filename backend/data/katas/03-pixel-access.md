---
slug: 03-pixel-access
title: Pixel Access & Manipulation
level: beginner
concepts: [numpy indexing, pixel values, ROI, array slicing]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

An image is just a NumPy array. That means you can **read and write individual pixels** using standard Python array indexing — no special OpenCV functions needed.

## Accessing a Single Pixel

```python
pixel = img[y, x]   # Note: row (y) first, then column (x)!
```

This returns `[Blue, Green, Red]` for a color image.

> **Why y before x?** Arrays are indexed by `[row, column]`, and rows go top-to-bottom (the y direction). Remember: `img[row, col]` = `img[y, x]`.

## Region of Interest (ROI)

You can slice a rectangular region just like a 2D list:

```python
roi = img[y1:y2, x1:x2]
```

This is called an **ROI (Region of Interest)**. ROIs are **views** into the original array — modifying an ROI modifies the original image.

```python
# Copy to avoid modifying original
roi_copy = img[50:150, 50:200].copy()
```

## Accessing Individual Channels

```python
blue  = img[:, :, 0]   # Blue channel
green = img[:, :, 1]   # Green channel
red   = img[:, :, 2]   # Red channel
```

## This Is the Foundation Of

- Drawing operations
- Image cropping
- Masking
- Template matching

## Tips & Common Mistakes

- `img[y, x]` — always **row (y) first**, column (x) second.
- ROI slicing returns a **view**, not a copy. Use `.copy()` if you don't want to modify the original.
- `img[y, x, 0]` accesses only the **Blue** channel at pixel (x, y).
- Setting a region: `img[50:100, 50:100] = (0, 255, 0)` fills a rectangle with green.

## Starter Code

```python
import cv2
import numpy as np

# Create a white canvas
img = np.ones((300, 400, 3), dtype=np.uint8) * 255

# --- Single pixel access ---
# Read pixel at row=100, col=200 (y=100, x=200)
pixel = img[100, 200]
print(f'Pixel at (x=200, y=100): {pixel}')  # [255, 255, 255] = white

# Write a single pixel (make it red)
img[100, 200] = (0, 0, 255)  # BGR: Red

# --- ROI (Region of Interest) ---
# Draw a blue rectangle by setting a region
img[50:150, 50:200] = (255, 0, 0)   # Blue rectangle

# Draw a green rectangle
img[50:150, 210:360] = (0, 255, 0)  # Green rectangle

# --- Extract and copy an ROI ---
roi = img[50:150, 50:200].copy()    # Copy the blue region
img[160:260, 100:250] = roi         # Paste it lower

# --- Access individual channels ---
b_channel = img[:, :, 0]  # Blue channel only
print(f'Blue channel shape: {b_channel.shape}')  # (300, 400)

cv2.imshow('Pixel Access Demo', img)
```
