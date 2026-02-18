---
slug: 01-image-loading
title: Image Loading & Display
level: beginner
concepts: [cv2.imread, numpy arrays, image shape, cv2.imshow]
prerequisites: [00-opencv-basics]
---

## What Problem Are We Solving?

Before you can do anything with OpenCV, you need to understand what an image actually **is** in memory.

When you call `cv2.imread()`, OpenCV reads the file and returns a **NumPy array** — a grid of numbers. Each number represents a pixel's color intensity. This is the foundation of everything in computer vision.

## How Images Are Stored in Memory

A color image is a **3D NumPy array** with shape `(height, width, channels)`.

For a 300×400 color image:
- `img.shape` → `(300, 400, 3)`
- The **3 channels** are **Blue, Green, Red** — in that order (BGR, not RGB!)

```
img[y, x] = [Blue, Green, Red]   ← one pixel
```

A **grayscale** image has shape `(height, width)` — no channel dimension.

## Why BGR and Not RGB?

Historical reason: early camera hardware stored bytes in BGR order. OpenCV kept this convention. Always remember:
- Channel 0 = **Blue**
- Channel 1 = **Green**
- Channel 2 = **Red**

This trips up almost every beginner when they first try to display an OpenCV image in another library (like matplotlib) and the colors look wrong.

## Tips & Common Mistakes

- `cv2.imread()` returns `None` if the file path is wrong — always check before using.
- `img.shape` returns `(height, width, channels)` — **height comes before width**.
- `np.zeros((h, w, 3), np.uint8)` creates a black image from scratch.
- `uint8` means pixel values are integers from **0 to 255**.
- To create a white image: `np.ones((h, w, 3), np.uint8) * 255`

## Starter Code

```python
import cv2
import numpy as np

# Create a simple colored image from scratch (no file needed)
# np.zeros creates a black canvas: shape is (height, width, channels)
img = np.zeros((300, 400, 3), dtype=np.uint8)

# Fill the entire image with a color (BGR order!)
# Blue=0, Green=128, Red=255 → orange
img[:] = (0, 128, 255)

# Print image info
print(f'Shape: {img.shape}')   # (300, 400, 3)
print(f'Dtype: {img.dtype}')   # uint8 (values 0-255)
print(f'Size:  {img.size}')    # total number of values = 300*400*3

# Display the image
cv2.imshow('My First Image', img)
```
