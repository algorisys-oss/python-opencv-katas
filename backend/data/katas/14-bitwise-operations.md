---
slug: 14-bitwise-operations
title: Bitwise Operations
level: beginner
concepts: [cv2.bitwise_and, cv2.bitwise_or, cv2.bitwise_xor, cv2.bitwise_not]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

When you want to combine parts of two images — like placing a logo on a background, or extracting only certain regions — simple addition or blending won't work cleanly. You need **pixel-level logic** to decide which pixels to keep and which to discard. That's exactly what **bitwise operations** do.

Bitwise operations work on each bit of every pixel value. In practice, they let you **combine, mask, and manipulate image regions** with precision.

## The Four Bitwise Operations

OpenCV provides four bitwise functions. Each operates **pixel by pixel** on the binary representation of pixel values:

### AND — `cv2.bitwise_and`

A pixel in the result is "on" (white) only if the corresponding pixel in **both** inputs is "on".

```python
result = cv2.bitwise_and(img1, img2)
```

Think of it as **intersection** — keeps only the area where both images overlap.

### OR — `cv2.bitwise_or`

A pixel in the result is "on" if **either** (or both) inputs have that pixel "on".

```python
result = cv2.bitwise_or(img1, img2)
```

Think of it as **union** — keeps everything from both images.

### XOR — `cv2.bitwise_xor`

A pixel is "on" if the inputs **differ** at that position. If both are the same, it's off.

```python
result = cv2.bitwise_xor(img1, img2)
```

Think of it as **difference** — highlights where the images don't match.

### NOT — `cv2.bitwise_not`

Inverts every pixel. White becomes black, black becomes white. Each bit is flipped.

```python
result = cv2.bitwise_not(img)
```

For a `uint8` pixel, NOT transforms value `v` to `255 - v`.

## How Bits Actually Work

Each pixel value (0-255) is an 8-bit number. Bitwise operations work on each bit independently:

```
Pixel A:  200  = 11001000
Pixel B:  150  = 10010110

AND:       136  = 10000000  (both bits must be 1)
OR:        214  = 11011110  (either bit can be 1)
XOR:        78  = 01011110  (bits must differ)
NOT A:      55  = 00110111  (flip all bits)
```

For images, this happens to **every pixel in every channel** simultaneously.

## Using Masks with Bitwise Operations

Every bitwise function accepts an optional `mask` parameter:

```python
result = cv2.bitwise_and(img1, img2, mask=my_mask)
```

The mask must be a **single-channel (grayscale) image** of the same height and width. Only pixels where the mask is **non-zero** are processed — everywhere else becomes black (0).

This is incredibly useful for extracting specific regions of an image.

## Practical Use Case: Logo Overlay

The most common real-world use of bitwise ops is placing a logo on an image:

1. Create a mask from the logo (threshold to get white-logo-on-black-background).
2. Use `bitwise_not` to invert the mask.
3. Use `bitwise_and` with the inverted mask on the background to "punch a hole" where the logo goes.
4. Use `bitwise_and` with the original mask on the logo to extract just the logo pixels.
5. Use `bitwise_or` (or `cv2.add`) to combine the two results.

## Tips & Common Mistakes

- Bitwise operations require both images to have the **same size and type**.
- `cv2.bitwise_and` with a white image leaves the original unchanged. With a black image, everything becomes black.
- `cv2.bitwise_not` is useful for inverting masks (swap foreground and background).
- The `mask` parameter must be **single-channel `uint8`** — not a 3-channel color image.
- For binary images (black and white only), AND acts as intersection, OR acts as union.
- XOR is useful for detecting differences between two similar images.

## Starter Code

```python
import cv2
import numpy as np

# Create two images with white shapes on black background
h, w = 300, 300
img1 = np.zeros((h, w, 3), dtype=np.uint8)
img2 = np.zeros((h, w, 3), dtype=np.uint8)

# Image 1: white rectangle
cv2.rectangle(img1, (50, 50), (200, 200), (255, 255, 255), -1)

# Image 2: white circle
cv2.circle(img2, (175, 150), 100, (255, 255, 255), -1)

# --- Apply bitwise operations ---
bw_and = cv2.bitwise_and(img1, img2)   # Intersection
bw_or  = cv2.bitwise_or(img1, img2)    # Union
bw_xor = cv2.bitwise_xor(img1, img2)   # Difference
bw_not = cv2.bitwise_not(img1)         # Invert

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1, 'Rectangle', (10, 25), font, 0.6, (0, 255, 0), 1)
cv2.putText(img2, 'Circle', (10, 25), font, 0.6, (0, 255, 0), 1)
cv2.putText(bw_and, 'AND', (10, 25), font, 0.6, (0, 255, 0), 1)
cv2.putText(bw_or, 'OR', (10, 25), font, 0.6, (0, 255, 0), 1)
cv2.putText(bw_xor, 'XOR', (10, 25), font, 0.6, (0, 255, 0), 1)
cv2.putText(bw_not, 'NOT rect', (10, 25), font, 0.6, (0, 255, 0), 1)

# --- Bonus: using a mask ---
# Create a colorful image
colorful = np.zeros((h, w, 3), dtype=np.uint8)
colorful[:] = (0, 180, 255)  # Orange fill
cv2.rectangle(colorful, (20, 20), (280, 280), (255, 100, 50), -1)

# Create a circular mask (single channel)
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (150, 150), 100, 255, -1)

# Apply mask: only the circular region survives
masked_result = cv2.bitwise_and(colorful, colorful, mask=mask)
cv2.putText(masked_result, 'Masked', (10, 25), font, 0.6, (0, 255, 0), 1)

# Arrange results in a grid: 2 rows x 4 columns
# Pad mask to 3 channels for display
mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.putText(mask_display, 'Mask', (10, 25), font, 0.6, (0, 255, 0), 1)

top_row = np.hstack([img1, img2, bw_and, bw_or])
bot_row = np.hstack([bw_xor, bw_not, mask_display, masked_result])
result = np.vstack([top_row, bot_row])

print('Top row: Rectangle | Circle | AND | OR')
print('Bottom:  XOR | NOT | Mask | Masked result')

cv2.imshow('Bitwise Operations', result)
```
