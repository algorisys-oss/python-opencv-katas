---
slug: 33-morphology-erosion
title: "Morphology: Erosion"
level: intermediate
concepts: [cv2.erode, structuring elements, cv2.getStructuringElement]
prerequisites: [26-simple-thresholding]
---

## What Problem Are We Solving?

After thresholding, binary images often have **small white noise blobs**, **ragged edges**, or **connected regions that should be separate**. **Erosion** is a morphological operation that **shrinks white regions** — it slides a small shape (the structuring element) across the image and sets a pixel to white only if **all** pixels under the shape are white. This strips away boundary pixels, removing small noise and separating close objects.

## How Erosion Works

For each pixel in the image, the structuring element (kernel) is placed on top. The output pixel is white **only if every pixel under the kernel is white**. If even one pixel under the kernel is black, the output pixel becomes black.

In effect:
- White regions **shrink** by the size of the kernel
- Small white blobs **disappear**
- Thin white connections **break apart**

## Using cv2.erode()

```python
eroded = cv2.erode(src, kernel, iterations=1)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image (binary or grayscale) |
| `kernel` | Structuring element (NumPy array). Pass `None` for a default 3x3 rectangle |
| `iterations` | How many times to apply the erosion (default: 1) |

## Structuring Elements

The shape of the kernel determines **how** the erosion affects the image:

```python
# Rectangle — erodes uniformly in all directions
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Ellipse — rounded erosion, good for circular objects
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cross — erodes only horizontally and vertically
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

The size `(5, 5)` means a 5x5 pixel kernel. Larger kernels erode more aggressively.

## Iteration Count

Instead of using a larger kernel, you can apply a small kernel multiple times:

```python
# These produce similar (not identical) results:
eroded_large = cv2.erode(img, np.ones((7, 7), np.uint8), iterations=1)
eroded_multi = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=3)
```

Multiple iterations with a small kernel give a **more rounded** erosion than a single pass with a large rectangular kernel.

## Tips & Common Mistakes

- Erosion shrinks **white** (foreground) regions. If your foreground is black, you need dilation instead (or invert first).
- Passing `kernel=None` defaults to a 3x3 rectangle — fine for quick tests but always be explicit in production code.
- Too many iterations can make objects disappear entirely. Start with 1 or 2 and increase carefully.
- Erosion on a **grayscale** image replaces each pixel with the **minimum** value under the kernel — it darkens the image.
- The structuring element shape matters. Use `MORPH_ELLIPSE` for natural shapes and `MORPH_RECT` for text or rectangular features.
- Erosion is the first half of **morphological opening** (erosion + dilation), which removes noise while preserving object size.

## Starter Code

```python
import cv2
import numpy as np

# Create a binary image with various features
img = np.zeros((400, 600), dtype=np.uint8)

# Large solid rectangle
cv2.rectangle(img, (30, 30), (180, 180), 255, -1)

# Circle
cv2.circle(img, (300, 110), 80, 255, -1)

# Thin lines (erosion will break or remove them)
cv2.line(img, (420, 30), (580, 30), 255, 2)
cv2.line(img, (420, 60), (580, 60), 255, 1)

# Small noise dots
for _ in range(80):
    x, y = np.random.randint(30, 570), np.random.randint(220, 390)
    r = np.random.randint(1, 4)
    cv2.circle(img, (x, y), r, 255, -1)

# Larger blobs among the noise
cv2.rectangle(img, (50, 260), (150, 360), 255, -1)
cv2.circle(img, (300, 310), 50, 255, -1)
cv2.rectangle(img, (430, 260), (560, 370), 255, -1)

# --- Structuring elements ---
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# --- Apply erosion with different settings ---
eroded_rect = cv2.erode(img, rect_kernel, iterations=1)
eroded_ellipse = cv2.erode(img, ellipse_kernel, iterations=1)
eroded_cross = cv2.erode(img, cross_kernel, iterations=1)
eroded_multi = cv2.erode(img, rect_kernel, iterations=3)

# --- Build comparison display ---
def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
panels = [
    (to_bgr(img), 'Original'),
    (to_bgr(eroded_rect), 'Rect 5x5 (1 iter)'),
    (to_bgr(eroded_ellipse), 'Ellipse 5x5 (1 iter)'),
    (to_bgr(eroded_cross), 'Cross 5x5 (1 iter)'),
    (to_bgr(eroded_multi), 'Rect 5x5 (3 iter)'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Arrange: original + 4 erosion results
top_row = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
# Pad the bottom row to match width
blank = np.zeros_like(panels[0][0])
cv2.putText(blank, '(blank)', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
bottom_row = np.hstack([panels[3][0], panels[4][0], blank])
result = np.vstack([top_row, bottom_row])

white_orig = np.count_nonzero(img)
white_eroded = np.count_nonzero(eroded_rect)
print(f'White pixels - Original: {white_orig}, After erosion: {white_eroded}')
print(f'Reduction: {white_orig - white_eroded} pixels ({(white_orig - white_eroded) / white_orig * 100:.1f}%)')
print(f'Rect kernel:\n{rect_kernel}')

cv2.imshow('Morphology: Erosion', result)
```
