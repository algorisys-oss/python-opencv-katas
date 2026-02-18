---
slug: 34-morphology-dilation
title: "Morphology: Dilation"
level: intermediate
concepts: [cv2.dilate, expanding regions, filling gaps]
prerequisites: [33-morphology-erosion]
---

## What Problem Are We Solving?

Sometimes binary images have **broken edges**, **small gaps** in contours, or **thin features that need to be thicker**. **Dilation** is the opposite of erosion — it **expands white regions** by setting a pixel to white if **any** pixel under the structuring element is white. This fills small holes, connects nearby regions, and thickens features.

## How Dilation Works

For each pixel, the structuring element is placed on top. The output pixel is white if **at least one** pixel under the kernel is white. In effect:

- White regions **grow** outward by the kernel size
- Small black holes inside white regions get **filled**
- Nearby white regions **merge** together
- Thin features become **thicker**

## Using cv2.dilate()

```python
dilated = cv2.dilate(src, kernel, iterations=1)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image (binary or grayscale) |
| `kernel` | Structuring element. Pass `None` for a default 3x3 rectangle |
| `iterations` | How many times to apply dilation (default: 1) |

The API mirrors `cv2.erode()` — same kernel types, same iteration parameter.

## Dilation with Different Kernels

```python
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

dilated_rect = cv2.dilate(binary, rect)
dilated_ellipse = cv2.dilate(binary, ellipse)
dilated_cross = cv2.dilate(binary, cross)
```

- **Rectangle** expands uniformly, giving blocky growth
- **Ellipse** gives rounded, natural-looking expansion
- **Cross** expands only along horizontal and vertical axes

## Combining Erosion and Dilation

A common pattern is to erode first (to remove noise), then dilate (to restore object size):

```python
# Remove noise, then restore size
cleaned = cv2.erode(binary, kernel, iterations=1)
cleaned = cv2.dilate(cleaned, kernel, iterations=1)
```

This two-step sequence is called **morphological opening** and is so common that OpenCV provides `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)` as a shortcut.

The reverse — dilate first, then erode — is called **closing** and fills small holes.

## Practical Uses of Dilation

- **Filling broken edges:** After Canny edge detection, dilate to connect gaps in contours
- **Thickening text:** Make thin text more readable for OCR
- **Connecting nearby objects:** Merge close blobs into a single region for counting
- **Creating masks:** Dilate a detected region to create a margin around it

## Tips & Common Mistakes

- Dilation expands **white** (foreground) regions. If your foreground is black, the objects will appear to shrink.
- Too many iterations creates **bloated blobs** that merge together — start small.
- On **grayscale** images, dilation replaces each pixel with the **maximum** value under the kernel — it brightens the image.
- Use the same kernel shape and size for erosion and dilation when you want to preserve the original object size (as in opening/closing).
- Dilation after erosion (opening) removes noise. Dilation before erosion (closing) fills holes. Don't mix up the order.
- For connecting broken contour lines, a small `(3, 3)` kernel with 1-2 iterations usually works well.

## Starter Code

```python
import cv2
import numpy as np

# Create a binary image demonstrating dilation use cases
img = np.zeros((400, 600), dtype=np.uint8)

# --- Broken edges (dilation will connect them) ---
cv2.rectangle(img, (30, 30), (170, 170), 255, 2)
# Erase parts of the rectangle to simulate broken edges
img[30:50, 80:120] = 0   # Break top edge
img[100:140, 168:172] = 0  # Break right edge

# --- Thin text ---
cv2.putText(img, 'THIN', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 1)

# --- Scattered dots (dilation will merge nearby ones) ---
dot_positions = [(420, 60), (440, 65), (460, 55), (480, 70),
                 (430, 100), (450, 95), (470, 105), (490, 95)]
for (x, y) in dot_positions:
    cv2.circle(img, (x, y), 3, 255, -1)

# --- Objects with small holes ---
cv2.circle(img, (100, 310), 60, 255, -1)
# Punch small holes
for _ in range(15):
    hx = np.random.randint(60, 140)
    hy = np.random.randint(270, 350)
    cv2.circle(img, (hx, hy), 3, 0, -1)

# --- Thin lines ---
cv2.line(img, (220, 220), (560, 220), 255, 1)
cv2.line(img, (220, 250), (560, 250), 255, 1)

# --- Small separated features ---
for i in range(8):
    x = 250 + i * 40
    cv2.circle(img, (x, 320), 5, 255, -1)

# --- Apply dilation with different settings ---
kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

dilated_1 = cv2.dilate(img, kernel_3, iterations=1)
dilated_2 = cv2.dilate(img, kernel_3, iterations=2)
dilated_ellipse = cv2.dilate(img, kernel_5, iterations=1)

# --- Erosion then dilation (opening preview) ---
eroded = cv2.erode(img, kernel_3, iterations=1)
erode_then_dilate = cv2.dilate(eroded, kernel_3, iterations=1)

# --- Build comparison display ---
def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
panels = [
    (to_bgr(img), 'Original'),
    (to_bgr(dilated_1), 'Rect 3x3 (1 iter)'),
    (to_bgr(dilated_2), 'Rect 3x3 (2 iter)'),
    (to_bgr(dilated_ellipse), 'Ellipse 5x5 (1 iter)'),
    (to_bgr(erode_then_dilate), 'Erode then Dilate'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

blank = np.zeros_like(panels[0][0])
cv2.putText(blank, '(blank)', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
top_row = np.hstack([panels[0][0], panels[1][0], panels[2][0]])
bottom_row = np.hstack([panels[3][0], panels[4][0], blank])
result = np.vstack([top_row, bottom_row])

white_orig = np.count_nonzero(img)
white_dilated = np.count_nonzero(dilated_1)
print(f'White pixels - Original: {white_orig}, After dilation: {white_dilated}')
print(f'Growth: {white_dilated - white_orig} pixels ({(white_dilated - white_orig) / max(white_orig, 1) * 100:.1f}%)')

cv2.imshow('Morphology: Dilation', result)
```
