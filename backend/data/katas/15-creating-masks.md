---
slug: 15-creating-masks
title: Creating Masks
level: beginner
concepts: [binary masks, cv2.threshold, combining masks, mask application]
prerequisites: [14-bitwise-operations]
---

## What Problem Are We Solving?

You often want to process **only a part** of an image — maybe you want to change the color of a specific region, blur just the background, or extract an object from a scene. A **mask** tells OpenCV which pixels to include and which to ignore.

A mask is simply a **binary image** — same height and width as your image, single channel, where white pixels (255) mean "include this pixel" and black pixels (0) mean "ignore this pixel."

## What Is a Mask?

A mask is a grayscale image with only two values:

```
0   (black) → pixel is excluded / hidden
255 (white) → pixel is included / visible
```

When you apply a mask to an image, only the white regions "pass through." Everything under the black regions becomes black (zeroed out).

```python
# Apply mask: keep only white regions
result = cv2.bitwise_and(image, image, mask=my_mask)
```

## Creating Masks from Shapes

The simplest way to create a mask is to draw white shapes on a black canvas:

```python
# Start with a black (all-zero) single-channel image
mask = np.zeros((height, width), dtype=np.uint8)

# Draw a white filled circle — this region will be "visible"
cv2.circle(mask, (150, 150), 80, 255, -1)

# Draw a white filled rectangle
cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)
```

Key points:
- The mask is **single channel** — use `np.zeros((h, w), dtype=np.uint8)`, not `(h, w, 3)`.
- Draw with color `255` (white) and thickness `-1` (filled).

## Creating Masks with Thresholding

You can create a mask from an existing image by **thresholding** — converting pixel values to either 0 or 255 based on a cutoff:

```python
# Convert to grayscale first
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold: pixels above 127 become 255 (white), below become 0 (black)
ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

| Parameter | Meaning |
|---|---|
| `gray` | Input grayscale image |
| `127` | Threshold value — the cutoff |
| `255` | Value to assign to pixels above the threshold |
| `cv2.THRESH_BINARY` | Thresholding method |

`ret` is the threshold value used (useful with automatic thresholding methods).

You can also invert the mask:

```python
# Inverted: pixels BELOW 127 become white
ret, mask_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
```

## Combining Masks

You can combine multiple masks using bitwise operations to create complex selection regions:

```python
# Create two separate masks
mask_circle = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask_circle, (150, 150), 80, 255, -1)

mask_rect = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask_rect, (100, 100), (250, 250), 255, -1)

# AND: only where BOTH masks overlap
combined_and = cv2.bitwise_and(mask_circle, mask_rect)

# OR: where EITHER mask has white
combined_or = cv2.bitwise_or(mask_circle, mask_rect)

# Subtract: circle minus the rectangle overlap
mask_subtract = cv2.bitwise_and(mask_circle, cv2.bitwise_not(mask_rect))
```

This lets you build up complex shapes from simple primitives.

## Applying a Mask to a Color Image

To apply a single-channel mask to a 3-channel color image, use `cv2.bitwise_and` with the `mask` parameter:

```python
# The mask must be single-channel, same height and width
result = cv2.bitwise_and(color_image, color_image, mask=mask)
```

Both `src1` and `src2` are the same image — you're not combining two images, you're just filtering one image through the mask.

## Inverting a Mask

To flip what's visible and what's hidden:

```python
mask_inverted = cv2.bitwise_not(mask)
```

This swaps all 0s to 255 and all 255s to 0. Useful when you want to process **everything except** the masked region.

## Tips & Common Mistakes

- A mask must be **single-channel `uint8`** — shape `(h, w)`, not `(h, w, 3)`.
- A mask must be the **same height and width** as the image you're masking.
- Use `255` for white (visible) regions, not `1`. OpenCV checks for non-zero, but `255` is the convention.
- `cv2.threshold` returns **two values** — don't forget to unpack: `ret, mask = cv2.threshold(...)`.
- To create a mask from a color image, **convert to grayscale first** with `cv2.cvtColor`.
- Combining masks with AND gives you intersection; OR gives you union.
- To "cut out" a shape from another, use AND with the NOT of the shape you want to remove.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful source image with regions
h, w = 300, 400
image = np.zeros((h, w, 3), dtype=np.uint8)

# Paint four colored quadrants
image[0:150, 0:200] = (255, 100, 50)     # Top-left: blue-ish
image[0:150, 200:400] = (50, 255, 100)   # Top-right: green-ish
image[150:300, 0:200] = (50, 100, 255)   # Bottom-left: red-ish
image[150:300, 200:400] = (200, 200, 50) # Bottom-right: cyan-ish

# --- Create a circular mask ---
mask_circle = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask_circle, (200, 150), 100, 255, -1)

# --- Create a rectangular mask ---
mask_rect = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(mask_rect, (100, 50), (300, 250), 255, -1)

# --- Combine masks ---
mask_and = cv2.bitwise_and(mask_circle, mask_rect)    # Intersection
mask_or  = cv2.bitwise_or(mask_circle, mask_rect)     # Union
mask_sub = cv2.bitwise_and(mask_circle, cv2.bitwise_not(mask_rect))  # Circle minus rect

# --- Apply masks to the colorful image ---
result_circle = cv2.bitwise_and(image, image, mask=mask_circle)
result_rect   = cv2.bitwise_and(image, image, mask=mask_rect)
result_and    = cv2.bitwise_and(image, image, mask=mask_and)
result_or     = cv2.bitwise_and(image, image, mask=mask_or)

# Convert masks to 3-channel for display
m_circle_3 = cv2.cvtColor(mask_circle, cv2.COLOR_GRAY2BGR)
m_rect_3   = cv2.cvtColor(mask_rect, cv2.COLOR_GRAY2BGR)
m_and_3    = cv2.cvtColor(mask_and, cv2.COLOR_GRAY2BGR)
m_or_3     = cv2.cvtColor(mask_or, cv2.COLOR_GRAY2BGR)

# Label everything
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(m_circle_3, 'Circle Mask', (10, 25), font, 0.5, (0, 255, 0), 1)
cv2.putText(m_rect_3, 'Rect Mask', (10, 25), font, 0.5, (0, 255, 0), 1)
cv2.putText(m_and_3, 'AND Mask', (10, 25), font, 0.5, (0, 255, 0), 1)
cv2.putText(m_or_3, 'OR Mask', (10, 25), font, 0.5, (0, 255, 0), 1)
cv2.putText(result_circle, 'Circle Applied', (10, 25), font, 0.5, (255, 255, 255), 1)
cv2.putText(result_rect, 'Rect Applied', (10, 25), font, 0.5, (255, 255, 255), 1)
cv2.putText(result_and, 'AND Applied', (10, 25), font, 0.5, (255, 255, 255), 1)
cv2.putText(result_or, 'OR Applied', (10, 25), font, 0.5, (255, 255, 255), 1)

# Arrange: top row = masks, bottom row = applied results
top_row = np.hstack([m_circle_3, m_rect_3, m_and_3, m_or_3])
bot_row = np.hstack([result_circle, result_rect, result_and, result_or])
result = np.vstack([top_row, bot_row])

print('Top row: Circle mask | Rect mask | AND mask | OR mask')
print('Bottom:  Circle applied | Rect applied | AND applied | OR applied')

cv2.imshow('Creating Masks', result)
```
