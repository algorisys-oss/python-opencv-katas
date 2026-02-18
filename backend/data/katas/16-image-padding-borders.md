---
slug: 16-image-padding-borders
title: Image Padding & Borders
level: beginner
concepts: [cv2.copyMakeBorder, BORDER_CONSTANT, BORDER_REFLECT, BORDER_REPLICATE]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

When you apply a filter (like blur or edge detection) to an image, the filter needs to look at **neighboring pixels** — but at the edges of the image, there are no neighbors. The solution is to **pad** the image with extra pixels around the border before processing.

Padding is also useful for:
- Adding decorative borders or frames around images.
- Making images a specific size without stretching (e.g., making a non-square image square).
- Preparing images for neural networks that require fixed input dimensions.

## cv2.copyMakeBorder

This function adds padding around all four sides of an image:

```python
padded = cv2.copyMakeBorder(src, top, bottom, left, right, borderType)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image |
| `top` | Padding in pixels on the top |
| `bottom` | Padding in pixels on the bottom |
| `left` | Padding in pixels on the left |
| `right` | Padding in pixels on the right |
| `borderType` | How to fill the padded area |

```python
# Add 20px of padding on all sides with black border
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
```

## Border Types Explained

### BORDER_CONSTANT — Fill with a solid color

```python
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                            cv2.BORDER_CONSTANT, value=(255, 0, 0))
```

The `value` parameter sets the color (BGR). This is the only border type that uses `value`.

```
value value | A B C D | value value
value value | E F G H | value value
```

### BORDER_REFLECT — Mirror the edge pixels

```python
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REFLECT)
```

Pixels are reflected like a mirror at the border:

```
C B | A B C D | C B
G F | E F G H | G F
```

This is very common for image filtering because it creates smooth, natural-looking borders.

### BORDER_REFLECT_101 — Mirror without repeating the edge pixel

```python
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REFLECT_101)
```

Similar to REFLECT, but the edge pixel itself is not duplicated:

```
D C | A B C D | B A
H G | E F G H | F E
```

This is actually the **default** border mode in many OpenCV functions like `cv2.GaussianBlur`.

### BORDER_REPLICATE — Repeat the edge pixel

```python
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
```

The outermost pixel is stretched outward:

```
A A | A B C D | D D
E E | E F G H | H H
```

### BORDER_WRAP — Wrap around (tile)

```python
padded = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_WRAP)
```

The opposite side of the image is used to fill:

```
C D | A B C D | A B
G H | E F G H | E F
```

## Custom Border Color

Only `BORDER_CONSTANT` supports a custom color via the `value` parameter:

```python
# Red border
padded = cv2.copyMakeBorder(img, 30, 30, 30, 30,
                            cv2.BORDER_CONSTANT, value=(0, 0, 255))

# White border
padded = cv2.copyMakeBorder(img, 30, 30, 30, 30,
                            cv2.BORDER_CONSTANT, value=(255, 255, 255))
```

## Asymmetric Padding

You can pad different amounts on each side:

```python
# More padding on top and bottom than left and right
padded = cv2.copyMakeBorder(img, 50, 50, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
```

This is useful when you need to make a landscape image square — add more padding to top and bottom.

## Why Does Border Type Matter?

For image filtering (blur, edge detection), the border type affects the result at the edges:
- **REFLECT** and **REFLECT_101** produce the most natural-looking results because the padded area resembles the actual image content.
- **CONSTANT** (zero-padding) can create dark edges in blurred images.
- **REPLICATE** works well when the edges are relatively uniform in color.

## Tips & Common Mistakes

- The output image is **larger** than the input: new width = `left + original_width + right`.
- `value` is only used with `BORDER_CONSTANT` — it's ignored for other border types.
- `value` takes a BGR tuple for color images, or a single int for grayscale.
- `BORDER_REFLECT_101` is OpenCV's default in most filtering functions — use it when you want consistent behavior.
- Don't forget the `value=` keyword when setting a custom color: `cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))`.
- Padding does not modify the original image — it returns a new, larger image.

## Starter Code

```python
import cv2
import numpy as np

# Create a small colorful image to clearly see border effects
img = np.zeros((100, 150, 3), dtype=np.uint8)

# Paint a gradient pattern so border effects are visible
for y in range(100):
    for x in range(150):
        img[y, x] = (int(255 * x / 150), int(255 * y / 100), 128)

# Add a small marker so orientation is clear
cv2.rectangle(img, (10, 10), (40, 30), (255, 255, 255), -1)
cv2.putText(img, 'AB', (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

pad = 40  # padding size in pixels

# --- Apply different border types ---
border_const = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 200))

border_reflect = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT)

border_reflect101 = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                        cv2.BORDER_REFLECT_101)

border_replicate = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                       cv2.BORDER_REPLICATE)

border_wrap = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                  cv2.BORDER_WRAP)

# Custom colored border (white frame)
border_custom = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

# Add labels to each
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(border_const, 'CONSTANT', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(border_reflect, 'REFLECT', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(border_reflect101, 'REFLECT_101', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(border_replicate, 'REPLICATE', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(border_wrap, 'WRAP', (5, 20), font, 0.5, (255, 255, 255), 1)
cv2.putText(border_custom, 'WHITE', (5, 20), font, 0.5, (0, 0, 0), 1)

# Stack into a grid: 2 rows x 3 columns
top_row = np.hstack([border_const, border_reflect, border_reflect101])
bot_row = np.hstack([border_replicate, border_wrap, border_custom])
result = np.vstack([top_row, bot_row])

print(f'Original image size: {img.shape[1]}x{img.shape[0]}')
print(f'Padded image size:   {border_const.shape[1]}x{border_const.shape[0]}')
print(f'Padding added: {pad}px on each side')

cv2.imshow('Image Padding & Borders', result)
```
