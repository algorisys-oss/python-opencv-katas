---
slug: 46-flood-fill
title: Flood Fill
level: intermediate
concepts: [cv2.floodFill, seed point, tolerance]
prerequisites: [03-pixel-access]
---

## What Problem Are We Solving?

You click on a pixel in an image and want to select or recolor the entire connected region of similar color — exactly like the paint bucket tool in Photoshop or the magic wand selection tool. **Flood fill** starts from a seed point and spreads outward to neighboring pixels that fall within a color tolerance, filling them with a new color or marking them in a mask. It's useful for background removal, region selection, interactive segmentation, and removing uniform-colored areas.

## Basic Flood Fill

`cv2.floodFill()` fills a connected region starting from a seed point:

```python
retval, image, mask, rect = cv2.floodFill(image, mask, seedPoint, newVal)
```

| Parameter | Meaning |
|---|---|
| `image` | Input image (modified in place) |
| `mask` | A mask that is 2 pixels wider and taller than the image (can be `None` for simple fills) |
| `seedPoint` | The `(x, y)` starting pixel |
| `newVal` | The new color to fill with |

The simplest usage fills with a solid color:

```python
# Fill the region at (100, 100) with red
cv2.floodFill(img, None, (100, 100), (0, 0, 255))
```

This fills all connected pixels that have the **exact same color** as the seed pixel.

## Controlling Tolerance with loDiff and upDiff

Real images rarely have perfectly uniform regions. The `loDiff` and `upDiff` parameters define how different a neighboring pixel can be from the seed pixel (or from its neighbor) and still be included in the fill:

```python
retval, image, mask, rect = cv2.floodFill(
    image, mask, seedPoint, newVal,
    loDiff=(lo_b, lo_g, lo_r),
    upDiff=(up_b, up_g, up_r)
)
```

- `loDiff`: Maximum lower brightness/color difference (how much **darker** a pixel can be)
- `upDiff`: Maximum upper brightness/color difference (how much **brighter** a pixel can be)

```python
# Fill pixels within +/- 20 intensity of the seed pixel
cv2.floodFill(img, None, (100, 100), (0, 255, 0),
              loDiff=(20, 20, 20), upDiff=(20, 20, 20))
```

For a grayscale image, use single values:

```python
cv2.floodFill(gray, None, (100, 100), 128,
              loDiff=(15,), upDiff=(15,))
```

## Mask-Based Flood Fill

The mask parameter enables more powerful usage. The mask must be a single-channel `uint8` image that is **2 pixels wider and 2 pixels taller** than the input image:

```python
h, w = img.shape[:2]
mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
```

After flood fill, the mask will have non-zero values where the fill reached. This lets you know **which pixels were selected** without permanently modifying the image:

```python
h, w = img.shape[:2]
mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

# Fill and capture the mask
retval, _, mask, rect = cv2.floodFill(
    img.copy(), mask, (100, 100), (0, 255, 0),
    loDiff=(30, 30, 30), upDiff=(30, 30, 30)
)

# The fill region in the mask (strip the 1px border)
fill_region = mask[1:-1, 1:-1]
```

## Flood Fill Flags

The `flags` parameter controls fill behavior. You combine flags using bitwise OR:

```python
# Fill using fixed range (compare to seed pixel, not neighbor)
flags = 4 | cv2.FLOODFILL_FIXED_RANGE

# Set a custom mask fill value (e.g., 255)
flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
```

| Flag | Effect |
|---|---|
| `cv2.FLOODFILL_FIXED_RANGE` | Compare each pixel to the **seed** pixel, not to its neighbor |
| `cv2.FLOODFILL_MASK_ONLY` | Only update the mask, don't modify the image |
| `4` or `8` | 4-connectivity or 8-connectivity |

The default (without `FLOODFILL_FIXED_RANGE`) compares each pixel to its **neighbor**, so the fill can drift gradually into areas with very different colors from the seed. `FLOODFILL_FIXED_RANGE` prevents this drift.

## Magic Wand-Style Selection

To build a Photoshop-like magic wand tool, combine mask-based fill with `FLOODFILL_MASK_ONLY`:

```python
h, w = img.shape[:2]
mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
tolerance = 30

flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
cv2.floodFill(img, mask, seed_point, 0,
              loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
              flags=flags)

# Extract the selection mask (remove the 1px border)
selection = mask[1:-1, 1:-1]
```

The `(255 << 8)` in the flags sets the fill value in the mask to 255. Without this, the mask fill value defaults to 1.

## Tips & Common Mistakes

- The mask must be `(h+2, w+2)` in size — exactly 2 pixels wider and taller than the image. This is a hard requirement.
- `cv2.floodFill` modifies the image **in place** unless you use `FLOODFILL_MASK_ONLY`. Pass `img.copy()` if you want to preserve the original.
- The `seedPoint` is `(x, y)`, which is `(column, row)` — not `(row, column)`.
- Without `FLOODFILL_FIXED_RANGE`, the fill compares each pixel to its already-filled neighbor, which can cause the fill to "creep" through gradual gradients. Use `FLOODFILL_FIXED_RANGE` for predictable behavior.
- The mask also acts as a **barrier**: pre-set pixels in the mask (non-zero) will block the fill from spreading into those areas.
- `retval` returns the number of pixels that were filled.
- `rect` returns the bounding rectangle `(x, y, w, h)` of the filled region.
- For color images, `loDiff` and `upDiff` are 3-element tuples for `(B, G, R)` channels.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful scene with distinct regions
canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# Background gradient (so flood fill has something to work with)
for y in range(400):
    for x in range(600):
        canvas[y, x] = (100 + x // 10, 80, 60 + y // 8)

# Draw solid colored regions
cv2.rectangle(canvas, (50, 50), (200, 180), (200, 50, 50), -1)    # Blue region
cv2.rectangle(canvas, (250, 50), (400, 180), (50, 180, 50), -1)   # Green region
cv2.circle(canvas, (500, 120), 70, (50, 50, 200), -1)             # Red circle
cv2.rectangle(canvas, (50, 230), (550, 370), (180, 180, 60), -1)  # Teal bar

# Add some internal variation (so tolerance matters)
noise = np.random.randint(-15, 16, canvas.shape, dtype=np.int16)
canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Demo 1: Simple flood fill (exact color match, will barely spread due to noise) ---
demo1 = canvas.copy()
seed1 = (125, 115)  # Inside the blue rectangle
cv2.floodFill(demo1, None, seed1, (0, 255, 255))  # Fill with yellow
cv2.circle(demo1, seed1, 4, (255, 255, 255), -1)
cv2.putText(demo1, 'loDiff=0, upDiff=0', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Demo 2: Flood fill with tolerance ---
demo2 = canvas.copy()
seed2 = (125, 115)
cv2.floodFill(demo2, None, seed2, (0, 255, 255),
              loDiff=(25, 25, 25), upDiff=(25, 25, 25))
cv2.circle(demo2, seed2, 4, (255, 255, 255), -1)
cv2.putText(demo2, 'loDiff=25, upDiff=25', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Demo 3: Mask-based flood fill (magic wand selection) ---
demo3 = canvas.copy()
h, w = canvas.shape[:2]
mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
seed3 = (325, 115)  # Inside the green rectangle
tolerance = 30

flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
num_filled, _, mask, rect = cv2.floodFill(
    demo3, mask, seed3, 0,
    loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
    flags=flags
)

# Apply mask to highlight selection
selection = mask[1:-1, 1:-1]
highlight = demo3.copy()
highlight[selection == 255] = (0, 255, 255)
demo3 = cv2.addWeighted(canvas, 0.5, highlight, 0.5, 0)
demo3[selection == 255] = highlight[selection == 255]
cv2.circle(demo3, seed3, 4, (255, 255, 255), -1)
cv2.putText(demo3, f'Mask fill: {num_filled} px', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Demo 4: Fixed range vs default (neighbor) comparison ---
demo4 = canvas.copy()
seed4 = (300, 300)  # Inside the teal bar
cv2.floodFill(demo4, None, seed4, (0, 200, 255),
              loDiff=(20, 20, 20), upDiff=(20, 20, 20),
              flags=4 | cv2.FLOODFILL_FIXED_RANGE)
cv2.circle(demo4, seed4, 4, (255, 255, 255), -1)
cv2.putText(demo4, 'FIXED_RANGE tol=20', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Print results
print(f'Demo 1: Exact fill at {seed1}')
print(f'Demo 2: Tolerant fill at {seed2} (loDiff=upDiff=25)')
print(f'Demo 3: Mask-only fill at {seed3}, {num_filled} pixels selected')
print(f'Demo 4: Fixed-range fill at {seed4}')
print(f'Filled bounding rect: x={rect[0]}, y={rect[1]}, w={rect[2]}, h={rect[3]}')

# Stack results in 2x2 grid
top_row = np.hstack([demo1, demo2])
bot_row = np.hstack([demo3, demo4])
result = np.vstack([top_row, bot_row])

cv2.imshow('Flood Fill', result)
```
