---
slug: 10-image-flipping
title: Image Flipping
level: beginner
concepts: [cv2.flip, horizontal flip, vertical flip]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

Flipping an image means mirroring it along an axis — horizontally (left-right), vertically (top-bottom), or both. This is one of the simplest geometric operations, but it has important practical uses: creating mirror effects, correcting mirrored webcam feeds, and especially **data augmentation** in machine learning (flipping training images effectively doubles your dataset size).

## cv2.flip() Basics

```python
flipped = cv2.flip(img, flipCode)
```

The entire operation is controlled by a single integer parameter, `flipCode`:

| `flipCode` | Effect | Axis of Reflection |
|---|---|---|
| `1` | **Horizontal flip** (left-right mirror) | Vertical axis (y-axis) |
| `0` | **Vertical flip** (top-bottom mirror) | Horizontal axis (x-axis) |
| `-1` | **Both** (horizontal + vertical) | Both axes (180-degree rotation) |

> **Naming can be confusing:** A "horizontal flip" mirrors **across the vertical axis**, so the image content moves horizontally. Think of it as: "the flip code tells you which axis to flip **around**" — `1` = around y-axis (horizontal flip), `0` = around x-axis (vertical flip).

## Horizontal Flip (flipCode = 1)

This mirrors the image left to right, like looking in a mirror:

```python
h_flip = cv2.flip(img, 1)
```

The leftmost column becomes the rightmost, and vice versa. Rows stay in the same position.

**Common uses:**
- Correcting selfie/webcam images that appear mirrored
- Data augmentation — objects look equally valid when flipped horizontally
- Creating symmetrical designs

## Vertical Flip (flipCode = 0)

This mirrors the image top to bottom:

```python
v_flip = cv2.flip(img, 0)
```

The top row becomes the bottom row, and vice versa. Columns stay in the same position.

**Common uses:**
- Correcting upside-down images from certain camera sensors
- Data augmentation (less commonly used than horizontal, since many objects look unnatural upside down)

## Both Axes Flip (flipCode = -1)

This flips both horizontally and vertically simultaneously, which is equivalent to rotating the image 180 degrees:

```python
both_flip = cv2.flip(img, -1)
```

```python
# These produce the same result:
both = cv2.flip(img, -1)
same = cv2.flip(cv2.flip(img, 0), 1)
also_same = cv2.rotate(img, cv2.ROTATE_180)
```

## Flipping and NumPy Equivalents

Since images are NumPy arrays, you can also flip using NumPy directly. However, `cv2.flip()` is optimized and typically faster:

```python
# Horizontal flip (equivalent to cv2.flip(img, 1))
h_flip_np = img[:, ::-1]

# Vertical flip (equivalent to cv2.flip(img, 0))
v_flip_np = img[::-1, :]

# Both (equivalent to cv2.flip(img, -1))
both_np = img[::-1, ::-1]
```

The NumPy versions return **views** (not copies), so they use no extra memory. `cv2.flip()` creates a new array. Use NumPy slicing if memory matters; use `cv2.flip()` for clarity and when you need a proper copy.

## Practical Use: Data Augmentation

In machine learning, flipping is one of the easiest ways to augment training data:

```python
# Simple augmentation: for each training image, also use its horizontal flip
images = [original]
images.append(cv2.flip(original, 1))  # horizontal flip

# For some tasks (e.g., satellite imagery), vertical flips are also valid
images.append(cv2.flip(original, 0))
images.append(cv2.flip(original, -1))
# Now you have 4x the training data
```

**Caution:** Only apply flips that make semantic sense for your task. Horizontal flips work for most natural images, but vertical flips may not (e.g., text becomes unreadable, gravity-dependent scenes look wrong).

## Tips & Common Mistakes

- `flipCode` values: `1` = horizontal, `0` = vertical, `-1` = both. These are easy to mix up — remember `1` corresponds to the y-axis (horizontal mirror).
- `cv2.flip()` returns a **new** image. The original is unchanged.
- Flipping is its own inverse: flipping the same way twice gives you back the original. `cv2.flip(cv2.flip(img, 1), 1)` equals `img`.
- For 90-degree and 270-degree rotations, use `cv2.rotate()` instead of `cv2.flip()`. Flipping is only for mirror reflections.
- When augmenting data, if your images have **labels** (bounding boxes, keypoints), you must flip the labels too. A bounding box at `(x, y)` in the original becomes `(width - x - box_w, y)` after a horizontal flip.
- `cv2.flip()` works on any number of channels (grayscale, BGR, BGRA, etc.).

## Starter Code

```python
import cv2
import numpy as np

# Create an asymmetric image so flip directions are clearly visible
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:] = (40, 35, 30)

# Draw an "F" shape — asymmetric in both axes, making flips obvious
# Vertical bar of the F
cv2.rectangle(img, (60, 40), (100, 260), (0, 180, 220), -1)
# Top horizontal bar
cv2.rectangle(img, (100, 40), (230, 80), (0, 180, 220), -1)
# Middle horizontal bar
cv2.rectangle(img, (100, 120), (190, 155), (0, 180, 220), -1)

# Add a colored dot in top-left corner as orientation marker
cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)  # Red dot

# Label the original
cv2.putText(img, 'F', (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1, cv2.LINE_AA)

# --- Apply all three flip modes ---
h_flip = cv2.flip(img, 1)    # Horizontal (left-right mirror)
v_flip = cv2.flip(img, 0)    # Vertical (top-bottom mirror)
both_flip = cv2.flip(img, -1) # Both axes (180-degree rotation)

# --- Label each result ---
cv2.putText(img, 'Original', (10, 290), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(h_flip, 'Flip H (code=1)', (10, 290), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(v_flip, 'Flip V (code=0)', (10, 290), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(both_flip, 'Flip Both (code=-1)', (10, 290), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1, cv2.LINE_AA)

# --- Arrange in a 2x2 grid ---
top_row = np.hstack([img, h_flip])
bottom_row = np.hstack([v_flip, both_flip])
grid = np.vstack([top_row, bottom_row])

print(f'Original shape: {img.shape}')
print(f'flipCode  1  (horizontal): left-right mirror')
print(f'flipCode  0  (vertical):   top-bottom mirror')
print(f'flipCode -1  (both):       180-degree rotation')

cv2.imshow('Image Flipping', grid)
```
