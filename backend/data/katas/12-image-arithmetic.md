---
slug: 12-image-arithmetic
title: Image Arithmetic
level: beginner
concepts: [cv2.add, cv2.subtract, saturation arithmetic, numpy arithmetic]
prerequisites: [01-image-loading, 03-pixel-access]
---

## What Problem Are We Solving?

Adding or subtracting values from pixel intensities is how you adjust **brightness**, blend images together, or compute differences between frames. But there is a critical subtlety: pixel values are stored as unsigned 8-bit integers (`uint8`), which can only hold values from 0 to 255. What happens when you add 200 + 100? The answer depends on whether you use **OpenCV arithmetic** or **NumPy arithmetic** — and getting this wrong produces wildly incorrect images.

## The Core Problem: Overflow and Underflow

A `uint8` pixel value ranges from 0 to 255. When arithmetic pushes a value outside this range, something must happen:

```python
# What should 200 + 100 be? It exceeds 255.
# What should 50 - 80 be? It goes below 0.
```

There are two different behaviors:

| Method | Overflow (>255) | Underflow (<0) | Name |
|---|---|---|---|
| `cv2.add()` | Clips to **255** | Clips to **0** | **Saturation** arithmetic |
| NumPy `+` | Wraps to **44** (300 % 256) | Wraps to **226** (-30 % 256) | **Modulo** (wrapping) arithmetic |

This difference is not a minor detail — it fundamentally changes your image.

## cv2.add() — Saturation Arithmetic

`cv2.add()` clips results to the valid range [0, 255]:

```python
result = cv2.add(img1, img2)
```

```python
# Adding two images pixel by pixel
bright = cv2.add(img, img)  # Every pixel doubled, capped at 255

# Adding a scalar to increase brightness
brighter = cv2.add(img, np.full_like(img, 50))
```

When a value would exceed 255, it is **clamped** to 255. When it would go below 0 (with `cv2.subtract`), it is clamped to 0.

```python
import numpy as np

a = np.array([200], dtype=np.uint8)
b = np.array([100], dtype=np.uint8)

print(cv2.add(a, b))  # [[255]]  — clamped at 255
print(a + b)           # [44]     — 300 % 256 = 44 (WRONG!)
```

## NumPy + Operator — Modulo Wrapping

NumPy performs modulo-256 arithmetic on `uint8` arrays:

```python
result = img1 + img2  # DANGER: wraps around!
```

A pixel at 200 + 100 = 300 becomes 300 % 256 = **44** — a bright pixel suddenly becomes dark. This is almost never what you want for image processing.

```python
a = np.array([200, 10], dtype=np.uint8)
b = np.array([100, 250], dtype=np.uint8)

# NumPy wrapping
print(a + b)           # [ 44   4]  — both wrapped!

# OpenCV saturation
print(cv2.add(a, b))   # [[255] [255]]  — clamped correctly
```

## cv2.subtract() — Saturation Subtraction

```python
result = cv2.subtract(img1, img2)
```

Values that would go below 0 are clamped to 0:

```python
a = np.array([50], dtype=np.uint8)
b = np.array([80], dtype=np.uint8)

print(cv2.subtract(a, b))  # [[0]]    — clamped at 0
print(a - b)                # [226]    — 50 - 80 = -30 % 256 = 226 (WRONG!)
```

With NumPy subtraction, a dark pixel minus a slightly brighter value produces a **very bright** pixel (226), which is completely wrong.

## Brightness Adjustment

The most common use of image arithmetic is adjusting brightness — adding or subtracting a constant value from every pixel:

```python
# Increase brightness by 60
bright_img = cv2.add(img, np.full(img.shape, 60, dtype=np.uint8))

# Decrease brightness by 40
dark_img = cv2.subtract(img, np.full(img.shape, 40, dtype=np.uint8))
```

A shorter way using a scalar tuple (works because OpenCV broadcasts it):

```python
# Increase brightness by 60 (for BGR images)
bright_img = cv2.add(img, (60, 60, 60, 0))

# Decrease brightness by 40
dark_img = cv2.subtract(img, (40, 40, 40, 0))
```

> **Note:** The scalar tuple must have 4 elements (OpenCV always expects 4 channels for scalar operations, even for 3-channel images). The 4th value is ignored for BGR images.

## Why Saturation Matters Visually

Consider brightening an image with NumPy wrapping vs. OpenCV saturation:

- **Saturation (cv2.add):** Bright areas become pure white (255). The image looks washed out but still recognizable.
- **Wrapping (NumPy +):** Bright areas wrap to dark values. Bright skies become black, white shirts become dark gray. The image looks **corrupted**.

This is why you should **always use `cv2.add()` and `cv2.subtract()`** for image arithmetic, never plain NumPy `+` and `-`.

## Adding Two Images Together

You can add two images of the **same size and type** pixel by pixel:

```python
# Both images must be the same shape and dtype
combined = cv2.add(img1, img2)
```

This is a simple additive blend — useful for overlaying effects, combining masks, or creating double-exposure effects. For weighted blending, use `cv2.addWeighted()` instead (covered in a later kata).

## Tips & Common Mistakes

- **Always use `cv2.add()` / `cv2.subtract()`** instead of NumPy `+` / `-` for images. Wrapping arithmetic corrupts images in subtle, hard-to-debug ways.
- Both images passed to `cv2.add()` must have the **same shape** and **same dtype**. Mismatched shapes cause an error.
- When using a scalar value (like brightness adjustment), create a full array with `np.full()` or `np.full_like()`, or use a 4-element tuple.
- Saturation means information is lost at the extremes — if many pixels are already at 255 and you add more, they stay at 255. You cannot recover the original values by subtracting later.
- `cv2.add()` and `cv2.subtract()` always return `uint8` arrays when given `uint8` inputs. If you need higher precision, convert to `float32` first, do the math, then convert back.
- The visual difference between saturation and wrapping is dramatic. If your brightened image looks "psychedelic" with inverted colors, you are probably using NumPy arithmetic by accident.

## Starter Code

```python
import cv2
import numpy as np

# Create a gradient image to clearly show arithmetic behavior
img = np.zeros((200, 300, 3), dtype=np.uint8)

# Horizontal gradient: black on left (0) to white on right (255)
for x in range(300):
    val = int(x * 255 / 299)
    img[:, x] = (val, val, val)

# Add a colored rectangle for visual interest
cv2.rectangle(img, (20, 20), (100, 80), (0, 100, 200), -1)
cv2.putText(img, 'Original', (110, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 2, cv2.LINE_AA)

# --- Demonstrate saturation vs wrapping ---
add_value = 100
brightness = np.full(img.shape, add_value, dtype=np.uint8)

# OpenCV saturation add (correct)
bright_cv = cv2.add(img, brightness)

# NumPy wrapping add (wrong for images!)
bright_np = img + brightness  # DON'T do this for real images

# --- Brightness adjustment ---
dark_cv = cv2.subtract(img, np.full(img.shape, 80, dtype=np.uint8))

# --- Saturation vs wrapping subtraction ---
sub_value = 120
sub_arr = np.full(img.shape, sub_value, dtype=np.uint8)
sub_cv = cv2.subtract(img, sub_arr)   # Correct: clips to 0
sub_np = img - sub_arr                 # Wrong: wraps to 255-ish

# --- Label the images ---
cv2.putText(bright_cv, f'cv2.add +{add_value}', (5, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(bright_np, f'numpy + {add_value}', (5, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(dark_cv, 'cv2.subtract -80', (5, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(sub_cv, f'cv2.sub -{sub_value}', (5, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(sub_np, f'numpy - {sub_value}', (5, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

# --- Build comparison grid ---
# Row 1: Original, cv2.add, numpy +
row1 = np.hstack([img, bright_cv, bright_np])

# Row 2: darkened, cv2.subtract, numpy -
row2 = np.hstack([dark_cv, sub_cv, sub_np])

grid = np.vstack([row1, row2])

# Print numerical proof of the difference
print('=== Saturation vs Wrapping ===')
print(f'Pixel value 200 + 100:')
a = np.array([[200]], dtype=np.uint8)
b = np.array([[100]], dtype=np.uint8)
print(f'  cv2.add:  {cv2.add(a, b)[0][0]}  (clipped to 255)')
print(f'  numpy +:  {(a + b)[0][0]}  (wrapped to {(200+100) % 256})')
print()
print(f'Pixel value 50 - 80:')
c = np.array([[50]], dtype=np.uint8)
d = np.array([[80]], dtype=np.uint8)
print(f'  cv2.sub:  {cv2.subtract(c, d)[0][0]}  (clipped to 0)')
print(f'  numpy -:  {(c - d)[0][0]}  (wrapped to {(50-80) % 256})')

cv2.imshow('Image Arithmetic', grid)
```
