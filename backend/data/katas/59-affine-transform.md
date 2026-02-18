---
slug: 59-affine-transform
title: Affine Transform
level: intermediate
concepts: [cv2.getAffineTransform, cv2.warpAffine, three-point mapping]
prerequisites: [09-image-rotation]
---

## What Problem Are We Solving?

You need to apply a combination of **rotation, scaling, translation, and shearing** to an image -- all in a single operation. Rather than applying each transformation separately (which requires multiple interpolation passes and accumulates quality loss), an **affine transform** lets you express all these geometric changes in one 2x3 matrix. This is also essential for aligning images, correcting skew, and mapping between coordinate systems.

## Affine vs Perspective Transforms

Both are geometric transforms, but they have a key difference:

| Property | Affine | Perspective |
|---|---|---|
| Points needed | 3 | 4 |
| Matrix size | 2x3 | 3x3 |
| Parallel lines | Preserved | May converge |
| Rectangles become | Parallelograms | Any quadrilateral |
| Use case | Rotation, scaling, shear, translation | Camera angle correction |

Affine transforms preserve **parallelism** -- parallel lines remain parallel after the transform. Perspective transforms do not. If your distortion involves converging lines (like a photo taken at an angle), you need perspective. For rotation, scaling, and shear, affine is the right choice.

## Three-Point Mapping

An affine transform is defined by how **three points** in the source image map to three points in the destination. These three point pairs uniquely determine the 2x3 transformation matrix:

```python
src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
dst_pts = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(src_pts, dst_pts)
```

| Parameter | Meaning |
|---|---|
| `src_pts` | Three source points as `float32` array of shape `(3, 2)` |
| `dst_pts` | Three destination points as `float32` array of shape `(3, 2)` |

The returned `M` is a 2x3 `float64` matrix.

## Understanding the 2x3 Matrix

The affine matrix `M` encodes all the geometric operations:

```python
# M = [[a, b, tx],
#       [c, d, ty]]
#
# For each point (x, y):
# new_x = a*x + b*y + tx
# new_y = c*x + d*y + ty
```

Different matrix values produce different effects:

```python
# Identity (no change)
M_identity = np.float32([[1, 0, 0], [0, 1, 0]])

# Pure translation (shift by tx, ty)
M_translate = np.float32([[1, 0, 50], [0, 1, 30]])

# Scale by 2x
M_scale = np.float32([[2, 0, 0], [0, 2, 0]])

# Horizontal shear
M_shear = np.float32([[1, 0.3, 0], [0, 1, 0]])
```

## Applying with cv2.warpAffine()

```python
result = cv2.warpAffine(img, M, (width, height))
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `M` | 2x3 affine transformation matrix |
| `(width, height)` | Output image size (cols, rows) |

Additional useful parameters:

```python
# With border handling
result = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(128, 128, 128))
```

## Combining Rotation + Scaling + Translation

Instead of defining three point pairs, you can build the matrix from rotation parameters using `cv2.getRotationMatrix2D()`, and then add translation by modifying the matrix directly:

```python
# Rotate 30 degrees around center, scale by 0.8
center = (img.shape[1] // 2, img.shape[0] // 2)
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# Add translation: shift 50 pixels right, 20 pixels down
M[0, 2] += 50
M[1, 2] += 20

result = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
```

This is often more intuitive than specifying three point pairs when you know the specific operations you want.

## Building Custom Affine Matrices

You can compose transforms by matrix multiplication. Since affine matrices are 2x3, extend them to 3x3 for multiplication, then truncate back:

```python
def make_3x3(M_2x3):
    M = np.vstack([M_2x3, [0, 0, 1]])
    return M

# Rotation around center
M_rot = cv2.getRotationMatrix2D((150, 150), 45, 1.0)
M_rot_3x3 = make_3x3(M_rot)

# Scaling
M_scale_3x3 = np.float64([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 1]])

# Combined: scale then rotate
M_combined = M_rot_3x3 @ M_scale_3x3
M_final = M_combined[:2, :]  # Back to 2x3

result = cv2.warpAffine(img, M_final, (300, 300))
```

## Shearing

Shearing slants the image along one axis. It's a unique property of affine transforms that can't be achieved with just rotation and scaling:

```python
# Horizontal shear: tilts the image sideways
M_shear_h = np.float32([[1, 0.3, 0], [0, 1, 0]])

# Vertical shear: tilts the image up/down
M_shear_v = np.float32([[1, 0, 0], [0.3, 1, 0]])
```

The shear factor (0.3 in these examples) controls how much the image slants. Positive values shear in one direction, negative in the other.

## Tips & Common Mistakes

- Point arrays must be `np.float32`, not `float64` or integers. `cv2.getAffineTransform()` is strict about this.
- Exactly **three points** are required. Using fewer or more causes an error (for more points, use `cv2.estimateAffinePartial2D()` or `cv2.estimateAffine2D()`).
- The output size in `cv2.warpAffine()` is `(width, height)` -- columns first. Mixing this up swaps the dimensions.
- Pixels that map outside the output bounds are lost. Make the output size large enough to contain the transformed image, or use `borderMode` to handle edges.
- The three source points should **not be collinear** (all on the same line). Collinear points don't define a unique 2D transform.
- When building custom matrices, remember that rotation matrices from `cv2.getRotationMatrix2D()` already include the translation to rotate around the specified center.
- Multiple `cv2.warpAffine()` calls degrade quality. Combine transforms into a single matrix whenever possible.
- For reflections (mirroring), use negative scale factors: `[[-1, 0, w], [0, 1, 0]]` mirrors horizontally.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with clear orientation markers
img = np.full((300, 300, 3), (200, 200, 200), dtype=np.uint8)
cv2.rectangle(img, (40, 40), (260, 260), (180, 100, 50), 2)
cv2.circle(img, (80, 80), 20, (0, 0, 255), -1)       # Red: top-left
cv2.circle(img, (220, 80), 20, (0, 255, 0), -1)       # Green: top-right
cv2.circle(img, (80, 220), 20, (255, 0, 0), -1)       # Blue: bottom-left
cv2.putText(img, 'TL', (68, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(img, 'TR', (208, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(img, 'BL', (68, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.arrowedLine(img, (150, 150), (250, 150), (0, 0, 0), 2, tipLength=0.3)
cv2.putText(img, 'X', (245, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# --- Transform 1: Three-point mapping ---
src_pts = np.float32([[40, 40], [260, 40], [40, 260]])
dst_pts = np.float32([[60, 20], [240, 60], [20, 260]])
M1 = cv2.getAffineTransform(src_pts, dst_pts)
result1 = cv2.warpAffine(img, M1, (300, 300))

# --- Transform 2: Rotation + Scaling + Translation ---
center = (150, 150)
M2 = cv2.getRotationMatrix2D(center, 25, 0.8)  # 25 degrees, scale 0.8
M2[0, 2] += 30   # Shift right by 30
M2[1, 2] += 10   # Shift down by 10
result2 = cv2.warpAffine(img, M2, (300, 300))

# --- Transform 3: Horizontal shear ---
M3 = np.float32([[1, 0.3, 0], [0, 1, 0]])
result3 = cv2.warpAffine(img, M3, (400, 300))
result3 = cv2.resize(result3, (300, 300))  # Resize for consistent display

# --- Transform 4: Vertical shear ---
M4 = np.float32([[1, 0, 0], [0.3, 1, 0]])
result4 = cv2.warpAffine(img, M4, (300, 400))
result4 = cv2.resize(result4, (300, 300))  # Resize for consistent display

# --- Transform 5: Reflection (horizontal mirror) ---
M5 = np.float32([[-1, 0, 300], [0, 1, 0]])
result5 = cv2.warpAffine(img, M5, (300, 300))

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Original', (5, 295), font, 0.5, (0, 0, 0), 1)
cv2.putText(result1, '3-Point Map', (5, 295), font, 0.5, (0, 0, 0), 1)
cv2.putText(result2, 'Rot+Scale+Trans', (5, 295), font, 0.5, (0, 0, 0), 1)
cv2.putText(result3, 'H Shear', (5, 295), font, 0.5, (0, 0, 0), 1)
cv2.putText(result4, 'V Shear', (5, 295), font, 0.5, (0, 0, 0), 1)
cv2.putText(result5, 'H Mirror', (5, 295), font, 0.5, (0, 0, 0), 1)

# Build grid: 2 rows x 3 columns
row1 = np.hstack([img, result1, result2])
row2 = np.hstack([result3, result4, result5])
result = np.vstack([row1, row2])

print(f'Image shape: {img.shape}')
print(f'3-point mapping matrix:\n{M1}')
print(f'Rotation+Scale+Translation matrix:\n{M2}')
print(f'Shear matrix:\n{M3}')
print(f'Mirror matrix:\n{M5}')

cv2.imshow('Affine Transform', result)
```
