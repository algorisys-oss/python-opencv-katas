---
slug: 09-image-rotation
title: Image Rotation
level: beginner
concepts: [cv2.getRotationMatrix2D, cv2.warpAffine, rotation center]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

You need to rotate an image — perhaps to correct a tilted scan, align a document, or create data augmentation for training. OpenCV provides rotation through **affine transformations**: you build a 2x3 rotation matrix and apply it with `cv2.warpAffine()`. Understanding how this works gives you control over the rotation center, the angle, optional scaling, and whether the rotated image gets clipped or the canvas expands to fit.

## The Rotation Matrix

`cv2.getRotationMatrix2D()` constructs a 2x3 matrix that describes a rotation (and optional scaling) around a given center point:

```python
M = cv2.getRotationMatrix2D(center, angle, scale)
```

| Parameter | Meaning |
|---|---|
| `center` | The `(x, y)` point to rotate around |
| `angle` | Rotation angle in **degrees** — positive = counter-clockwise |
| `scale` | Scale factor applied during rotation (1.0 = no scaling) |

The returned matrix `M` is a NumPy array of shape `(2, 3)` containing float values. You do not need to understand the math inside it to use it, but it encodes: "for every pixel in the output, where should I sample from the input?"

```python
# Rotate 45 degrees counter-clockwise around the image center, no scaling
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
```

## Applying the Rotation with warpAffine

Once you have the matrix, apply it with `cv2.warpAffine()`:

```python
rotated = cv2.warpAffine(img, M, (output_width, output_height))
```

| Parameter | Meaning |
|---|---|
| `img` | Source image |
| `M` | The 2x3 transformation matrix |
| `(output_width, output_height)` | Size of the output image — `(width, height)`, not `(height, width)` |

```python
rotated = cv2.warpAffine(img, M, (w, h))
```

If you pass the **same size** as the original, parts of the rotated image that fall outside the original bounds are **clipped** (lost). The empty areas are filled with black (zeros) by default.

## Rotation Around the Image Center

The most common rotation: spin the image around its own center.

```python
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

This rotates 30 degrees counter-clockwise. The corners of the image will be clipped because the rotated rectangle is larger than the original canvas.

## Rotation with Scaling

You can combine rotation and scaling in one step by setting the `scale` parameter:

```python
# Rotate 45 degrees and shrink to 70%
M = cv2.getRotationMatrix2D(center, 45, 0.7)
rotated = cv2.warpAffine(img, M, (w, h))
```

This is useful when you want the rotated image to fit within the original dimensions — shrinking it prevents clipping.

## Expanding the Canvas to Avoid Clipping

When you rotate an image by an angle other than 0/90/180/270, the bounding box of the rotated image is **larger** than the original. To avoid losing corners, you need to:

1. Compute the new bounding box size.
2. Adjust the rotation matrix to account for the new center.

```python
h, w = img.shape[:2]
center = (w // 2, h // 2)
angle = 30

M = cv2.getRotationMatrix2D(center, angle, 1.0)

# Compute the sine and cosine from the matrix
cos = abs(M[0, 0])
sin = abs(M[0, 1])

# New bounding dimensions
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)

# Adjust the matrix to translate to the new center
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

rotated = cv2.warpAffine(img, M, (new_w, new_h))
```

Now the entire rotated image is visible — no clipping. The extra space is filled with black pixels.

## Rotating Around a Custom Point

You can rotate around any point, not just the center. For example, rotate around the top-left corner:

```python
M = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

Or around a specific feature point:

```python
# Rotate around the point (100, 50)
M = cv2.getRotationMatrix2D((100, 50), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
```

## Specifying a Border Fill Color

By default, areas outside the source image are filled with black. You can change this:

```python
rotated = cv2.warpAffine(img, M, (new_w, new_h),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(128, 128, 128))  # gray fill
```

## Tips & Common Mistakes

- The angle is in **degrees**, not radians. Positive angles rotate **counter-clockwise**.
- The output size in `cv2.warpAffine()` is `(width, height)` — the reverse of `img.shape[:2]`.
- If the output size equals the input size, rotated corners **will be clipped**. Use the bounding box expansion technique to preserve the full image.
- The rotation center is `(x, y)`, not `(y, x)`. For center rotation, use `(w // 2, h // 2)`.
- `cv2.warpAffine()` returns a **new** image. The original is not modified.
- For 90/180/270-degree rotations, `cv2.rotate()` is simpler and faster: `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)`.
- Repeated rotations accumulate interpolation errors. Always rotate from the original image, not from an already-rotated version.

## Starter Code

```python
import cv2
import numpy as np

# Create a recognizable source image (asymmetric so rotation is visible)
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (40, 35, 30)

# Draw an arrow-like shape so orientation is obvious
cv2.rectangle(img, (50, 80), (350, 220), (0, 140, 200), -1)
cv2.putText(img, 'ROTATE ME', (70, 165), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 2, cv2.LINE_AA)
# Draw a small triangle pointer on the right side
pts = np.array([[350, 120], [390, 150], [350, 180]], np.int32)
cv2.fillPoly(img, [pts], (0, 200, 255))

h, w = img.shape[:2]
center = (w // 2, h // 2)

# --- Rotation 1: 30 degrees, same canvas (clipped) ---
M1 = cv2.getRotationMatrix2D(center, 30, 1.0)
rot_clipped = cv2.warpAffine(img, M1, (w, h))

# --- Rotation 2: 30 degrees, expanded canvas (no clipping) ---
angle = 30
M2 = cv2.getRotationMatrix2D(center, angle, 1.0)
cos = abs(M2[0, 0])
sin = abs(M2[0, 1])
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)
M2[0, 2] += (new_w / 2) - center[0]
M2[1, 2] += (new_h / 2) - center[1]
rot_expanded = cv2.warpAffine(img, M2, (new_w, new_h))

# --- Rotation 3: 45 degrees with scaling (shrink to fit) ---
M3 = cv2.getRotationMatrix2D(center, 45, 0.65)
rot_scaled = cv2.warpAffine(img, M3, (w, h))

# --- Rotation 4: -90 degrees (clockwise 90) ---
M4 = cv2.getRotationMatrix2D(center, -90, 1.0)
rot_90 = cv2.warpAffine(img, M4, (w, h))

# --- Build comparison display ---
display_h = 200
def fit_height(image, th):
    ih, iw = image.shape[:2]
    s = th / ih
    return cv2.resize(image, (int(iw * s), th), interpolation=cv2.INTER_LINEAR)

col1 = fit_height(img, display_h)
col2 = fit_height(rot_clipped, display_h)
col3 = fit_height(rot_expanded, display_h)
col4 = fit_height(rot_scaled, display_h)
col5 = fit_height(rot_90, display_h)

for label, col in [('Original', col1), ('30deg Clipped', col2),
                   ('30deg Expanded', col3), ('45deg + Scale', col4),
                   ('-90deg', col5)]:
    cv2.putText(col, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 255), 1, cv2.LINE_AA)

row = np.hstack([col1, col2, col3, col4, col5])

print(f'Original:    {img.shape[1]}x{img.shape[0]}')
print(f'Clipped:     {rot_clipped.shape[1]}x{rot_clipped.shape[0]}')
print(f'Expanded:    {rot_expanded.shape[1]}x{rot_expanded.shape[0]}')
print(f'Scaled:      {rot_scaled.shape[1]}x{rot_scaled.shape[0]}')

cv2.imshow('Image Rotation', row)
```
