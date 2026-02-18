---
slug: 58-perspective-transform
title: Perspective Transform
level: intermediate
concepts: [cv2.getPerspectiveTransform, cv2.warpPerspective, four-point transform]
prerequisites: [09-image-rotation]
---

## What Problem Are We Solving?

You've taken a photo of a document, whiteboard, or sign at an angle, and it looks skewed -- the rectangle appears as a trapezoid. **Perspective transformation** corrects this distortion by mapping four points from the skewed view to four points in a straight-on, rectangular view. This is the core operation behind document scanning apps, license plate reading, and augmented reality overlays.

## What is a Perspective Transform?

A perspective transform maps points from one 2D plane to another, accounting for the depth distortion that occurs when viewing a flat surface at an angle. Unlike simpler transforms (rotation, scaling), perspective transforms can change the shape of objects -- parallel lines may converge, and rectangles become trapezoids.

The transform is defined by a **3x3 matrix** that maps any point `(x, y)` in the source image to a point `(x', y')` in the destination image. You don't need to compute this matrix yourself -- OpenCV calculates it from four corresponding point pairs.

## Selecting Four Source and Destination Points

You need exactly **four points** in the source image (the corners of the region you want to straighten) and four corresponding destination points (where those corners should end up):

```python
# Source points: corners of the skewed document in the photo
src_pts = np.float32([
    [56, 65],    # Top-left
    [368, 52],   # Top-right
    [28, 387],   # Bottom-left
    [389, 390]   # Bottom-right
])

# Destination points: where we want those corners to be (a clean rectangle)
dst_pts = np.float32([
    [0, 0],        # Top-left
    [400, 0],      # Top-right
    [0, 400],      # Bottom-left
    [400, 400]     # Bottom-right
])
```

The points must be in `float32` format and the order must be consistent between source and destination (i.e., the first source point maps to the first destination point).

## Computing the Transform Matrix

`cv2.getPerspectiveTransform()` computes the 3x3 transformation matrix from the four point pairs:

```python
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
```

| Parameter | Meaning |
|---|---|
| `src_pts` | Four source points as `float32` array of shape `(4, 2)` |
| `dst_pts` | Four destination points as `float32` array of shape `(4, 2)` |

The returned matrix is a 3x3 `float64` array.

## Applying the Transform with cv2.warpPerspective()

Once you have the matrix, apply it to the image:

```python
result = cv2.warpPerspective(img, matrix, (width, height))
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `matrix` | 3x3 perspective transform matrix |
| `(width, height)` | Size of the output image |

The output size determines how large the result image is. For document scanning, you'd typically set this to a standard paper size or to match the actual document dimensions.

## Document Scanning Workflow

A typical document scanning pipeline:

```python
# 1. Detect the document corners (manually or with edge detection)
src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

# 2. Define the output rectangle
w, h = 400, 500  # Desired output size
dst_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# 3. Compute and apply the transform
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
scanned = cv2.warpPerspective(img, M, (w, h))
```

The result looks like a flat, top-down scan of the document.

## Inverse Perspective Transform

You can also go the other direction -- warp a flat image onto a surface in a photo. Just swap source and destination points:

```python
# Forward: photo -> flat scan
M_forward = cv2.getPerspectiveTransform(photo_pts, flat_pts)

# Inverse: flat image -> overlay on photo
M_inverse = cv2.getPerspectiveTransform(flat_pts, photo_pts)
warped = cv2.warpPerspective(flat_img, M_inverse, (photo_w, photo_h))
```

This is how augmented reality apps overlay content onto surfaces.

## Point Ordering Matters

The four points must be in a consistent order. A common convention:

```python
# Top-left, Top-right, Bottom-left, Bottom-right
pts = np.float32([
    [top_left_x, top_left_y],
    [top_right_x, top_right_y],
    [bottom_left_x, bottom_left_y],
    [bottom_right_x, bottom_right_y]
])
```

If the points are in the wrong order, the image will be flipped, rotated, or severely distorted.

## Tips & Common Mistakes

- Both point arrays must be `np.float32`. Passing integers or `float64` causes errors.
- Exactly **four points** are required -- no more, no less. For more points, use `cv2.findHomography()` which handles overdetermined systems.
- The output size `(width, height)` in `cv2.warpPerspective()` is `(cols, rows)` -- width first, not height. Getting this backwards stretches the image.
- Point ordering must be consistent between source and destination. If point 1 is the top-left in the source, it should map to the top-left in the destination.
- Large perspective changes can produce visible artifacts, especially in text and fine details. The quality depends on the resolution of the source image.
- If the source quadrilateral is very distorted (nearly a triangle), the transform may produce extreme stretching.
- For real applications, automate corner detection using contour finding or feature matching rather than hardcoding coordinates.

## Starter Code

```python
import cv2
import numpy as np

# Create a "document" image with text and a border
doc = np.full((400, 300, 3), (240, 240, 240), dtype=np.uint8)
cv2.rectangle(doc, (10, 10), (289, 389), (0, 0, 0), 2)
cv2.putText(doc, 'DOCUMENT', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
cv2.putText(doc, 'Line 1: Hello', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
cv2.putText(doc, 'Line 2: OpenCV', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
cv2.putText(doc, 'Line 3: Perspective', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
cv2.rectangle(doc, (20, 250), (280, 350), (200, 200, 200), -1)
cv2.putText(doc, '[Signature]', (80, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

# Create a "photo" showing the document at an angle (simulate perspective distortion)
photo = np.full((500, 600, 3), (100, 80, 60), dtype=np.uint8)

# Warp the document to look like it's photographed at an angle
doc_corners = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])
skewed_corners = np.float32([[120, 50], [450, 80], [70, 420], [480, 380]])

M_skew = cv2.getPerspectiveTransform(doc_corners, skewed_corners)
warped_doc = cv2.warpPerspective(doc, M_skew, (600, 500))

# Composite onto the "photo" background
mask = cv2.cvtColor(warped_doc, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
photo_bg = cv2.bitwise_and(photo, photo, mask=mask_inv)
photo = cv2.add(photo_bg, warped_doc)

# Mark the corners on the photo
for pt in skewed_corners:
    cv2.circle(photo, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

# --- Now correct the perspective (this is the main operation) ---
src_pts = skewed_corners
dst_pts = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])

M_correct = cv2.getPerspectiveTransform(src_pts, dst_pts)
corrected = cv2.warpPerspective(photo, M_correct, (300, 400))

# --- Display ---
# Resize photo to match height for side-by-side
photo_display = photo.copy()
cv2.putText(photo_display, 'Skewed Photo', (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Pad corrected image to match photo height
corrected_padded = np.full((500, 300, 3), (50, 50, 50), dtype=np.uint8)
corrected_padded[50:450, :] = corrected
cv2.putText(corrected_padded, 'Corrected', (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Original document padded
doc_padded = np.full((500, 300, 3), (50, 50, 50), dtype=np.uint8)
doc_padded[50:450, :] = doc
cv2.putText(doc_padded, 'Original Doc', (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

result = np.hstack([doc_padded, photo_display, corrected_padded])

print(f'Photo size: {photo.shape}')
print(f'Corrected size: {corrected.shape}')
print(f'Transform matrix shape: {M_correct.shape}')
print(f'Source points:\n{src_pts}')
print(f'Destination points:\n{dst_pts}')

cv2.imshow('Perspective Transform', result)
```
