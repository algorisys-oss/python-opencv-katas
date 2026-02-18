---
slug: 56-image-inpainting
title: Image Inpainting
level: intermediate
concepts: [cv2.inpaint, INPAINT_TELEA, INPAINT_NS]
prerequisites: [15-creating-masks]
---

## What Problem Are We Solving?

You have an image with unwanted elements -- text overlays, scratches, watermarks, small objects that need to be removed. **Inpainting** fills in damaged or selected regions by borrowing information from surrounding pixels, producing a result that looks like the unwanted element was never there. OpenCV provides two algorithms for this: Telea's method and Navier-Stokes-based inpainting.

## What is Image Inpainting?

Inpainting is the process of reconstructing lost or corrupted parts of an image. The algorithm looks at the pixels surrounding the damaged area and fills in the gap with plausible pixel values. Think of it as an intelligent "content-aware fill" -- it doesn't just smear nearby colors but tries to continue edges, textures, and gradients seamlessly.

## Creating Inpaint Masks

The inpaint mask tells OpenCV **which pixels to reconstruct**. It's a single-channel image where:
- **White pixels (255)** = regions to inpaint (damaged/unwanted areas).
- **Black pixels (0)** = regions to keep unchanged.

```python
# Create a mask marking the area to repair
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (100, 50), (300, 80), 255, -1)  # Mark a rectangular region
```

For removing text, you might create the mask by thresholding the text color:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
```

## Using cv2.inpaint()

The function takes the image, the mask, a radius, and the algorithm:

```python
result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image (BGR or grayscale) |
| `mask` | Binary mask: 255 = pixels to inpaint, 0 = keep |
| `inpaintRadius` | Radius of neighborhood to consider around each inpainted pixel |
| `flags` | Algorithm: `cv2.INPAINT_TELEA` or `cv2.INPAINT_NS` |

## Telea vs Navier-Stokes Methods

OpenCV offers two inpainting algorithms:

**cv2.INPAINT_TELEA** (Telea, 2004):
```python
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
```
- Based on the Fast Marching Method.
- Fills from the boundary inward, weighting nearby pixels more heavily.
- Generally produces smoother results.
- Better for larger damaged areas.

**cv2.INPAINT_NS** (Navier-Stokes, 2001):
```python
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
```
- Based on fluid dynamics equations.
- Tries to continue edges and isophote lines (lines of equal intensity) into the damaged region.
- Better at preserving sharp edges.
- Can produce artifacts in large uniform areas.

In practice, try both and see which looks better for your specific case.

## The Inpaint Radius Parameter

The `inpaintRadius` controls how far the algorithm looks for information:

```python
# Small radius: fast, uses only immediate neighbors
result_small = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)

# Medium radius: good balance
result_medium = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# Large radius: considers wider context, slower
result_large = cv2.inpaint(img, mask, 10, cv2.INPAINT_TELEA)
```

- Too small: the fill may not have enough context and looks patchy.
- Too large: slows computation and may pull in irrelevant colors from far away.
- A radius of 3-5 is a good starting point for most cases.

## Practical Use: Removing Text from Images

A common workflow for removing overlaid text:

```python
# 1. Create mask where the text is
# (often text is the brightest or darkest element)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, text_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

# 2. Dilate the mask slightly to cover anti-aliased edges
kernel = np.ones((3, 3), np.uint8)
text_mask = cv2.dilate(text_mask, kernel, iterations=1)

# 3. Inpaint
cleaned = cv2.inpaint(img, text_mask, 3, cv2.INPAINT_TELEA)
```

## Tips & Common Mistakes

- The mask must be **single-channel** and **uint8**, with only values 0 and 255. Passing a 3-channel mask or float values will fail or produce wrong results.
- The inpaint radius should roughly match the size of the features you're removing. For thin scratches, radius 2-3 is enough. For thick text, try 5-7.
- Inpainting works best for **small or narrow** damaged regions. Large missing areas (like removing a person from a photo) will look blurry or smeared -- that requires more advanced techniques.
- Dilating the mask by 1-2 pixels helps cover anti-aliased edges of text or soft borders of objects being removed.
- The image and mask must have the **same height and width**. A size mismatch causes an error.
- For color images, inpainting is performed on all three channels simultaneously -- you don't need to process channels separately.
- Compare both TELEA and NS results before choosing. The "better" method depends on the specific image content.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with a nice scene
img = np.zeros((350, 500, 3), dtype=np.uint8)

# Sky gradient (top portion)
for y in range(180):
    blue = int(200 - y * 0.5)
    img[y, :] = (blue, int(150 + y * 0.3), int(80 + y * 0.5))

# Ground (bottom portion)
for y in range(180, 350):
    img[y, :] = (50, int(100 + (y - 180) * 0.3), 60)

# Sun
cv2.circle(img, (400, 60), 35, (0, 200, 255), -1)

# Mountains
pts = np.array([[0, 180], [100, 100], [200, 160], [300, 80], [400, 140], [500, 180]], np.int32)
cv2.fillPoly(img, [pts], (100, 100, 120))

# Tree
cv2.rectangle(img, (70, 200), (90, 300), (30, 60, 40), -1)    # Trunk
cv2.circle(img, (80, 180), 40, (20, 120, 30), -1)              # Canopy

# --- Add text overlay that we want to remove ---
damaged = img.copy()
cv2.putText(damaged, 'SAMPLE TEXT', (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.putText(damaged, 'watermark', (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

# --- Create inpaint mask (detect the white/bright text) ---
gray = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Mask is where damaged differs significantly from the original
diff = cv2.absdiff(gray, gray_orig)
_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Dilate mask to cover anti-aliased text edges
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

# --- Inpaint using both methods ---
result_telea = cv2.inpaint(damaged, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
result_ns = cv2.inpaint(damaged, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

# --- Different inpaint radii comparison ---
result_r1 = cv2.inpaint(damaged, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
result_r10 = cv2.inpaint(damaged, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

cv2.putText(damaged, 'Damaged', (5, 345), font, 0.5, (255, 255, 255), 1)
cv2.putText(mask_display, 'Mask', (5, 345), font, 0.5, (255, 255, 255), 1)
cv2.putText(result_telea, 'TELEA r=5', (5, 345), font, 0.5, (255, 255, 255), 1)
cv2.putText(result_ns, 'NS r=5', (5, 345), font, 0.5, (255, 255, 255), 1)

# Build grid
top_row = np.hstack([damaged, mask_display])
bottom_row = np.hstack([result_telea, result_ns])
display = np.vstack([top_row, bottom_row])

print(f'Image shape: {img.shape}')
print(f'Mask non-zero pixels: {cv2.countNonZero(mask)}')
print(f'Mask coverage: {cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1]) * 100:.1f}%')

cv2.imshow('Image Inpainting', display)
```
