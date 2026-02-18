---
slug: 48-grabcut-segmentation
title: GrabCut Segmentation
level: intermediate
concepts: [cv2.grabCut, foreground extraction, bounding box init]
prerequisites: [15-creating-masks]
---

## What Problem Are We Solving?

You have a photo and you want to extract the foreground object from the background — like cutting out a person from a group photo or isolating a product from its backdrop. Simple thresholding won't work because the foreground and background have overlapping colors. **GrabCut** is an iterative algorithm that uses a bounding box (or a user-provided mask) to estimate which pixels are foreground and background, refining the segmentation over multiple iterations using graph cuts and Gaussian mixture models.

## How GrabCut Works

At a high level:

1. You provide a **bounding box** around the foreground object (or a rough mask).
2. GrabCut builds Gaussian Mixture Models (GMMs) for both the foreground and background colors.
3. It uses a graph-cut optimization to label each pixel as probable foreground or probable background.
4. The process iterates, refining the GMMs and pixel labels each time.
5. After several iterations, you get a mask indicating foreground and background.

The key advantage over simple thresholding is that GrabCut considers both color distributions and spatial coherence — neighboring pixels tend to belong to the same region.

## Basic Usage: Bounding Box Initialization

```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (x, y, width, height)  # Bounding box around the object

cv2.grabCut(img, mask, rect, bgdModel, fgdModel,
            iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image (8-bit, 3-channel) |
| `mask` | Output/input mask (modified in place) |
| `rect` | Bounding box `(x, y, width, height)` |
| `bgdModel` | Internal model for background (do not modify) |
| `fgdModel` | Internal model for foreground (do not modify) |
| `iterCount` | Number of iterations (5-10 is typical) |
| `mode` | `GC_INIT_WITH_RECT` or `GC_INIT_WITH_MASK` |

The `bgdModel` and `fgdModel` arrays must be `np.float64` with shape `(1, 65)`. They store the internal GMM parameters and are updated in place.

## Understanding the Mask Values

After GrabCut, each pixel in the mask is one of four values:

| Value | Constant | Meaning |
|---|---|---|
| `0` | `cv2.GC_BGD` | Definite background |
| `1` | `cv2.GC_FGD` | Definite foreground |
| `2` | `cv2.GC_PR_BGD` | Probable background |
| `3` | `cv2.GC_PR_FGD` | Probable foreground |

To create a binary foreground mask, keep pixels that are either definite or probable foreground:

```python
# Convert to binary mask: foreground = 1, background = 0
fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

# Apply to original image
result = img * fg_mask[:, :, np.newaxis]
```

## Iterative Refinement with a Mask

You can refine the result by manually marking areas as definite foreground or background, then re-running GrabCut:

```python
# Initial run with bounding box
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Mark a region as definite foreground
mask[100:150, 200:250] = cv2.GC_FGD

# Mark a region as definite background
mask[0:50, 0:50] = cv2.GC_BGD

# Refine with the updated mask
cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
```

When using `GC_INIT_WITH_MASK`, the `rect` parameter is ignored (pass `None`). The algorithm uses the existing mask values as the starting point.

## Initialization with a Mask Only

If you have a rough segmentation (from another method, or hand-drawn), you can skip the bounding box entirely:

```python
mask = np.zeros(img.shape[:2], np.uint8)
mask[:] = cv2.GC_PR_BGD  # Start with everything as probable background

# Mark known foreground region
mask[100:300, 150:400] = cv2.GC_PR_FGD

# Run with mask initialization
cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
```

## Choosing the Bounding Box

The bounding box should:
- Completely contain the foreground object (nothing outside the box is considered foreground)
- Be as tight as possible (less background inside the box means better results)
- Not clip any part of the object (clipped parts will be labeled as background)

```python
# Good: tight box around the object with small margin
rect = (obj_x - 10, obj_y - 10, obj_w + 20, obj_h + 20)

# Bad: box too large (includes too much background)
# Bad: box clips the object (loses foreground pixels)
```

## Tips & Common Mistakes

- The bounding box `rect` is `(x, y, width, height)`, **not** `(x1, y1, x2, y2)`. Width and height, not bottom-right corner.
- Everything **outside** the bounding box is automatically labeled as definite background. If your box is too small, parts of the object will be lost.
- `bgdModel` and `fgdModel` must be `np.zeros((1, 65), np.float64)`. If you use the wrong type or shape, you'll get a cryptic error.
- More iterations improve quality but take longer. 5 iterations is a good starting point; going above 10-15 rarely helps.
- GrabCut works best when the foreground and background have distinct color distributions. It struggles when they look similar.
- The algorithm is slow on large images. Consider resizing the image down, running GrabCut, and then upscaling the mask.
- Always use `np.where` with both `GC_FGD` and `GC_PR_FGD` to create the final mask — using only `GC_FGD` will miss most foreground pixels.
- GrabCut requires an 8-bit, 3-channel input image. Grayscale images must be converted to BGR first.

## Starter Code

```python
import cv2
import numpy as np

# Create a synthetic scene: colored object on a textured background
h, w = 400, 600
img = np.zeros((h, w, 3), dtype=np.uint8)

# Textured background (gradient + noise)
for y in range(h):
    for x in range(w):
        img[y, x] = (80 + x // 8, 100 + y // 6, 60)
bg_noise = np.random.randint(-20, 21, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)

# Draw a foreground object (colorful shape)
# Main body
cv2.ellipse(img, (300, 200), (120, 80), 0, 0, 360, (40, 80, 220), -1)
# Head
cv2.circle(img, (300, 100), 50, (50, 90, 200), -1)
# Arms
cv2.ellipse(img, (170, 190), (30, 60), 30, 0, 360, (45, 85, 210), -1)
cv2.ellipse(img, (430, 190), (30, 60), -30, 0, 360, (45, 85, 210), -1)

# Add noise to make it more realistic
fg_noise = np.random.randint(-10, 11, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + fg_noise, 0, 255).astype(np.uint8)

# --- GrabCut with bounding box ---
mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Bounding box around the foreground object
rect = (130, 40, 340, 260)  # (x, y, width, height)

# Run GrabCut
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create binary foreground mask
fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

# Apply mask to extract foreground
foreground = img * fg_mask[:, :, np.newaxis]

# --- Refinement: mark the head as definite foreground ---
mask_refined = mask.copy()
bgdModel2 = bgdModel.copy()
fgdModel2 = fgdModel.copy()

# Mark head center as definite foreground
cv2.circle(mask_refined, (300, 100), 30, int(cv2.GC_FGD), -1)
# Mark corners as definite background
mask_refined[0:30, 0:30] = cv2.GC_BGD
mask_refined[0:30, w-30:w] = cv2.GC_BGD

cv2.grabCut(img, mask_refined, None, bgdModel2, fgdModel2, 3, cv2.GC_INIT_WITH_MASK)

fg_mask_refined = np.where(
    (mask_refined == cv2.GC_FGD) | (mask_refined == cv2.GC_PR_FGD), 1, 0
).astype(np.uint8)
foreground_refined = img * fg_mask_refined[:, :, np.newaxis]

# --- Build visualization ---
# Panel 1: Original with bounding box
p1 = img.copy()
rx, ry, rw, rh = rect
cv2.rectangle(p1, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
cv2.putText(p1, 'Original + BBox', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Panel 2: Mask visualization
mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
mask_vis[mask == cv2.GC_BGD] = (0, 0, 100)      # Definite BG: dark red
mask_vis[mask == cv2.GC_FGD] = (0, 200, 0)      # Definite FG: green
mask_vis[mask == cv2.GC_PR_BGD] = (100, 50, 50)  # Probable BG: dark blue
mask_vis[mask == cv2.GC_PR_FGD] = (0, 255, 200)  # Probable FG: yellow-green
cv2.putText(mask_vis, 'Mask (4 classes)', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Panel 3: Initial foreground extraction
p3 = foreground.copy()
cv2.putText(p3, 'Initial GrabCut', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Panel 4: Refined foreground extraction
p4 = foreground_refined.copy()
cv2.putText(p4, 'Refined GrabCut', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Print statistics
fg_pixels_initial = np.sum(fg_mask)
fg_pixels_refined = np.sum(fg_mask_refined)
total_pixels = h * w

print(f'Image size: {w}x{h} ({total_pixels} pixels)')
print(f'Bounding box: x={rx}, y={ry}, w={rw}, h={rh}')
print(f'Initial GrabCut foreground: {fg_pixels_initial} pixels '
      f'({100*fg_pixels_initial/total_pixels:.1f}%)')
print(f'Refined GrabCut foreground: {fg_pixels_refined} pixels '
      f'({100*fg_pixels_refined/total_pixels:.1f}%)')
print(f'Mask values: BGD={np.sum(mask==0)}, FGD={np.sum(mask==1)}, '
      f'PR_BGD={np.sum(mask==2)}, PR_FGD={np.sum(mask==3)}')

top_row = np.hstack([p1, mask_vis])
bot_row = np.hstack([p3, p4])
result = np.vstack([top_row, bot_row])

cv2.imshow('GrabCut Segmentation', result)
```
