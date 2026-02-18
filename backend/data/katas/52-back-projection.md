---
slug: 52-back-projection
title: Back Projection
level: intermediate
concepts: [cv2.calcBackProject, histogram-based detection]
prerequisites: [18-understanding-histograms, 50-color-object-detection]
---

## What Problem Are We Solving?

You have a sample patch of the object you want to find -- maybe a small crop of skin, a piece of fabric, or a section of sky -- and you want to locate **where similar colors appear** in a larger image. Unlike `cv2.inRange()`, which requires you to manually specify color bounds, **histogram back projection** learns the color distribution from your sample and scores every pixel in the target image by how well it matches that distribution. Pixels that match the sample's colors get high values; everything else gets low values.

## What is Histogram Back Projection?

The idea is straightforward:

1. Compute the **color histogram** of your sample region (the "model").
2. For each pixel in the target image, look up its color in the model histogram.
3. The histogram bin value becomes the pixel's score -- high counts mean "this color appears often in the sample."

The result is a probability map showing where the sample's colors appear in the full image.

## Computing the Model Histogram

First, compute a histogram from your sample region in HSV space. Using Hue and Saturation channels (ignoring Value) makes the detection more robust to lighting changes:

```python
hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample_hist = cv2.calcHist([hsv_sample], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(sample_hist, sample_hist, 0, 255, cv2.NORM_MINMAX)
```

| Parameter | Meaning |
|---|---|
| `[hsv_sample]` | The sample image (in a list) |
| `[0, 1]` | Channels to use: H and S |
| `None` | No mask -- use the entire sample |
| `[180, 256]` | Number of bins for H and S |
| `[0, 180, 0, 256]` | Value ranges for H (0-180) and S (0-256) |

Normalizing the histogram to 0-255 ensures the back projection output is a proper grayscale image.

## Applying cv2.calcBackProject()

Now use the model histogram to score every pixel in the target image:

```python
hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
back_proj = cv2.calcBackProject([hsv_target], [0, 1], sample_hist, [0, 180, 0, 256], 1)
```

| Parameter | Meaning |
|---|---|
| `[hsv_target]` | Target image to search in |
| `[0, 1]` | Channels (must match the histogram channels) |
| `sample_hist` | The model histogram from the sample |
| `[0, 180, 0, 256]` | Value ranges (must match the histogram ranges) |
| `1` | Scale factor (1 = no scaling) |

The output `back_proj` is a grayscale image the same size as the target, where bright pixels indicate strong matches.

## Improving Results with Filtering

The raw back projection is often noisy. A disc-shaped filter smooths it out and connects nearby high-probability regions:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(back_proj, -1, kernel, back_proj)
```

You can also threshold the result to get a clean binary mask:

```python
_, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
```

## Using Back Projection for Object Localization

Combine the mask with the original image to highlight detected regions:

```python
mask_3ch = cv2.merge([mask, mask, mask])
result = cv2.bitwise_and(target, mask_3ch)
```

For tracking applications, you can find the centroid of the bright region in the back projection to locate the object.

## Tips & Common Mistakes

- Use **HSV** (Hue + Saturation) rather than BGR for back projection. It's far more robust to lighting changes.
- Always **normalize** the model histogram before passing it to `cv2.calcBackProject()`. Without normalization, the output values may be too small to see.
- The channel indices and ranges in `cv2.calcBackProject()` must **exactly match** those used in `cv2.calcHist()`. Mismatches cause silent wrong results.
- A larger sample region gives a more representative histogram. Tiny samples may not capture enough color variation.
- The scale parameter (last argument) is almost always `1`. Other values scale the output, which is rarely needed.
- Back projection works best for objects with distinctive colors. Objects that share colors with the background will produce noisy results.
- Morphological operations (opening, closing) can clean up the back projection mask significantly.

## Starter Code

```python
import cv2
import numpy as np

# Create a target image with multiple colored regions
target = np.zeros((400, 600, 3), dtype=np.uint8)
target[:] = (220, 220, 220)  # Light gray background

# Draw various colored shapes
cv2.rectangle(target, (20, 20), (150, 150), (0, 100, 200), -1)    # Orange-ish
cv2.rectangle(target, (400, 200), (580, 380), (0, 120, 210), -1)  # Similar orange
cv2.circle(target, (300, 100), 70, (200, 50, 50), -1)             # Blue
cv2.circle(target, (150, 300), 60, (50, 180, 50), -1)             # Green
cv2.ellipse(target, (450, 80), (80, 40), 0, 0, 360, (0, 90, 190), -1)  # Another orange-ish

# Define a sample region (crop from the first orange rectangle)
sample = target[40:130, 40:130].copy()

# Convert both to HSV
hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

# --- Compute the model histogram from the sample ---
sample_hist = cv2.calcHist([hsv_sample], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(sample_hist, sample_hist, 0, 255, cv2.NORM_MINMAX)

# --- Back project onto the target image ---
back_proj = cv2.calcBackProject([hsv_target], [0, 1], sample_hist, [0, 180, 0, 256], 1)

# --- Smooth with a disc kernel to reduce noise ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cv2.filter2D(back_proj, -1, kernel, back_proj)

# --- Threshold to create a clean mask ---
_, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

# Apply mask to the original image
mask_3ch = cv2.merge([mask, mask, mask])
detected = cv2.bitwise_and(target, mask_3ch)

# Draw a rectangle around the sample region on the target for reference
target_display = target.copy()
cv2.rectangle(target_display, (40, 40), (130, 130), (0, 255, 0), 2)
cv2.putText(target_display, 'Sample', (42, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Resize sample for display
sample_display = cv2.resize(sample, (150, 150))

# Convert back projection to color for display
back_proj_color = cv2.cvtColor(back_proj, cv2.COLOR_GRAY2BGR)
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Add labels
cv2.putText(target_display, 'Target', (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.putText(back_proj_color, 'Back Projection', (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(detected, 'Detected Regions', (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Stack results
result = np.hstack([target_display, back_proj_color, detected])

print(f'Target shape: {target.shape}')
print(f'Sample shape: {sample.shape}')
print(f'Back projection shape: {back_proj.shape}')
print(f'Back projection max value: {back_proj.max()}')
print(f'Detected pixels: {cv2.countNonZero(mask)}')

cv2.imshow('Back Projection', result)
```
