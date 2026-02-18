---
slug: 50-color-object-detection
title: Color-Based Object Detection
level: intermediate
concepts: [cv2.inRange, HSV masking, color filtering]
prerequisites: [02-color-spaces, 14-bitwise-operations]
---

## What Problem Are We Solving?

You want to isolate objects of a specific color from an image -- for example, picking out all the red apples in a photo, or tracking a blue ball in a video frame. Working in RGB makes this surprisingly hard because lighting changes affect all three channels unpredictably. The solution is to convert to **HSV color space** and use `cv2.inRange()` to create a binary mask that captures only the pixels within your target color range.

## Why HSV Instead of BGR?

In BGR, a "red" pixel might be `(0, 0, 200)` under bright light but `(0, 0, 80)` in shadow. You'd need to define ranges across all three channels simultaneously, and those ranges shift with lighting. HSV separates **color identity** (Hue) from **brightness** (Value) and **purity** (Saturation):

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

| Channel | Range | Meaning |
|---|---|---|
| H (Hue) | 0-179 | The color itself (red, green, blue, etc.) |
| S (Saturation) | 0-255 | How pure/vivid the color is (0 = gray, 255 = fully saturated) |
| V (Value) | 0-255 | How bright the pixel is (0 = black, 255 = bright) |

Because Hue is separated from brightness, you can detect "blue" objects whether they're in sunlight or shade -- only the V channel changes significantly.

## Creating a Color Mask with cv2.inRange()

`cv2.inRange()` checks each pixel against a lower and upper bound. Pixels within the range become white (255); everything else becomes black (0):

```python
lower = np.array([100, 50, 50])    # Lower bound: H=100, S=50, V=50
upper = np.array([130, 255, 255])  # Upper bound: H=130, S=255, V=255
mask = cv2.inRange(hsv, lower, upper)
```

The result is a single-channel binary image (the mask) where white pixels represent your target color.

| Parameter | Meaning |
|---|---|
| `hsv` | Input image in HSV color space |
| `lower` | Numpy array with minimum H, S, V values |
| `upper` | Numpy array with maximum H, S, V values |

## Applying the Mask

Once you have the mask, use `cv2.bitwise_and()` to extract only the colored regions from the original image:

```python
result = cv2.bitwise_and(img, img, mask=mask)
```

This keeps the original pixel colors wherever the mask is white, and sets everything else to black. The `mask` parameter tells OpenCV which pixels to keep.

## Common HSV Ranges for Standard Colors

Here are approximate HSV ranges for common colors in OpenCV (where H is 0-179):

```python
# Blue objects
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Green objects
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Yellow objects
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
```

These are starting points -- you'll need to tune them for your specific images and lighting conditions.

## Cleaning Up the Mask

Raw masks from `cv2.inRange()` often have noise -- small specks of white in the background or holes inside the detected object. Morphological operations clean this up:

```python
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
```

## Tips & Common Mistakes

- Always convert to HSV **before** calling `cv2.inRange()`. Applying it to BGR images gives meaningless results.
- OpenCV uses H: 0-179, not 0-360 like some other tools. Divide standard hue values by 2 when converting.
- Saturation and Value lower bounds matter -- setting S and V minimums to 50 or higher filters out near-white and near-black pixels that aren't clearly colored.
- Red is tricky because it wraps around the hue circle (0 and 179 are both red). You need two ranges and combine them with `cv2.bitwise_or()`.
- `cv2.inRange()` returns a `uint8` mask -- it works directly with `cv2.bitwise_and()` without conversion.
- If your detection misses objects or picks up background, adjust the S and V ranges first, then fine-tune H.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful test image with distinct colored shapes
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (200, 200, 200)  # Light gray background

# Draw colored shapes: blue rectangle, green circle, red triangle, yellow ellipse
cv2.rectangle(img, (30, 50), (160, 180), (255, 50, 50), -1)      # Blue
cv2.circle(img, (300, 120), 70, (50, 200, 50), -1)                # Green
pts = np.array([[480, 50], [420, 180], [540, 180]], np.int32)
cv2.fillPoly(img, [pts], (50, 50, 255))                           # Red
cv2.ellipse(img, (120, 300), (80, 40), 0, 0, 360, (0, 230, 230), -1)  # Yellow

# --- Convert to HSV ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Detect blue objects ---
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
result_blue = cv2.bitwise_and(img, img, mask=mask_blue)

# --- Detect green objects ---
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
result_green = cv2.bitwise_and(img, img, mask=mask_green)

# --- Detect red objects ---
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)
result_red = cv2.bitwise_and(img, img, mask=mask_red)

# --- Detect yellow objects ---
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
result_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Original', (10, 25), font, 0.6, (0, 0, 0), 2)
cv2.putText(result_blue, 'Blue Only', (10, 25), font, 0.6, (255, 255, 255), 2)
cv2.putText(result_green, 'Green Only', (10, 25), font, 0.6, (255, 255, 255), 2)
cv2.putText(result_red, 'Red Only', (10, 25), font, 0.6, (255, 255, 255), 2)

# Build grid: top row = original + blue, bottom row = green + red
top_half = np.hstack([img, result_blue])
# Resize yellow result to show in a label
cv2.putText(result_yellow, 'Yellow Only', (10, 25), font, 0.6, (255, 255, 255), 2)
bottom_half = np.hstack([result_green, result_red])
result = np.vstack([top_half, bottom_half])

print(f'Image shape: {img.shape}')
print(f'Blue mask unique values: {np.unique(mask_blue)}')
print(f'Blue pixels detected: {cv2.countNonZero(mask_blue)}')
print(f'Green pixels detected: {cv2.countNonZero(mask_green)}')
print(f'Red pixels detected: {cv2.countNonZero(mask_red)}')

cv2.imshow('Color-Based Object Detection', result)
```
