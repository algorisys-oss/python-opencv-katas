---
slug: 02-color-spaces
title: Color Spaces
level: beginner
concepts: [cv2.cvtColor, BGR, RGB, Grayscale, HSV]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

OpenCV stores images in **BGR** by default, but the world uses many different ways to represent color. A **color space** is a mathematical model for describing colors.

## Why Do Color Spaces Matter?

Different tasks need different color representations:

| Color Space | Best For |
|---|---|
| **BGR/RGB** | Natural display, what humans see |
| **Grayscale** | Removes color, keeps brightness — faster processing |
| **HSV** | Color-based detection — separates hue from brightness |

## HSV Explained Simply

HSV stands for **Hue, Saturation, Value**:

- **Hue** (0–179): The color itself. Red≈0, Green≈60, Blue≈120.
- **Saturation** (0–255): How vivid the color is. 0=gray, 255=pure color.
- **Value** (0–255): Brightness. 0=black, 255=full brightness.

> **Why use HSV for detection?** You can say "find pixels where hue is between 0–10" to find red objects, regardless of lighting conditions. This is impossible with BGR.

## Converting Between Color Spaces

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

`cv2.cvtColor()` always returns a **new array** — it never modifies the original.

## Tips & Common Mistakes

- HSV hue range in OpenCV is **0–179** (not 0–360 like design tools). Divide by 2.
- Grayscale images have shape `(h, w)` not `(h, w, 1)` — watch out when doing math.
- To display OpenCV images in matplotlib: convert BGR→RGB first.
- `cv2.inRange()` + HSV is the classic recipe for color-based object detection.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful test image with distinct regions
img = np.zeros((200, 400, 3), dtype=np.uint8)

# Draw colored rectangles (BGR order)
img[0:200, 0:100]   = (255, 0, 0)    # Blue region
img[0:200, 100:200] = (0, 255, 0)    # Green region
img[0:200, 200:300] = (0, 0, 255)    # Red region
img[0:200, 300:400] = (0, 255, 255)  # Yellow region

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Grayscale shape: {gray.shape}')  # (200, 400) — no channel dim!

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(f'HSV shape: {hsv.shape}')         # (200, 400, 3)

# Sample HSV values at each color region
print(f'Blue  HSV: {hsv[100, 50]}')
print(f'Green HSV: {hsv[100, 150]}')
print(f'Red   HSV: {hsv[100, 250]}')

# Stack original + grayscale (converted back to BGR) + HSV side by side
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
result = np.hstack([img, gray_bgr, hsv])
cv2.imshow('BGR | Grayscale | HSV', result)
```
