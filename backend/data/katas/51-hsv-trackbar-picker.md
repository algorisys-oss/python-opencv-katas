---
slug: 51-hsv-trackbar-picker
title: HSV Range Tuning
level: intermediate
concepts: [cv2.inRange, HSV ranges, interactive tuning]
prerequisites: [50-color-object-detection]
---

## What Problem Are We Solving?

When you use `cv2.inRange()` for color detection, the hardest part is figuring out the **right HSV lower and upper bounds** for your target color. Guessing values and re-running the script is tedious. The solution is to build an interactive tool with **trackbars** (sliders) that let you adjust HSV ranges in real-time and immediately see which pixels get selected.

## OpenCV Trackbars

OpenCV provides `cv2.createTrackbar()` to add sliders to a window. Each trackbar has a name, a parent window, a range, and a callback function:

```python
cv2.namedWindow('Controls')
cv2.createTrackbar('H Min', 'Controls', 0, 179, callback)
cv2.createTrackbar('H Max', 'Controls', 179, 179, callback)
```

| Parameter | Meaning |
|---|---|
| `'H Min'` | Label displayed next to the slider |
| `'Controls'` | Name of the window to attach the trackbar to |
| `0` | Initial value of the trackbar |
| `179` | Maximum value (minimum is always 0) |
| `callback` | Function called when the slider moves |

To read the current value of a trackbar:

```python
h_min = cv2.getTrackbarPos('H Min', 'Controls')
```

## Setting Up Six Trackbars for HSV Range

You need six sliders -- minimum and maximum for each of H, S, and V:

```python
def nothing(x):
    pass  # Dummy callback -- we read values in the main loop instead

cv2.namedWindow('Trackbars')
cv2.createTrackbar('H Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nothing)
```

Then in a loop, read all six values and apply `cv2.inRange()`:

```python
h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])
mask = cv2.inRange(hsv, lower, upper)
```

## Common HSV Ranges for Standard Colors

These are practical starting ranges you can dial in with trackbars:

```python
# Red (lower range) -- Hue wraps around!
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])

# Red (upper range)
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])

# Blue
blue_lower = np.array([100, 100, 50])
blue_upper = np.array([130, 255, 255])

# Green
green_lower = np.array([35, 80, 50])
green_upper = np.array([85, 255, 255])

# Yellow
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])
```

## Dealing with Red Hue Wraparound

Red is the trickiest color in HSV because the hue value **wraps around** from 179 back to 0. Pure red sits at H=0 (and H=179), so a continuous range like `[0, 10]` misses the reds near 170-179. The fix is to create two masks and combine them:

```python
mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
red_mask = cv2.bitwise_or(mask1, mask2)
```

When using trackbars, you can handle this by adding a "Red Mode" toggle or simply accepting that red detection needs two passes.

## The Interactive Tuning Loop

The typical pattern for a trackbar-based tuning tool:

```python
while True:
    # Read trackbar positions
    # Build lower/upper arrays
    # Apply cv2.inRange()
    # Show result
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
```

Once you find values that work, note them down and hardcode them into your detection script.

## Tips & Common Mistakes

- Start with full ranges (H: 0-179, S: 0-255, V: 0-255) and narrow down. Starting too narrow means you might miss the target entirely.
- Adjust **Hue first** to isolate the general color, then tighten **Saturation** to remove grayish pixels, and finally adjust **Value** to handle lighting.
- A `nothing` callback is fine -- reading values in the loop is simpler than reacting to every slider event.
- The Hue range in OpenCV is 0-179 (not 0-255). Setting a trackbar max to 255 for Hue is a common bug.
- Print the final HSV values when you press a key so you can copy them into your code.
- For real-world images, HSV ranges are rarely as clean as synthetic examples. Expect to use ranges of 15-30 for Hue and wide Saturation/Value ranges.
- Red detection almost always needs two separate `cv2.inRange()` calls due to the hue wraparound.

## Starter Code

```python
import cv2
import numpy as np

# Create a colorful test image with multiple colored regions
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (180, 180, 180)  # Gray background

# Draw colored shapes
cv2.rectangle(img, (20, 20), (140, 140), (255, 0, 0), -1)       # Blue
cv2.rectangle(img, (160, 20), (280, 140), (0, 180, 0), -1)      # Green
cv2.rectangle(img, (300, 20), (420, 140), (0, 0, 255), -1)      # Red
cv2.rectangle(img, (440, 20), (560, 140), (0, 240, 240), -1)    # Yellow
cv2.circle(img, (80, 250), 60, (200, 100, 50), -1)              # Teal
cv2.circle(img, (240, 250), 60, (50, 50, 220), -1)              # Dark red
cv2.circle(img, (400, 250), 60, (180, 0, 180), -1)              # Purple
cv2.ellipse(img, (300, 360), (120, 30), 0, 0, 360, (0, 165, 255), -1)  # Orange

# Convert to HSV once
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Create trackbar window ---
def nothing(x):
    pass

cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Trackbars', 400, 300)
cv2.createTrackbar('H Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('S Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('V Max', 'Trackbars', 255, 255, nothing)

print('Adjust the trackbars to find HSV ranges for each color.')
print('Press ESC to exit and print the final values.')

while True:
    # Read current trackbar positions
    h_min = cv2.getTrackbarPos('H Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H Max', 'Trackbars')
    s_min = cv2.getTrackbarPos('S Min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S Max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V Min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V Max', 'Trackbars')

    # Build bounds arrays
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask and apply it
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Convert mask to 3-channel for display
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Add text overlays
    info = f'H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]'
    cv2.putText(img, 'Original', (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(mask_display, 'Mask', (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, info, (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Stack: original | mask | filtered result
    display = np.hstack([img, mask_display, result])

    cv2.imshow('HSV Range Tuning', display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print(f'\nFinal HSV Range:')
        print(f'  Lower: [{h_min}, {s_min}, {v_min}]')
        print(f'  Upper: [{h_max}, {s_max}, {v_max}]')
        break

cv2.destroyAllWindows()
```
