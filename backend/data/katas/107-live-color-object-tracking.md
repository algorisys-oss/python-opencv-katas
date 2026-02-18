---
slug: 107-live-color-object-tracking
title: Live Color Object Tracking
level: live
concepts: [cv2.inRange, HSV masking, centroid tracking, cv2.moments, drawing trails]
prerequisites: [100-live-camera-fps, 50-color-object-detection]
---

## What Problem Are We Solving?

Tracking a colored object in real-time is one of the simplest yet most practical computer vision tasks. Think of a robot following a colored ball, a gesture-controlled interface using a colored fingertip, or an interactive art installation tracking a colored wand. The approach is straightforward: convert each frame to HSV color space, create a binary mask for the target color range, clean up the mask, find contours, and compute the centroid of the largest contour.

The key challenge is **robustness**. Raw color thresholding produces noisy masks with holes and speckles. Morphological operations (erosion and dilation) clean this up. Additionally, drawing a trail of past centroid positions creates a visual history of the object's path — useful for gesture recognition or motion analysis.

## Why HSV Instead of BGR?

In BGR color space, color information is entangled with brightness. A green ball in sunlight and the same ball in shadow have very different BGR values. HSV (Hue-Saturation-Value) separates **color** (Hue) from **intensity** (Value), making color-based thresholding much more robust to lighting changes.

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

| Channel | Range | Meaning |
|---|---|---|
| **H** (Hue) | 0-179 | The color angle on the color wheel (OpenCV uses half-degrees) |
| **S** (Saturation) | 0-255 | Color purity — 0 is gray, 255 is fully saturated |
| **V** (Value) | 0-255 | Brightness — 0 is black, 255 is brightest |

> **Important:** OpenCV uses a Hue range of 0-179 (not 0-360). This is a common source of bugs when converting HSV values from other tools or references.

## Color Thresholding with cv2.inRange

`cv2.inRange` creates a binary mask where pixels within the specified range are white (255) and all others are black (0):

```python
mask = cv2.inRange(hsv, lower_bound, upper_bound)
```

| Parameter | Type | Meaning |
|---|---|---|
| `hsv` | ndarray | Input HSV image |
| `lower_bound` | ndarray or tuple | Lower (H, S, V) threshold |
| `upper_bound` | ndarray or tuple | Upper (H, S, V) threshold |
| **Returns** | ndarray | Binary mask (uint8, same size as input) |

### Common Color Ranges

| Color | Lower HSV | Upper HSV | Notes |
|---|---|---|---|
| Blue | (100, 120, 70) | (130, 255, 255) | Adjust saturation for pale vs vivid blue |
| Green | (35, 100, 70) | (85, 255, 255) | Wide hue range; narrow if too many false positives |
| Red (low) | (0, 120, 70) | (10, 255, 255) | Red wraps around hue=0, so you need two ranges |
| Red (high) | (170, 120, 70) | (180, 255, 255) | Combine with low range using bitwise OR |
| Yellow | (20, 100, 100) | (35, 255, 255) | Narrow hue range |
| Orange | (10, 100, 100) | (20, 255, 255) | Between red and yellow |

For red (which wraps around hue 0/180):

```python
mask_low = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
mask_high = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
mask = cv2.bitwise_or(mask_low, mask_high)
```

## Morphological Cleanup

Raw masks from `cv2.inRange` contain noise — small speckles from sensor noise and holes inside the object. Morphological operations fix this:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Erode: remove small white speckles (noise)
mask = cv2.erode(mask, kernel, iterations=2)

# Dilate: fill small holes and restore object size
mask = cv2.dilate(mask, kernel, iterations=2)
```

| Operation | Effect | Use Case |
|---|---|---|
| `cv2.erode` | Shrinks white regions, removes small blobs | Eliminate speckle noise |
| `cv2.dilate` | Expands white regions, fills small holes | Restore object after erosion |
| Erode then Dilate ("opening") | Removes noise without changing object size | Standard cleanup sequence |
| Dilate then Erode ("closing") | Fills holes without changing object size | When object has internal gaps |

An elliptical kernel produces smoother results than a rectangular one for round objects.

## Finding the Object Centroid

After cleaning the mask, find contours and pick the largest one (which is most likely the target object):

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)

    # Only process if area is above a minimum threshold
    if cv2.contourArea(largest) > 500:
        # Compute centroid using image moments
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
```

| Moment | Formula | Meaning |
|---|---|---|
| `m00` | Sum of all mask pixels | Area of the contour |
| `m10` | Sum of x-coordinates | X-weighted area |
| `m01` | Sum of y-coordinates | Y-weighted area |
| `m10/m00` | — | X-coordinate of centroid |
| `m01/m00` | — | Y-coordinate of centroid |

The minimum area threshold (e.g., 500 pixels) prevents the tracker from locking onto small noise blobs.

## Drawing a Trail

Store past centroid positions in a deque and draw lines between consecutive points:

```python
from collections import deque

trail = deque(maxlen=64)  # Keep last 64 positions

# After computing centroid:
trail.appendleft((cx, cy))

# Draw trail with fading thickness
for i in range(1, len(trail)):
    if trail[i - 1] is None or trail[i] is None:
        continue
    thickness = max(1, int((len(trail) - i) / len(trail) * 5))
    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), thickness)
```

The trail fades from thick (recent) to thin (old), giving a visual sense of direction and speed.

## Keyboard HSV Range Tuning

Since lighting conditions vary, allow runtime adjustment of HSV bounds with keyboard controls:

```python
key = cv2.waitKey(1) & 0xFF

if key == ord('1'):    # Hue lower +
    h_low = min(h_low + 2, 179)
elif key == ord('2'):  # Hue lower -
    h_low = max(h_low - 2, 0)
elif key == ord('3'):  # Hue upper +
    h_high = min(h_high + 2, 179)
elif key == ord('4'):  # Hue upper -
    h_high = max(h_high - 2, 0)
elif key == ord('5'):  # Sat lower +
    s_low = min(s_low + 5, 255)
elif key == ord('6'):  # Sat lower -
    s_low = max(s_low - 5, 0)
```

Display the current HSV range on the frame so you can see what values you are using and tune in real-time.

## Tips & Common Mistakes

- **Use a color picker tool first.** Point the camera at your target object, convert a sample pixel to HSV, and build your range from there. Do not guess HSV values — they are unintuitive.
- If your target color is not detected, widen the Saturation and Value ranges first, then adjust Hue. Low saturation or value thresholds cause most missed detections.
- Red requires two `inRange` calls because hue wraps around at 0/180. Forgetting this means you only detect half the red spectrum.
- Erode before dilate (opening) for noise removal. If you dilate first, you amplify the noise.
- The minimum contour area threshold is critical. Without it, single-pixel noise clusters will cause the centroid to jump erratically.
- If multiple objects of the same color exist, `max(contours, key=cv2.contourArea)` always picks the largest. For multi-object tracking, iterate over all contours above the area threshold.
- The trail deque `maxlen` controls how long the trail persists. Increase it for a longer trail, decrease for a shorter one.
- A GaussianBlur before HSV conversion reduces noise and makes the mask cleaner: `frame = cv2.GaussianBlur(frame, (11, 11), 0)`.
- Webcam auto-exposure and auto-white-balance can shift colors between frames. Consider disabling auto settings for more stable detection.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Hold a brightly colored object (e.g., a green ball or marker) in front of the camera — you should see a contour outline, bounding circle, and red centroid dot appear on it
- Move the object around and confirm a fading yellow trail follows its path
- Use the keyboard keys **1-8** to adjust HSV thresholds if the object is not detected, and press **'r'** to reset to defaults
- Check the mask preview in the top-left corner — the tracked object should appear as a clean white blob on black background

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)

# --- HSV range for bright green (adjust for your object!) ---
h_low, h_high = 40, 80
s_low, s_high = 100, 255
v_low, v_high = 100, 255

# --- Morphological kernel ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# --- Trail storage ---
trail = deque(maxlen=64)
MIN_CONTOUR_AREA = 500

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Blur to reduce noise ---
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # --- Convert to HSV and threshold ---
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower, upper)

        # --- Morphological cleanup ---
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # --- Find contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroid = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                # Draw contour outline
                cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)

                # Bounding circle
                ((bx, by), radius) = cv2.minEnclosingCircle(largest)
                cv2.circle(frame, (int(bx), int(by)), int(radius), (255, 0, 0), 2)

                # Centroid from moments
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = (cx, cy)
                    cv2.circle(frame, centroid, 7, (0, 0, 255), -1)

        # --- Update trail ---
        trail.appendleft(centroid)

        # --- Draw trail with fading thickness ---
        for i in range(1, len(trail)):
            if trail[i - 1] is None or trail[i] is None:
                continue
            thickness = max(1, int((len(trail) - i) / len(trail) * 5))
            alpha = 1.0 - (i / len(trail))
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.line(frame, trail[i - 1], trail[i], color, thickness)

        # --- Show mask preview (small, in corner) ---
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (160, 120))
        frame[0:120, 0:160] = mask_small

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"HSV: H[{h_low}-{h_high}] S[{s_low}-{s_high}] V[{v_low}-{v_high}]",
                    (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        status = "Tracking" if centroid else "Lost"
        status_color = (0, 255, 0) if centroid else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (170, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        if centroid:
            cv2.putText(frame, f"Position: {centroid}", (170, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "1/2:H- 3/4:H+ 5/6:S 7/8:V  r:reset  q:quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Color Object Tracking', frame)

        # --- Keyboard controls for HSV tuning ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            h_low = max(h_low - 2, 0)
        elif key == ord('2'):
            h_low = min(h_low + 2, h_high - 1)
        elif key == ord('3'):
            h_high = max(h_high - 2, h_low + 1)
        elif key == ord('4'):
            h_high = min(h_high + 2, 179)
        elif key == ord('5'):
            s_low = max(s_low - 5, 0)
        elif key == ord('6'):
            s_low = min(s_low + 5, 255)
        elif key == ord('7'):
            v_low = max(v_low - 5, 0)
        elif key == ord('8'):
            v_low = min(v_low + 5, 255)
        elif key == ord('r'):
            h_low, h_high = 40, 80
            s_low, s_high = 100, 255
            v_low, v_high = 100, 255
            trail.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()
```
