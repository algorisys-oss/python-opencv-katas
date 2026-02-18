---
slug: 123-live-motion-heatmap
title: Live Motion Heatmap
level: live
concepts: [frame accumulation, cv2.applyColorMap, heatmap overlay, decay factor, cv2.addWeighted]
prerequisites: [100-live-camera-fps, 73-frame-differencing, 94-motion-heatmap]
---

## What Problem Are We Solving?

In kata 94 you built a motion heatmap from a pre-recorded sequence of frames. That approach accumulates all motion at once and produces a single summary image. But what if you want to watch activity patterns **as they develop** -- seeing where motion is happening right now, where it happened recently, and having old activity gradually fade away?

A **live motion heatmap** does exactly that. It runs in real-time on your camera feed, accumulating frame-to-frame differences into a floating-point buffer, applying a **decay factor** each frame so that old motion fades out, normalizing the result to 0-255, applying a colormap like `COLORMAP_JET`, and blending it with the original camera image. The result is a colorful overlay where hot zones (red/yellow) show current or recent activity and cool zones (blue) show areas where motion happened a while ago.

This technique is used in retail analytics (tracking where customers walk), sports analysis (visualizing player movement patterns), security systems (identifying high-traffic zones), and interactive art installations.

## The Live Heatmap Pipeline

Each frame, the pipeline performs these steps in order:

1. **Capture** the current frame and convert to grayscale
2. **Difference** the current frame against the previous frame using `cv2.absdiff`
3. **Threshold** the difference to remove sensor noise
4. **Accumulate** the thresholded result into a `float32` buffer
5. **Decay** the entire accumulator by multiplying with a factor (e.g., 0.95)
6. **Normalize** the accumulator to the 0-255 range
7. **Colormap** the normalized result with `cv2.applyColorMap`
8. **Blend** the colormap with the original frame using `cv2.addWeighted`

## Understanding the Decay Factor

Without decay, the accumulator grows forever -- every area that ever had motion stays permanently hot. The decay factor controls how quickly old motion fades:

```python
accumulator *= decay_factor  # Apply decay BEFORE adding new motion
accumulator += thresh.astype(np.float32)
```

| Decay Factor | Behavior | Use Case |
|---|---|---|
| 0.99 | Very slow fade, long memory | Tracking cumulative paths over minutes |
| 0.95 | Moderate fade, ~1-2 second memory | General-purpose motion heatmap |
| 0.90 | Fast fade, sub-second memory | Showing only very recent motion |
| 0.80 | Very fast fade, nearly instantaneous | Real-time motion highlighting |
| 1.00 | No decay at all | Permanent accumulation (like kata 94) |

The decay is exponential: after `N` frames, old motion is scaled by `decay^N`. At 30 FPS with decay 0.95, motion from 1 second ago is reduced to `0.95^30 = 0.21` (21% of original). After 2 seconds: `0.95^60 = 0.046` (4.6%).

## Colormaps for Heatmap Visualization

`cv2.applyColorMap` converts a single-channel `uint8` image into a 3-channel BGR color image using a predefined lookup table:

```python
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
```

| Parameter | Type | Description |
|---|---|---|
| `src` | `np.ndarray` | Single-channel `uint8` input image |
| `colormap` | `int` | Colormap constant (e.g., `cv2.COLORMAP_JET`) |
| **Returns** | `np.ndarray` | 3-channel BGR image of the same size |

Popular colormaps for heatmaps:

| Colormap | Visual Range | Best For |
|---|---|---|
| `COLORMAP_JET` | Blue -> Cyan -> Yellow -> Red | Classic heatmap, high contrast |
| `COLORMAP_HOT` | Black -> Red -> Yellow -> White | Thermal camera style |
| `COLORMAP_INFERNO` | Black -> Purple -> Orange -> Yellow | Perceptually uniform, publication quality |
| `COLORMAP_TURBO` | Blue -> Cyan -> Yellow -> Red | Improved JET with better perceptual uniformity |
| `COLORMAP_MAGMA` | Black -> Purple -> Orange -> Yellow | Dark backgrounds, subtle data |

## Blending with cv2.addWeighted

To overlay the heatmap on the original camera image:

```python
result = cv2.addWeighted(frame, alpha, heatmap_color, beta, gamma)
```

| Parameter | Type | Description |
|---|---|---|
| `src1` | `np.ndarray` | First image (the camera frame) |
| `alpha` | `float` | Weight of the first image (0.0 to 1.0) |
| `src2` | `np.ndarray` | Second image (the colored heatmap) |
| `beta` | `float` | Weight of the second image (0.0 to 1.0) |
| `gamma` | `float` | Scalar added to the result (usually 0) |

Good blending ratios:

| Alpha (Frame) | Beta (Heatmap) | Effect |
|---|---|---|
| 0.7 | 0.3 | Subtle heatmap overlay, camera dominant |
| 0.5 | 0.5 | Equal blend, both clearly visible |
| 0.3 | 0.7 | Heatmap dominant, camera as context |

## Reset Capability

Over time, even with decay, the heatmap accumulates residual values. A manual reset (pressing a key) clears the accumulator to start fresh:

```python
if key == ord('r'):
    accumulator[:] = 0  # Clear the entire heatmap
```

This is useful when you reposition the camera, change the scene, or want to start a new observation period.

## Tips & Common Mistakes

- Use `float32` for the accumulator, not `uint8`. With `uint8`, values overflow at 255 and wrap around, producing nonsensical heatmaps.
- Apply decay **before** adding new motion each frame. If you decay after adding, you immediately reduce the motion you just detected.
- The threshold value for frame differencing (e.g., 25) controls sensitivity. Lower values detect subtle motion but also amplify noise. Adjust based on your camera's sensor quality.
- `cv2.applyColorMap` expects single-channel `uint8` input. If you pass a float or multi-channel image, the output will be incorrect with no error message.
- Apply `cv2.GaussianBlur` to the grayscale frames before differencing to suppress sensor noise that would otherwise create a noisy heatmap.
- If the heatmap looks pixelated or blocky, increase the Gaussian blur kernel size or dilate the thresholded difference before accumulating.
- Normalize using the accumulator's actual max value, not a fixed value. A fixed normalization constant either clips hot zones or makes cold zones invisible.

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

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# --- Heatmap state ---
accumulator = np.zeros((height, width), dtype=np.float32)
prev_gray = None
decay_factor = 0.95       # How fast old motion fades (0.0-1.0)
threshold_val = 25         # Minimum pixel change to count as motion
colormap = cv2.COLORMAP_JET
blend_alpha = 0.6          # Camera frame weight
blend_beta = 0.4           # Heatmap weight

# --- FPS tracking ---
frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame_times.append(time.time())
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps = 0.0

        # --- Convert to grayscale and blur ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            # --- Compute frame difference ---
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

            # Dilate to fill small gaps in motion regions
            thresh = cv2.dilate(thresh, None, iterations=2)

            # --- Decay old motion, then accumulate new ---
            accumulator *= decay_factor
            accumulator += thresh.astype(np.float32)

        prev_gray = gray.copy()

        # --- Normalize accumulator to 0-255 ---
        max_val = accumulator.max()
        if max_val > 0:
            heatmap_norm = (accumulator / max_val * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros((height, width), dtype=np.uint8)

        # --- Apply colormap ---
        heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

        # --- Blend heatmap with camera frame ---
        overlay = cv2.addWeighted(frame, blend_alpha, heatmap_color, blend_beta, 0)

        # --- Draw FPS and controls ---
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Decay: {decay_factor:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Motion percentage
        motion_pixels = cv2.countNonZero(heatmap_norm)
        total_pixels = width * height
        motion_pct = 100.0 * motion_pixels / total_pixels
        cv2.putText(overlay, f"Active area: {motion_pct:.1f}%", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(overlay, "'r'=reset  'q'=quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Motion Heatmap', overlay)

        # --- Handle keypresses ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            accumulator[:] = 0
            print("Heatmap reset")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
