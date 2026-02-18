---
slug: 124-live-speed-estimation
title: Live Speed Estimation
level: live
concepts: [centroid tracking, frame-to-frame displacement, pixels per second, velocity estimation, cv2.norm]
prerequisites: [100-live-camera-fps, 107-live-color-object-tracking]
---

## What Problem Are We Solving?

Detecting that an object is moving is useful, but knowing **how fast** it is moving is far more powerful. Speed estimation enables applications like traffic monitoring (is a car speeding?), sports analytics (how fast did the ball travel?), robotics (is the robot approaching the target at a safe velocity?), and wildlife tracking (is the animal running or walking?).

This kata teaches you to track the centroid of a colored object across consecutive frames, compute the pixel displacement between frames, convert that displacement into a speed measurement (pixels per second) using the known FPS, and draw velocity vectors that show both speed and direction. Optionally, if you know a real-world reference distance in the scene, you can calibrate the system to report speed in cm/s or m/s.

## The Speed Estimation Pipeline

1. **Detect** the target object in each frame (using color-based segmentation in HSV)
2. **Find the centroid** of the detected region using `cv2.moments`
3. **Compare** the current centroid position to the previous frame's position
4. **Compute displacement** as the Euclidean distance between the two positions
5. **Multiply by FPS** to convert pixels-per-frame into pixels-per-second
6. **Draw** an arrowed line from the previous position to the current position, scaled by speed

## Centroid Detection with cv2.moments

Image moments provide a compact summary of a binary region's shape. The centroid (center of mass) is derived from the zeroth and first moments:

```python
M = cv2.moments(mask)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
```

| Moment | Meaning |
|---|---|
| `m00` | Total area (number of white pixels in the mask) |
| `m10` | Sum of x-coordinates of all white pixels |
| `m01` | Sum of y-coordinates of all white pixels |
| `m10 / m00` | X coordinate of centroid |
| `m01 / m00` | Y coordinate of centroid |

The `m00 > 0` check is critical -- if no object is detected, `m00` is zero and the division would crash.

## Computing Displacement and Speed

Once you have the centroid in the current frame `(cx, cy)` and the previous frame `(px, py)`:

```python
dx = cx - px
dy = cy - py
displacement_px = np.sqrt(dx**2 + dy**2)  # Euclidean distance in pixels
speed_px_per_sec = displacement_px * fps    # Convert to px/s
```

**Why multiply by FPS?** The displacement is in pixels *per frame*. If the camera runs at 30 FPS, an object moving 5 pixels per frame is actually moving `5 * 30 = 150` pixels per second. Without the FPS conversion, the same physical speed would appear different on cameras running at different frame rates.

## Drawing Velocity Vectors

`cv2.arrowedLine` draws a line with an arrowhead -- perfect for showing direction and magnitude of motion:

```python
cv2.arrowedLine(frame, (px, py), (cx, cy), color, thickness, tipLength=0.3)
```

| Parameter | Type | Description |
|---|---|---|
| `img` | `np.ndarray` | Image to draw on |
| `pt1` | `tuple` | Start point (previous position) |
| `pt2` | `tuple` | End point (current position) |
| `color` | `tuple` | BGR color |
| `thickness` | `int` | Line thickness in pixels |
| `tipLength` | `float` | Arrow tip length as fraction of line length (0.0-1.0) |

For small displacements, the arrow becomes invisibly tiny. Scale the vector to make it visible:

```python
scale = 5  # Exaggerate the arrow length for visibility
end_x = px + int(dx * scale)
end_y = py + int(dy * scale)
cv2.arrowedLine(frame, (px, py), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
```

## Color-Based Object Detection in HSV

To track a colored object, convert the frame to HSV and create a mask for the target color range:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_bound, upper_bound)
```

| Parameter | Type | Description |
|---|---|---|
| `src` | `np.ndarray` | Input HSV image |
| `lowerb` | `np.ndarray` or `tuple` | Lower HSV bound (H, S, V) |
| `upperb` | `np.ndarray` or `tuple` | Upper HSV bound (H, S, V) |
| **Returns** | `np.ndarray` | Binary mask (255 where in range, 0 elsewhere) |

Common HSV ranges for tracking:

| Color | Lower (H, S, V) | Upper (H, S, V) |
|---|---|---|
| Bright red | `(0, 120, 100)` | `(10, 255, 255)` |
| Green | `(35, 100, 100)` | `(85, 255, 255)` |
| Blue | `(100, 120, 100)` | `(130, 255, 255)` |
| Yellow | `(20, 100, 100)` | `(35, 255, 255)` |

## Optional: Real-World Calibration

If you place a reference object of known size in the scene (e.g., an A4 sheet that is 297mm wide), you can compute a pixel-to-millimeter conversion factor:

```python
pixels_per_mm = measured_width_in_pixels / 297.0
speed_mm_per_sec = speed_px_per_sec / pixels_per_mm
speed_cm_per_sec = speed_mm_per_sec / 10.0
```

This calibration is only valid at a fixed camera distance and angle. If the camera or object moves closer/farther, the scale changes.

## Smoothing Speed Readings

Raw frame-to-frame speed is noisy -- tiny centroid jitter from segmentation noise causes nonzero speed even for stationary objects. Use a rolling average to smooth:

```python
from collections import deque
speed_history = deque(maxlen=10)
speed_history.append(speed_px_per_sec)
smoothed_speed = sum(speed_history) / len(speed_history)
```

Also apply a **minimum speed threshold** to filter out jitter:

```python
if displacement_px < 3:  # Less than 3 pixels of movement = stationary
    speed_px_per_sec = 0.0
```

## Tips & Common Mistakes

- Always check `M["m00"] > 0` before computing the centroid. Dividing by zero crashes silently or produces `inf`.
- Clean the mask with `cv2.erode` followed by `cv2.dilate` (opening) to remove noise blobs before computing moments. Noise shifts the centroid.
- The speed is only meaningful when the object is actually detected in both the current and previous frames. If detection is lost for a frame, skip the speed calculation for that frame.
- FPS must be measured, not assumed. Using a hardcoded 30 FPS when the actual rate is 15 FPS doubles your speed estimate.
- Pixel speed depends on distance from the camera. An object close to the camera appears to move faster (more pixels per second) than the same object far away.
- For objects that move very fast (large displacement per frame), the centroid might jump erratically if the object blurs or leaves the frame. Cap the maximum speed display value to avoid nonsensical readings.
- HSV hue wraps around at 180 in OpenCV. Red spans both `0-10` and `170-180`. Use two `inRange` calls and combine with `cv2.bitwise_or` if tracking red.

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

# --- HSV range for the tracked object (bright green by default) ---
# Adjust these values for your colored object
lower_hsv = np.array([35, 100, 100])
upper_hsv = np.array([85, 255, 255])

# --- Tracking state ---
prev_centroid = None
speed_history = deque(maxlen=10)  # Rolling average for smooth readings
trail = deque(maxlen=64)          # Trail of recent positions
min_displacement = 3              # Pixels below this = stationary
arrow_scale = 5                   # Scale factor for velocity arrow

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

        # --- Detect colored object ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Clean up mask: remove noise, fill gaps
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # --- Find centroid ---
        M = cv2.moments(mask)
        centroid = None

        if M["m00"] > 500:  # Minimum area to count as a valid detection
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
            trail.append(centroid)

            # Draw centroid marker
            cv2.circle(frame, centroid, 8, (0, 255, 255), -1)
            cv2.circle(frame, centroid, 10, (0, 200, 200), 2)

        # --- Compute speed ---
        speed_px_s = 0.0
        dx, dy = 0, 0

        if centroid is not None and prev_centroid is not None:
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            displacement = np.sqrt(dx**2 + dy**2)

            if displacement >= min_displacement and fps > 0:
                speed_px_s = displacement * fps
            else:
                speed_px_s = 0.0

            speed_history.append(speed_px_s)

            # --- Draw velocity arrow ---
            if speed_px_s > 0:
                end_x = prev_centroid[0] + int(dx * arrow_scale)
                end_y = prev_centroid[1] + int(dy * arrow_scale)
                # Clamp to frame bounds
                end_x = max(0, min(width - 1, end_x))
                end_y = max(0, min(height - 1, end_y))
                cv2.arrowedLine(frame, prev_centroid, (end_x, end_y),
                                (0, 255, 255), 2, tipLength=0.3)

        prev_centroid = centroid

        # --- Draw trail ---
        for i in range(1, len(trail)):
            thickness = max(1, int(i / len(trail) * 3))
            cv2.line(frame, trail[i - 1], trail[i], (0, 180, 255), thickness)

        # --- Compute smoothed speed ---
        smoothed_speed = sum(speed_history) / len(speed_history) if speed_history else 0.0

        # --- Draw speed info ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        speed_color = (0, 255, 0) if smoothed_speed < 200 else (0, 165, 255) if smoothed_speed < 500 else (0, 0, 255)
        cv2.putText(frame, f"Speed: {smoothed_speed:.0f} px/s", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2, cv2.LINE_AA)

        if centroid is not None:
            cv2.putText(frame, f"Position: ({centroid[0]}, {centroid[1]})", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Delta: ({dx}, {dy})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No object detected", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "'q'=quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Speed Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
