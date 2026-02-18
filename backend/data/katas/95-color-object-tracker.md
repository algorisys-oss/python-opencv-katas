---
slug: 95-color-object-tracker
title: Color-Based Object Tracker
level: advanced
concepts: [HSV masking, contour detection, centroid tracking]
prerequisites: [50-color-object-detection, 44-moments-centroids]
---

## What Problem Are We Solving?

You want to track a colored object as it moves through a video -- for example, following a red ball across a table, or tracking a green marker during a presentation. This requires combining **color detection** (to find the object), **contour analysis** (to locate it precisely), and **centroid computation** (to get its position). By recording positions across frames, you can draw a **tracking trail** showing the object's path.

This pipeline chains HSV color masking, morphological cleanup, contour detection, moment-based centroid computation, and trail visualization.

## Step 1: HSV Color Masking

Converting to HSV and using `cv2.inRange` isolates pixels matching your target color. The key advantage of HSV is that the Hue channel captures color identity independently of brightness:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower = np.array([0, 120, 100])     # Red hue range
upper = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
```

## Step 2: Morphological Cleanup

The raw mask often has noise (small false positives) and holes inside the object. Morphological opening removes small specks, and closing fills internal gaps:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

Using an elliptical kernel produces smoother results than a rectangular one for circular objects.

## Step 3: Find Contours and Select the Largest

After masking, `cv2.findContours` extracts the boundaries of white regions. We select the largest contour (by area) as our tracked object, ignoring small noise blobs:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
```

## Step 4: Compute the Centroid Using Moments

Image **moments** are weighted averages of pixel positions. The centroid (center of mass) is computed from the zeroth and first moments:

```python
M = cv2.moments(largest)
if M['m00'] > 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
```

| Moment | Meaning |
|---|---|
| `m00` | Total area (number of pixels) |
| `m10` | Sum of x-coordinates (weighted by intensity) |
| `m01` | Sum of y-coordinates (weighted by intensity) |
| `cx = m10/m00` | X coordinate of centroid |
| `cy = m01/m00` | Y coordinate of centroid |

## Step 5: Draw the Tracking Trail

By appending each centroid to a list, you build up a trail over time. Drawing lines between consecutive points creates a visible path:

```python
trail.append((cx, cy))
for i in range(1, len(trail)):
    cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)
```

For a fading trail effect, you can vary the color or thickness based on recency.

## The Complete Pipeline

1. **Input**: Video frame with a colored object
2. **HSV Conversion**: Convert from BGR to HSV color space
3. **Color Mask**: `cv2.inRange` to isolate target color
4. **Morphology**: Clean up mask with opening and closing
5. **Find Contours**: Detect object boundaries
6. **Centroid**: Compute center of mass from moments
7. **Trail**: Accumulate positions and draw path
8. **Output**: Frame with bounding circle, centroid marker, and trail

## Tips & Common Mistakes

- Red wraps around the HSV hue circle (both 0-10 and 170-179 are red). Create two masks and combine with `cv2.bitwise_or` for red objects.
- The `m00` check is essential. If the contour has zero area (degenerate case), dividing by zero crashes the program.
- Morphological kernel size should be proportional to the object size. Too large and you erode away small objects; too small and noise survives.
- For smoother tracking, apply a minimum area threshold to ignore tiny contours that are likely noise.
- In real video, add a Kalman filter or moving average to smooth the centroid position and handle brief occlusions.
- The trail can grow indefinitely. In practice, keep only the last N points (e.g., 50-100) for a fading trail effect.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create synthetic frames with a moving colored object
# =============================================================
frame_h, frame_w = 400, 600
num_frames = 60

# Background with some texture
background = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
background[:] = (60, 50, 40)
# Add grid texture to background
for x in range(0, frame_w, 30):
    cv2.line(background, (x, 0), (x, frame_h), (65, 55, 45), 1)
for y in range(0, frame_h, 30):
    cv2.line(background, (0, y), (frame_w, y), (65, 55, 45), 1)

# Generate object path: a figure-eight pattern
frames = []
object_positions = []
for i in range(num_frames):
    t = i * 2 * np.pi / num_frames
    # Figure-eight (lemniscate) path
    cx = int(frame_w // 2 + 180 * np.sin(t))
    cy = int(frame_h // 2 + 100 * np.sin(2 * t))
    object_positions.append((cx, cy))

    frame = background.copy()
    # Draw the red object (ball)
    cv2.circle(frame, (cx, cy), 22, (0, 0, 220), -1)
    cv2.circle(frame, (cx, cy), 22, (0, 0, 180), 2)
    # Add a highlight to make it look 3D
    cv2.circle(frame, (cx - 6, cy - 6), 6, (80, 80, 255), -1)

    # Add some distractor shapes (non-red)
    cv2.rectangle(frame, (30, 30), (90, 90), (200, 150, 0), -1)     # Cyan-ish
    cv2.circle(frame, (530, 350), 25, (0, 180, 0), -1)               # Green

    frames.append(frame)

print(f'Generated {num_frames} frames with moving red object')
print(f'Object follows a figure-eight path')

# =============================================================
# Step 2: Process each frame through the tracking pipeline
# =============================================================
trail = []  # List of (cx, cy) centroids
tracked_frames = []

# HSV range for red (lower red hue)
lower_red = np.array([0, 100, 100])
upper_red = np.array([15, 255, 255])

# Morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

for idx, frame in enumerate(frames):
    display = frame.copy()

    # --- HSV conversion and masking ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # --- Morphological cleanup ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Select the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 100:  # Minimum area threshold
            # --- Compute centroid ---
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                trail.append((cx, cy))

                # Draw bounding circle
                (bx, by), radius = cv2.minEnclosingCircle(largest)
                cv2.circle(display, (int(bx), int(by)), int(radius) + 5,
                          (0, 255, 0), 2)

                # Draw centroid
                cv2.circle(display, (cx, cy), 4, (255, 255, 0), -1)

    # --- Draw tracking trail ---
    for i in range(1, len(trail)):
        # Fade trail: older points are dimmer
        alpha = i / len(trail)
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
        thickness = max(1, int(3 * alpha))
        cv2.line(display, trail[i - 1], trail[i], color, thickness)

    # Add frame counter
    cv2.putText(display, f'Frame {idx}/{num_frames}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    tracked_frames.append(display)

print(f'Tracked {len(trail)} centroid positions across {num_frames} frames')

# =============================================================
# Step 3: Build visualization showing key frames
# =============================================================
# Select 4 frames at different stages
indices = [0, num_frames // 4, num_frames // 2, num_frames - 1]
selected = []
for i in indices:
    f = tracked_frames[i].copy()
    cv2.putText(f, f'Frame {i}', (10, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    selected.append(f)

# Also create a mask visualization from the last frame
hsv_last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2HSV)
mask_last = cv2.inRange(hsv_last, lower_red, upper_red)
mask_last = cv2.morphologyEx(mask_last, cv2.MORPH_OPEN, kernel)
mask_last = cv2.morphologyEx(mask_last, cv2.MORPH_CLOSE, kernel)
mask_bgr = cv2.cvtColor(mask_last, cv2.COLOR_GRAY2BGR)
cv2.putText(mask_bgr, 'Color Mask', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Final frame with full trail
final_frame = tracked_frames[-1].copy()
cv2.putText(final_frame, 'Full Trail', (10, frame_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Resize for grid: 3 rows x 2 cols
half_w = frame_w // 2
half_h = frame_h // 2

panels = []
for img in [selected[0], selected[1], selected[2], selected[3], mask_bgr, final_frame]:
    panels.append(cv2.resize(img, (half_w, half_h)))

row1 = np.hstack([panels[0], panels[1]])
row2 = np.hstack([panels[2], panels[3]])
row3 = np.hstack([panels[4], panels[5]])

result = np.vstack([row1, row2, row3])

cv2.imshow('Color-Based Object Tracker', result)
```
