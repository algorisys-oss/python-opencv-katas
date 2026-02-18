---
slug: 75-dense-optical-flow
title: Dense Optical Flow
level: advanced
concepts: [cv2.calcOpticalFlowFarneback, motion field, HSV visualization]
prerequisites: [02-color-spaces, 70-reading-video-files]
---

## What Problem Are We Solving?

When objects move between frames, we often need to know **how each pixel moved** — not just that motion occurred, but the **direction and magnitude** of movement at every point. **Dense optical flow** computes a motion vector for every pixel in the image, producing a complete **motion field**. This is essential for motion segmentation, action recognition, video compression, and frame interpolation.

## The Farneback Method

OpenCV implements dense optical flow using the Gunnar Farneback algorithm:

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, next_gray,
    None,            # flow output (None to create new)
    pyr_scale=0.5,   # pyramid scale
    levels=3,        # number of pyramid levels
    winsize=15,      # averaging window size
    iterations=3,    # iterations at each pyramid level
    poly_n=5,        # size of pixel neighborhood
    poly_sigma=1.2,  # std dev for polynomial expansion
    flags=0
)
```

The result `flow` is a 2-channel float array with shape `(H, W, 2)`:
- `flow[..., 0]` = horizontal displacement (dx) in pixels
- `flow[..., 1]` = vertical displacement (dy) in pixels

## Understanding the Flow Field

Each pixel `(y, x)` in the flow tells you where that pixel "came from" between the two frames:

```python
dx = flow[y, x, 0]  # How far the pixel moved horizontally
dy = flow[y, x, 1]  # How far the pixel moved vertically
```

Positive `dx` means the pixel moved right; positive `dy` means it moved down.

## Visualizing Flow as HSV

The standard way to visualize dense optical flow is using **HSV color space**:
- **Hue** encodes the direction of motion (angle)
- **Value** (brightness) encodes the speed (magnitude)
- **Saturation** is set to maximum (255)

```python
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros((h, w, 3), dtype=np.uint8)
hsv[..., 0] = ang * 180 / np.pi / 2   # Hue: 0-179
hsv[..., 1] = 255                       # Saturation: full
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

In the result:
- **Red** = rightward motion
- **Green** = downward motion
- **Blue** = leftward motion
- **Yellow/Cyan** = diagonal motion
- **Black** = no motion

## Parameter Tuning

| Parameter | Effect |
|---|---|
| `pyr_scale` | Scale between pyramid levels. 0.5 means each level is half the size |
| `levels` | More levels handle larger displacements but cost more time |
| `winsize` | Larger windows are smoother but less precise. 15 is a good default |
| `iterations` | More iterations improve accuracy at the cost of speed |
| `poly_n` | 5 or 7 — larger values produce smoother flow |
| `poly_sigma` | 1.1 for poly_n=5, 1.5 for poly_n=7 |

## Drawing Flow Arrows

An alternative visualization draws motion vectors as arrows on a grid:

```python
step = 16
y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
fx = flow[y_coords, x_coords, 0]
fy = flow[y_coords, x_coords, 1]
for i in range(y_coords.shape[0]):
    for j in range(y_coords.shape[1]):
        pt1 = (int(x_coords[i, j]), int(y_coords[i, j]))
        pt2 = (int(x_coords[i, j] + fx[i, j]),
               int(y_coords[i, j] + fy[i, j]))
        cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
```

## Tips & Common Mistakes

- Both input frames must be **single-channel grayscale** and the same size.
- Dense flow is **computationally expensive** — expect 10-50ms per frame pair depending on resolution and parameters.
- Normalize the magnitude for visualization. Raw magnitudes can be very small (sub-pixel) or very large.
- Large motions require more pyramid levels. If objects move more than ~20 pixels between frames, increase `levels` or `winsize`.
- Flow vectors represent pixel displacement, not velocity. To get velocity, divide by the time between frames.
- Noisy regions (uniform textures, flat colors) produce unreliable flow. The algorithm needs texture/gradients to track.

## Starter Code

```python
import cv2
import numpy as np

# --- Create two synthetic frames with known motion ---
frame_h, frame_w = 300, 400

# Frame 1: objects at initial positions
frame1 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
frame1[:] = (50, 50, 50)

# Add textured background for flow estimation
for y in range(0, frame_h, 20):
    for x in range(0, frame_w, 20):
        val = int(40 + 30 * ((x // 20 + y // 20) % 2))
        cv2.rectangle(frame1, (x, y), (x + 20, y + 20), (val, val, val), -1)

# Object 1: circle on the left
cv2.circle(frame1, (100, 150), 40, (200, 100, 50), -1)
# Object 2: rectangle at top
cv2.rectangle(frame1, (250, 50), (320, 110), (50, 100, 200), -1)

# Frame 2: objects shifted
frame2 = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
frame2[:] = (50, 50, 50)

for y in range(0, frame_h, 20):
    for x in range(0, frame_w, 20):
        val = int(40 + 30 * ((x // 20 + y // 20) % 2))
        cv2.rectangle(frame2, (x, y), (x + 20, y + 20), (val, val, val), -1)

# Object 1: moved right and slightly down
cv2.circle(frame2, (140, 165), 40, (200, 100, 50), -1)
# Object 2: moved left
cv2.rectangle(frame2, (220, 50), (290, 110), (50, 100, 200), -1)

# --- Compute dense optical flow ---
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# --- Visualize flow as HSV ---
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
hsv[..., 0] = ang * 180 / np.pi / 2  # Direction -> Hue
hsv[..., 1] = 255                      # Full saturation
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Speed -> brightness
flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# --- Draw flow arrows on frame ---
arrow_vis = frame2.copy()
step = 20
h, w = gray1.shape
for y in range(step // 2, h, step):
    for x in range(step // 2, w, step):
        dx = flow[y, x, 0]
        dy = flow[y, x, 1]
        if np.sqrt(dx**2 + dy**2) > 1.0:  # Only draw significant motion
            pt1 = (x, y)
            pt2 = (int(x + dx * 2), int(y + dy * 2))
            cv2.arrowedLine(arrow_vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

# --- Add labels ---
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame1, 'Frame 1', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(frame2, 'Frame 2', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(flow_bgr, 'Flow HSV', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(arrow_vis, 'Flow Arrows', (10, 25), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

# --- Build composite ---
row1 = np.hstack([frame1, frame2])
row2 = np.hstack([flow_bgr, arrow_vis])

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

result = np.vstack([
    make_label('Input Frames', row1.shape[1]),
    row1,
    make_label('Dense Optical Flow Visualization', row2.shape[1]),
    row2
])

print(f'Flow field shape: {flow.shape}')
print(f'Max displacement: {mag.max():.2f} pixels')
print(f'HSV: hue=direction, brightness=speed')

cv2.imshow('Dense Optical Flow', result)
```
