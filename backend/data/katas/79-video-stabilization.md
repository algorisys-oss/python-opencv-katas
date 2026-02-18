---
slug: 79-video-stabilization
title: Video Stabilization
level: advanced
concepts: [feature matching, affine correction, frame alignment]
prerequisites: [61-shi-tomasi-corners, 76-sparse-optical-flow]
---

## What Problem Are We Solving?

Handheld video footage is shaky — every small hand movement translates to frame-to-frame jitter that makes the video unpleasant to watch and hard to analyze. **Video stabilization** estimates the unwanted camera motion between frames and applies a **corrective transformation** to align them, producing smooth, stable output. This combines feature detection, optical flow, and geometric transformations into a complete pipeline.

## The Stabilization Pipeline

1. **Detect features** in each frame (Shi-Tomasi corners)
2. **Track features** to the next frame (Lucas-Kanade optical flow)
3. **Estimate motion** between frames (affine or rigid transformation)
4. **Accumulate transforms** to build a trajectory
5. **Smooth the trajectory** to remove jitter while preserving intentional motion
6. **Apply corrective transforms** to each frame

## Step 1-2: Feature Detection and Tracking

```python
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                     qualityLevel=0.01, minDistance=30)
curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                 prev_pts, None)
good_prev = prev_pts[status.flatten() == 1]
good_curr = curr_pts[status.flatten() == 1]
```

## Step 3: Estimating the Motion

Use `cv2.estimateAffinePartial2D()` to find the rigid transformation (rotation + translation + scale) between matched points:

```python
transform, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
```

The result is a 2x3 matrix:

```
[cos(theta)*s  -sin(theta)*s  tx]
[sin(theta)*s   cos(theta)*s  ty]
```

You can extract the individual components:

```python
dx = transform[0, 2]   # Translation x
dy = transform[1, 2]   # Translation y
da = np.arctan2(transform[1, 0], transform[0, 0])  # Rotation angle
```

## Step 4-5: Trajectory Smoothing

Accumulate the per-frame transforms into a trajectory, then smooth it:

```python
# Accumulate
trajectory = np.cumsum(transforms, axis=0)

# Smooth with a moving average
def smooth(trajectory, radius=15):
    smoothed = np.copy(trajectory)
    for i in range(trajectory.shape[1]):
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padded = np.pad(trajectory[:, i], radius, mode='edge')
        smoothed[:, i] = np.convolve(padded, kernel, mode='valid')
    return smoothed

smoothed_trajectory = smooth(trajectory)
```

The difference between the smoothed and original trajectory gives the correction:

```python
correction = smoothed_trajectory - trajectory
```

## Step 6: Applying the Correction

Add the correction to each frame's transform and apply:

```python
corrected_transform = original_transform + correction[i]
# Build 2x3 affine matrix from corrected dx, dy, da
m = np.array([
    [np.cos(da_c), -np.sin(da_c), dx_c],
    [np.sin(da_c),  np.cos(da_c), dy_c]
], dtype=np.float64)
stabilized = cv2.warpAffine(frame, m, (w, h))
```

## Border Handling

Stabilization shifts frames, creating black borders. Solutions:

- **Crop**: Remove a margin (e.g., 10%) from all sides
- **Scale up**: Slightly zoom in to cover the borders
- **Inpaint**: Fill borders using nearby pixel data

```python
# Crop approach
margin = 0.1
crop_x = int(w * margin)
crop_y = int(h * margin)
stabilized = stabilized[crop_y:h-crop_y, crop_x:w-crop_x]
stabilized = cv2.resize(stabilized, (w, h))
```

## Tips & Common Mistakes

- Use at least 100-200 feature points for robust motion estimation. Too few points make the estimate noisy.
- `cv2.estimateAffinePartial2D` is preferred over `estimateAffine2D` for stabilization because it constrains to rigid motion (rotation + translation + uniform scale), which matches camera shake better.
- The smoothing radius controls the trade-off: a larger radius removes more shake but may remove intentional panning. 15-30 frames is typical.
- Always handle the case where `transform` is `None` (not enough matching points). Use an identity matrix as fallback.
- For real-time stabilization, you need a small look-ahead buffer rather than processing the full video. This adds latency but enables smooth output.
- Black borders after stabilization are normal. Crop or zoom to remove them for a clean result.

## Starter Code

```python
import cv2
import numpy as np

# --- Create synthetic "shaky" frames to demonstrate stabilization ---
frame_h, frame_w = 250, 350

# Create a detailed scene (the "true" stable view)
scene = np.zeros((frame_h + 80, frame_w + 80, 3), dtype=np.uint8)
scene[:] = (60, 55, 50)

# Add textured elements to the scene
for y in range(0, scene.shape[0], 20):
    for x in range(0, scene.shape[1], 20):
        val = 50 + 25 * ((x // 20 + y // 20) % 2)
        cv2.rectangle(scene, (x, y), (x + 20, y + 20), (val, val, val), -1)

# Buildings / objects in the scene
cv2.rectangle(scene, (50, 40), (120, 180), (100, 120, 80), -1)
cv2.rectangle(scene, (160, 60), (220, 180), (80, 100, 120), -1)
cv2.rectangle(scene, (260, 50), (340, 180), (90, 110, 90), -1)
cv2.circle(scene, (200, 230), 30, (50, 80, 150), -1)
cv2.line(scene, (30, 200), (400, 200), (70, 90, 60), 3)

# --- Generate shaky frames by cropping with random offsets ---
num_frames = 8
np.random.seed(42)

# Simulate camera shake as random offsets
shake_x = np.random.randint(-15, 16, num_frames)
shake_y = np.random.randint(-10, 11, num_frames)
shake_angle = np.random.uniform(-3, 3, num_frames)  # Small rotation in degrees

shaky_frames = []
pad = 40  # Padding in the scene for shake room

for i in range(num_frames):
    # Extract a crop with the shake offset
    cx = pad + shake_x[i]
    cy = pad + shake_y[i]

    # Apply rotation around center
    M_rot = cv2.getRotationMatrix2D(
        (scene.shape[1] / 2, scene.shape[0] / 2),
        shake_angle[i], 1.0
    )
    rotated = cv2.warpAffine(scene, M_rot, (scene.shape[1], scene.shape[0]))
    crop = rotated[cy:cy + frame_h, cx:cx + frame_w]
    shaky_frames.append(crop)

# --- Stabilize: estimate and correct motion ---
transforms = []
prev_gray = cv2.cvtColor(shaky_frames[0], cv2.COLOR_BGR2GRAY)

for i in range(1, num_frames):
    curr_gray = cv2.cvtColor(shaky_frames[i], cv2.COLOR_BGR2GRAY)

    # Detect features
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=200, qualityLevel=0.01,
        minDistance=20, blockSize=7
    )

    if prev_pts is not None and len(prev_pts) > 0:
        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        # Estimate affine transform
        transform, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if transform is not None:
            dx = transform[0, 2]
            dy = transform[1, 2]
            da = np.arctan2(transform[1, 0], transform[0, 0])
            transforms.append([dx, dy, da])
        else:
            transforms.append([0, 0, 0])
    else:
        transforms.append([0, 0, 0])

    prev_gray = curr_gray.copy()

transforms = np.array(transforms)

# Compute trajectory (cumulative sum of transforms)
trajectory = np.cumsum(transforms, axis=0)

# Smooth trajectory with moving average
def smooth(traj, radius=2):
    smoothed = np.copy(traj)
    for col in range(traj.shape[1]):
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padded = np.pad(traj[:, col], radius, mode='edge')
        smoothed[:, col] = np.convolve(padded, kernel, mode='valid')
    return smoothed

smoothed_traj = smooth(trajectory, radius=2)
correction = smoothed_traj - trajectory

# Apply corrections
stabilized_frames = [shaky_frames[0]]  # First frame stays as-is

for i in range(len(transforms)):
    dx = transforms[i, 0] + correction[i, 0]
    dy = transforms[i, 1] + correction[i, 1]
    da = transforms[i, 2] + correction[i, 2]

    # Build stabilization matrix
    m = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da),  np.cos(da), dy]
    ], dtype=np.float64)

    # Apply inverse to correct
    m_inv = cv2.invertAffineTransform(m)
    stabilized = cv2.warpAffine(shaky_frames[i + 1], m_inv, (frame_w, frame_h))
    stabilized_frames.append(stabilized)

# --- Build composite display ---
font = cv2.FONT_HERSHEY_SIMPLEX

shaky_vis = []
stable_vis = []

for i in range(num_frames):
    s = shaky_frames[i].copy()
    cv2.putText(s, f'Shaky {i}', (10, 20), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    if i < len(shake_x):
        cv2.putText(s, f'dx={shake_x[i]:+d} dy={shake_y[i]:+d}', (10, frame_h - 10),
                    font, 0.35, (150, 150, 150), 1)
    shaky_vis.append(s)

    st = stabilized_frames[i].copy()
    cv2.putText(st, f'Stable {i}', (10, 20), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    stable_vis.append(st)

row1 = np.hstack(shaky_vis[:4])
row2 = np.hstack(shaky_vis[4:])
row3 = np.hstack(stable_vis[:4])
row4 = np.hstack(stable_vis[4:])

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

result = np.vstack([
    make_label('Shaky Input Frames (simulated camera shake)', row1.shape[1]),
    row1, row2,
    make_label('Stabilized Output (affine correction applied)', row3.shape[1]),
    row3, row4
])

print(f'Stabilized {num_frames} frames')
print(f'Max shake: dx={shake_x.max()}, dy={shake_y.max()}, angle={shake_angle.max():.1f} deg')

cv2.imshow('Video Stabilization', result)
```
