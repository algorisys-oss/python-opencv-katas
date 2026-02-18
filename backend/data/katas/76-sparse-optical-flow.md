---
slug: 76-sparse-optical-flow
title: Sparse Optical Flow
level: advanced
concepts: [cv2.calcOpticalFlowPyrLK, Lucas-Kanade, point tracking]
prerequisites: [61-shi-tomasi-corners, 75-dense-optical-flow]
---

## What Problem Are We Solving?

Dense optical flow computes motion for every pixel, which is slow and often unnecessary. When you only need to track **specific points** (corners, feature points, key landmarks), **sparse optical flow** using the **Lucas-Kanade method** is far more efficient. It tracks a set of points from one frame to the next, telling you exactly where each point moved. This is the foundation of feature tracking, visual odometry, and video stabilization.

## The Lucas-Kanade Method

The Lucas-Kanade algorithm assumes that motion is small and consistent within a local neighborhood around each point. It solves for the displacement that best explains the intensity change in a small window:

```python
next_pts, status, err = cv2.calcOpticalFlowPyrLK(
    prev_gray, next_gray,
    prev_pts, None,
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
```

| Parameter | Meaning |
|---|---|
| `prev_gray` | Previous frame (grayscale) |
| `next_gray` | Next frame (grayscale) |
| `prev_pts` | Points to track, shape `(N, 1, 2)`, dtype `float32` |
| `None` | Initial guess for next positions (None = use prev_pts) |
| `winSize` | Search window size around each point |
| `maxLevel` | Number of pyramid levels (0 = no pyramid) |
| `criteria` | Termination criteria for the iterative search |

## Understanding the Output

The function returns three arrays:

```python
next_pts, status, err = cv2.calcOpticalFlowPyrLK(...)
```

- `next_pts`: New positions of the tracked points, shape `(N, 1, 2)`
- `status`: Array of `uint8`, shape `(N, 1)` — `1` if the point was found, `0` if lost
- `err`: Error measure for each point (lower is better)

Always filter by status to keep only successfully tracked points:

```python
good_new = next_pts[status == 1]
good_old = prev_pts[status == 1]
```

## Selecting Points to Track

Typically you use Shi-Tomasi corner detection to find good points:

```python
prev_pts = cv2.goodFeaturesToTrack(
    prev_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)
```

The returned array has the right shape `(N, 1, 2)` and dtype `float32` for direct use with `calcOpticalFlowPyrLK`.

## Drawing Tracks

Visualize the motion by drawing lines from old positions to new positions:

```python
for new, old in zip(good_new, good_old):
    a, b = new.ravel().astype(int)
    c, d = old.ravel().astype(int)
    cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
result = cv2.add(frame, mask)
```

Use a persistent `mask` image (initialized to zeros) to accumulate tracks over time.

## Pyramid Levels

The `maxLevel` parameter enables **pyramidal** Lucas-Kanade. The image is downsampled multiple times, and tracking runs from coarse to fine:

- **Level 0**: Original resolution only — handles very small motions
- **Level 2-3**: Can handle motions up to ~30 pixels — good for most applications
- **Level 4+**: For very large motions, but increases computation

## Tips & Common Mistakes

- Input points must be `float32` with shape `(N, 1, 2)`. Use `np.float32(pts).reshape(-1, 1, 2)` to ensure the correct format.
- Always check the `status` array. Points near image borders, in occluded regions, or in textureless areas will be lost (`status=0`).
- Periodically re-detect features. Over many frames, tracked points drift and are lost. Refresh the point set every 20-50 frames.
- The `winSize` should match the expected motion magnitude. A 15x15 window handles ~7 pixel motions per level; with 3 pyramid levels, that's ~56 pixels total.
- Forward-backward validation: Track points forward, then backward, and keep only points where the round-trip error is small. This removes unreliable tracks.
- Sparse flow is much faster than dense flow — tracking 100 points takes <1ms vs. 20-50ms for dense flow.

## Starter Code

```python
import cv2
import numpy as np

# --- Create synthetic frames with trackable features ---
frame_h, frame_w = 300, 400

def make_scene(objects):
    """Create a frame with textured background and objects."""
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    # Add textured background (checkerboard)
    for y in range(0, frame_h, 15):
        for x in range(0, frame_w, 15):
            val = 40 + 20 * ((x // 15 + y // 15) % 2)
            cv2.rectangle(frame, (x, y), (x + 15, y + 15), (val, val, val), -1)
    # Draw objects
    for (cx, cy, r, color) in objects:
        cv2.circle(frame, (cx, cy), r, color, -1)
        # Add corner-like features on the object
        cv2.rectangle(frame, (cx - r//2, cy - r//2),
                      (cx - r//2 + 6, cy - r//2 + 6), (255, 255, 255), -1)
        cv2.rectangle(frame, (cx + r//3, cy + r//3),
                      (cx + r//3 + 6, cy + r//3 + 6), (0, 0, 0), -1)
    return frame

# Create 4 frames with objects moving
object_positions = [
    [(80, 130, 30, (200, 80, 50)),  (280, 90, 25, (50, 80, 200)),  (200, 220, 20, (80, 200, 80))],
    [(100, 135, 30, (200, 80, 50)), (265, 95, 25, (50, 80, 200)),  (210, 215, 20, (80, 200, 80))],
    [(120, 140, 30, (200, 80, 50)), (250, 100, 25, (50, 80, 200)), (220, 210, 20, (80, 200, 80))],
    [(140, 145, 30, (200, 80, 50)), (235, 105, 25, (50, 80, 200)), (230, 205, 20, (80, 200, 80))],
]

frames = [make_scene(objs) for objs in object_positions]

# --- Detect features in the first frame ---
gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(
    gray0, maxCorners=50, qualityLevel=0.2,
    minDistance=10, blockSize=7
)

print(f'Detected {len(prev_pts)} features in frame 0')

# --- Track points through all frames ---
track_mask = np.zeros_like(frames[0])  # Persistent track drawing
tracking_results = []
colors = np.random.randint(0, 255, (len(prev_pts), 3)).tolist()

prev_gray = gray0.copy()
all_vis = []

for i in range(1, len(frames)):
    next_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

    # Compute sparse optical flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, prev_pts, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Filter by status
    good_new = next_pts[status.flatten() == 1]
    good_old = prev_pts[status.flatten() == 1]

    # Draw tracks
    vis_frame = frames[i].copy()
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        color = colors[j % len(colors)]
        cv2.line(track_mask, (a, b), (c, d), color, 2)
        cv2.circle(vis_frame, (a, b), 4, (0, 0, 255), -1)

    vis_frame = cv2.add(vis_frame, track_mask)
    cv2.putText(vis_frame, f'Frame {i}: {len(good_new)} tracked', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    all_vis.append(vis_frame)

    tracked = len(good_new)
    lost = int(status.size - status.sum())
    print(f'Frame {i}: tracked={tracked}, lost={lost}')

    # Update for next iteration
    prev_pts = good_new.reshape(-1, 1, 2)
    prev_gray = next_gray.copy()

# --- Show initial detection ---
detect_vis = frames[0].copy()
for pt in cv2.goodFeaturesToTrack(gray0, maxCorners=50, qualityLevel=0.2,
                                   minDistance=10, blockSize=7):
    x, y = pt.ravel().astype(int)
    cv2.circle(detect_vis, (x, y), 4, (0, 255, 255), -1)
cv2.putText(detect_vis, 'Detected Features', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# --- Build composite ---
font = cv2.FONT_HERSHEY_SIMPLEX
row1 = np.hstack([detect_vis] + all_vis)

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

result = np.vstack([
    make_label('Sparse Optical Flow (Lucas-Kanade): Feature Detection -> Tracking', row1.shape[1]),
    row1
])

cv2.imshow('Sparse Optical Flow', result)
```
