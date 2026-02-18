---
slug: 77-object-tracking-csrt
title: Object Tracking (CSRT)
level: advanced
concepts: [cv2.TrackerCSRT, ROI selection, tracker update]
prerequisites: [70-reading-video-files]
---

## What Problem Are We Solving?

Object detection runs on each frame independently, which is expensive and doesn't maintain identity across frames. **Object tracking** solves this: you detect an object once (or select it manually), then the tracker follows it frame by frame using visual features and motion models. This is much faster than running detection every frame and preserves object identity over time.

The **CSRT (Channel and Spatial Reliability Tracking)** tracker is one of OpenCV's most accurate single-object trackers, using spatial reliability maps to handle partial occlusion and appearance changes.

## Creating and Initializing a Tracker

```python
tracker = cv2.TrackerCSRT_create()
```

Initialize it with the first frame and a bounding box `(x, y, width, height)`:

```python
bbox = (x, y, w, h)  # Bounding box of the object to track
tracker.init(frame, bbox)
```

The bounding box can come from:
- Manual selection with `cv2.selectROI()`
- An object detector (YOLO, Haar cascade, etc.)
- Predefined coordinates

## The Tracking Loop

In each subsequent frame, call `tracker.update()`:

```python
success, bbox = tracker.update(frame)
if success:
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    cv2.putText(frame, 'Tracking lost', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

- `success` is `True` if the tracker found the object, `False` if tracking failed.
- `bbox` is the updated bounding box as `(x, y, w, h)`.

## Selecting an ROI Interactively

OpenCV provides a built-in ROI selector:

```python
bbox = cv2.selectROI('Select Object', frame, fromCenter=False)
```

This opens a window where you draw a rectangle around the target. Press Enter/Space to confirm, or C to cancel.

## Available Trackers in OpenCV

OpenCV provides several tracker algorithms:

| Tracker | Speed | Accuracy | Notes |
|---|---|---|---|
| `TrackerCSRT` | Slow | High | Best accuracy, handles scale changes |
| `TrackerKCF` | Fast | Medium | Good speed-accuracy balance |
| `TrackerMOSSE` | Very Fast | Low | Fastest, but struggles with scale |
| `TrackerMIL` | Medium | Medium | Handles partial occlusion |

Create any of them the same way:

```python
tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerMOSSE_create()
```

## Handling Tracking Failure

When `update()` returns `success=False`, the object is lost. Common recovery strategies:

1. **Re-detect**: Run an object detector to find the object again
2. **Expand search**: Search in a larger region around the last known position
3. **Reset**: Re-initialize the tracker with new coordinates

```python
if not success:
    # Re-initialize with detection result
    bbox = run_detector(frame)
    if bbox is not None:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
```

## Tips & Common Mistakes

- CSRT is the most accurate OpenCV tracker but also the slowest. For real-time applications at 30+ fps, consider KCF or MOSSE.
- The initial bounding box quality matters enormously. A tight, accurate box around just the object (no background) gives the best results.
- Trackers do not handle objects leaving the frame and re-entering. You need a re-detection mechanism for that.
- CSRT handles scale changes well, but extreme scale changes (object going from near to far) may cause failure.
- The tracker maintains an appearance model. If the object changes appearance dramatically (rotation, deformation), tracking may drift.
- Always create a **new tracker instance** when re-initializing. Don't reuse an old one — call `cv2.TrackerCSRT_create()` again.

## Starter Code

```python
import cv2
import numpy as np

# --- Simulate object tracking with synthetic frames ---
num_frames = 8
frame_h, frame_w = 300, 400

# Define the object's trajectory (a circle moving in an arc)
object_positions = []
for i in range(num_frames):
    cx = int(80 + i * 35)
    cy = int(150 + 40 * np.sin(i * 0.6))
    object_positions.append((cx, cy))

# Object size
obj_w, obj_h = 50, 40

# Create frames with the moving object
frames = []
for i, (cx, cy) in enumerate(object_positions):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (60, 55, 50)
    # Static background elements
    cv2.rectangle(frame, (10, 10), (50, 50), (80, 90, 70), -1)
    cv2.rectangle(frame, (350, 250), (390, 290), (70, 80, 90), -1)
    cv2.circle(frame, (300, 60), 20, (90, 70, 80), -1)
    # Moving object (colored rectangle with internal features)
    x1 = cx - obj_w // 2
    y1 = cy - obj_h // 2
    cv2.rectangle(frame, (x1, y1), (x1 + obj_w, y1 + obj_h), (50, 50, 220), -1)
    cv2.rectangle(frame, (x1 + 5, y1 + 5), (x1 + 15, y1 + 15), (100, 100, 255), -1)
    cv2.line(frame, (x1 + 20, y1 + 5), (x1 + 45, y1 + 35), (150, 150, 255), 2)
    frames.append(frame)

# --- Initialize CSRT tracker on the first frame ---
tracker = cv2.TrackerCSRT_create()
# Initial bounding box from known object position
init_x = object_positions[0][0] - obj_w // 2
init_y = object_positions[0][1] - obj_h // 2
init_bbox = (init_x, init_y, obj_w, obj_h)
tracker.init(frames[0], init_bbox)

# --- Track through all frames ---
tracking_results = []
font = cv2.FONT_HERSHEY_SIMPLEX

# Show initialization on first frame
vis0 = frames[0].copy()
cv2.rectangle(vis0, (init_x, init_y), (init_x + obj_w, init_y + obj_h),
              (0, 255, 255), 2)
cv2.putText(vis0, 'Init', (init_x, init_y - 5), font, 0.4, (0, 255, 255), 1)
cv2.putText(vis0, 'Frame 0', (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
tracking_results.append(vis0)

for i in range(1, len(frames)):
    success, bbox = tracker.update(frames[i])
    vis = frames[i].copy()
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, 'Tracking', (x, y - 5), font, 0.4, (0, 255, 0), 1)
        print(f'Frame {i}: tracked at ({x}, {y}, {w}, {h})')
    else:
        cv2.putText(vis, 'LOST', (frame_w // 2 - 30, frame_h // 2),
                    font, 0.8, (0, 0, 255), 2)
        print(f'Frame {i}: tracking lost')
    cv2.putText(vis, f'Frame {i}', (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    tracking_results.append(vis)

# --- Build composite display ---
# Row 1: frames 0-3, Row 2: frames 4-7
row1 = np.hstack(tracking_results[:4])
row2 = np.hstack(tracking_results[4:])

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

result = np.vstack([
    make_label('CSRT Object Tracking: Init (yellow) -> Tracking (green)', row1.shape[1]),
    row1,
    make_label('Continued Tracking', row2.shape[1]),
    row2
])

print(f'Tracked object across {num_frames} frames using CSRT')

cv2.imshow('Object Tracking (CSRT)', result)
```
