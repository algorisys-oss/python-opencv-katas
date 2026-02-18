---
slug: 78-multi-object-tracking
title: Multi-Object Tracking
level: advanced
concepts: [multiple trackers, tracking multiple ROIs]
prerequisites: [77-object-tracking-csrt]
---

## What Problem Are We Solving?

Real-world scenes rarely have just one moving object. Surveillance systems track dozens of people, autonomous vehicles monitor multiple cars, and sports analytics follow every player. **Multi-object tracking** extends single-object tracking by managing **multiple tracker instances simultaneously**, updating each one per frame, and handling the complexities of objects entering, leaving, or occluding each other.

## Strategy: One Tracker Per Object

The simplest approach creates an independent tracker for each object:

```python
trackers = []
for bbox in bounding_boxes:
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    trackers.append(tracker)
```

Each frame, update all trackers:

```python
for i, tracker in enumerate(trackers):
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2)
```

## Using cv2.MultiTracker (Legacy)

Older OpenCV versions provided a `MultiTracker` class that managed multiple trackers together. In recent versions, it may not be available, so the manual approach above is more portable and gives you full control.

If available, the API looked like:

```python
multi_tracker = cv2.legacy.MultiTracker_create()
for bbox in bounding_boxes:
    multi_tracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

success, boxes = multi_tracker.update(frame)
```

## Managing Object Identity

Each tracker needs an identifier. Use a dictionary or list with IDs:

```python
tracked_objects = {}
next_id = 0

def add_object(frame, bbox):
    global next_id
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    tracked_objects[next_id] = {
        'tracker': tracker,
        'color': (np.random.randint(0, 255),
                  np.random.randint(0, 255),
                  np.random.randint(0, 255))
    }
    next_id += 1
```

## Handling Lost Objects

When a tracker reports failure, you have several options:

1. **Remove immediately**: Delete the tracker from the active set
2. **Grace period**: Keep the last known position for N frames before removing
3. **Re-detect and match**: Run detection and re-initialize lost trackers

```python
to_remove = []
for obj_id, obj in tracked_objects.items():
    success, bbox = obj['tracker'].update(frame)
    if not success:
        obj['lost_frames'] = obj.get('lost_frames', 0) + 1
        if obj['lost_frames'] > 10:  # Remove after 10 lost frames
            to_remove.append(obj_id)
for obj_id in to_remove:
    del tracked_objects[obj_id]
```

## Adding New Objects Dynamically

In a real system, new objects enter the scene. Periodically run a detector and initialize trackers for newly detected objects:

```python
# Run detector every 30 frames
if frame_count % 30 == 0:
    detections = run_detector(frame)
    for det_bbox in detections:
        if not overlaps_existing(det_bbox, tracked_objects):
            add_object(frame, det_bbox)
```

## Performance Considerations

- Each CSRT tracker takes ~5-10ms per update. With 10 objects, that is 50-100ms per frame.
- For many objects, use faster trackers (KCF ~1ms, MOSSE ~0.1ms per tracker).
- Consider running trackers in parallel using threading for better performance.
- Reduce the number of active trackers by removing objects that leave the frame.

## Tips & Common Mistakes

- Use a different color for each tracked object so you can visually verify tracking identity.
- When objects cross paths or occlude each other, individual trackers may swap targets. There is no built-in solution for this — you need higher-level logic (Re-ID, motion prediction).
- Initialize each tracker with a tight, accurate bounding box. Sloppy boxes cause drift early on.
- Don't forget to handle objects leaving the frame. Check if the bounding box is near the edge and remove if it exits.
- Creating many CSRT trackers can consume significant memory. If tracking 50+ objects, switch to KCF or MOSSE.
- Always create a fresh tracker instance for each object. Never reuse a tracker that was initialized for a different object.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a scene with multiple moving objects ---
num_frames = 8
frame_h, frame_w = 300, 500

# Define 3 objects with different trajectories
object_defs = [
    {'color': (50, 50, 220), 'size': (40, 35), 'label': 'A'},
    {'color': (50, 200, 50), 'size': (35, 45), 'label': 'B'},
    {'color': (220, 100, 50), 'size': (38, 38), 'label': 'C'},
]

trajectories = [
    [(60 + i * 40, 80 + int(30 * np.sin(i * 0.7))) for i in range(num_frames)],
    [(400 - i * 30, 200 - int(20 * np.cos(i * 0.5))) for i in range(num_frames)],
    [(100 + i * 25, 230 + int(15 * np.sin(i * 0.9 + 1))) for i in range(num_frames)],
]

# Generate frames
frames = []
for f_idx in range(num_frames):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (55, 50, 45)
    # Background elements
    cv2.rectangle(frame, (0, frame_h - 30), (frame_w, frame_h), (40, 60, 40), -1)
    cv2.circle(frame, (450, 40), 15, (70, 70, 90), -1)

    for obj_idx, obj in enumerate(object_defs):
        cx, cy = trajectories[obj_idx][f_idx]
        w, h = obj['size']
        x1, y1 = cx - w // 2, cy - h // 2
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), obj['color'], -1)
        # Internal feature for tracking
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    frames.append(frame)

# --- Initialize trackers for each object ---
trackers = []
track_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
font = cv2.FONT_HERSHEY_SIMPLEX

for obj_idx, obj in enumerate(object_defs):
    tracker = cv2.TrackerCSRT_create()
    cx, cy = trajectories[obj_idx][0]
    w, h = obj['size']
    bbox = (cx - w // 2, cy - h // 2, w, h)
    tracker.init(frames[0], bbox)
    trackers.append({
        'tracker': tracker,
        'label': obj['label'],
        'color': track_colors[obj_idx],
        'lost': False
    })

print(f'Initialized {len(trackers)} trackers')

# --- Track all objects through frames ---
result_frames = []

# Frame 0: show initialization
vis0 = frames[0].copy()
for obj_idx, obj in enumerate(object_defs):
    cx, cy = trajectories[obj_idx][0]
    w, h = obj['size']
    x1, y1 = cx - w // 2, cy - h // 2
    cv2.rectangle(vis0, (x1, y1), (x1 + w, y1 + h), track_colors[obj_idx], 2)
    cv2.putText(vis0, f'Init {obj["label"]}', (x1, y1 - 5),
                font, 0.35, track_colors[obj_idx], 1)
cv2.putText(vis0, 'Frame 0: Init', (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
result_frames.append(vis0)

for f_idx in range(1, num_frames):
    vis = frames[f_idx].copy()
    active_count = 0

    for t_info in trackers:
        if t_info['lost']:
            continue
        success, bbox = t_info['tracker'].update(frames[f_idx])
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(vis, (x, y), (x + w, y + h), t_info['color'], 2)
            cv2.putText(vis, t_info['label'], (x, y - 5),
                        font, 0.4, t_info['color'], 1)
            active_count += 1
        else:
            t_info['lost'] = True
            print(f'  Object {t_info["label"]} lost at frame {f_idx}')

    cv2.putText(vis, f'Frame {f_idx}: {active_count} active', (10, 20),
                font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    result_frames.append(vis)

    print(f'Frame {f_idx}: {active_count}/{len(trackers)} objects tracked')

# --- Build composite display ---
row1 = np.hstack(result_frames[:4])
row2 = np.hstack(result_frames[4:])

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

# Legend
legend = np.zeros((30, row1.shape[1], 3), dtype=np.uint8)
x_off = 10
for t_info, obj in zip(trackers, object_defs):
    cv2.rectangle(legend, (x_off, 5), (x_off + 15, 20), t_info['color'], -1)
    cv2.putText(legend, f'Object {t_info["label"]}', (x_off + 20, 18),
                font, 0.4, (200, 200, 200), 1)
    x_off += 130

result = np.vstack([
    make_label('Multi-Object Tracking with CSRT', row1.shape[1]),
    legend,
    row1, row2
])

print(f'\nTracked {len(trackers)} objects across {num_frames} frames')

cv2.imshow('Multi-Object Tracking', result)
```
