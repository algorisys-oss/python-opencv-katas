---
slug: 111-live-people-counter
title: Live People Counter (Line Crossing)
level: live
concepts: [background subtraction, cv2.createBackgroundSubtractorMOG2, centroid tracking, line crossing detection]
prerequisites: [100-live-camera-fps, 74-background-subtraction]
---

## What Problem Are We Solving?

Counting people (or vehicles, animals, or any moving objects) as they cross a boundary is a fundamental task in retail analytics, building occupancy management, traffic monitoring, and security systems. A store owner wants to know how many customers entered vs exited. A building manager needs real-time occupancy counts. A traffic engineer counts vehicles crossing an intersection.

The approach combines two techniques: **background subtraction** to detect moving objects, and **centroid tracking** to follow them frame-to-frame and detect when they cross a virtual counting line. Background subtraction is superior to simple frame differencing for this task because it builds and maintains a model of the static background, adapting to gradual changes like shifting shadows or slow lighting transitions, and cleanly separating foreground (moving) objects.

This kata builds a complete line-crossing counter with up/down counts, centroid tracking with ID assignment, and a visual counting line on a live webcam feed.

## Background Subtraction with MOG2

MOG2 (Mixture of Gaussians 2) models each pixel's background as a mixture of Gaussian distributions. It learns what "normal" looks like for each pixel and flags anything different as foreground:

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `history` | 500 | Number of recent frames used to build the background model. Higher = slower adaptation, more stable. Lower = faster adaptation, more responsive to scene changes |
| `varThreshold` | 16 | Pixel variance threshold for foreground classification. Higher = less sensitive (fewer false positives). Lower = more sensitive |
| `detectShadows` | True | If True, shadows are detected and marked as gray (127) instead of white (255) in the mask |

### Applying the Background Subtractor

```python
fg_mask = bg_subtractor.apply(frame, learningRate=-1)
```

| Parameter | Meaning |
|---|---|
| `frame` | Current BGR frame |
| `learningRate` | -1 = automatic. 0 = background never updates (freezes). 1 = every frame fully replaces the model |
| **Returns** | Foreground mask: 255=foreground, 127=shadow (if detectShadows=True), 0=background |

### Shadow Removal

When `detectShadows=True`, shadows appear as gray (127) in the mask. Remove them:

```python
# Threshold to remove shadows (keep only definite foreground)
_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
```

### MOG2 Parameter Tuning Guide

| Scenario | history | varThreshold | detectShadows | Notes |
|---|---|---|---|---|
| Indoor, stable lighting | 500 | 50 | True | Default works well |
| Outdoor, changing light | 1000 | 40 | True | Longer history for gradual changes |
| Fast-moving objects | 200 | 30 | False | Short history, lower threshold |
| Very noisy camera | 500 | 80 | True | Higher threshold to suppress noise |

## Cleaning the Foreground Mask

The raw foreground mask from MOG2 contains noise. Clean it before finding contours:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Remove small noise blobs
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Fill gaps inside detected objects
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Optional: dilate to merge nearby blobs (one person = one blob)
fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
```

## Centroid Tracking: Matching Objects Across Frames

The key challenge in people counting is **identity persistence**: knowing that the blob at position (200, 300) in frame N is the same person as the blob at (205, 310) in frame N+1. Without this, you cannot detect line crossings.

### Simple Centroid Distance Matching

The simplest approach: for each centroid in the current frame, find the nearest centroid from the previous frame within a maximum distance:

```python
import math

MAX_DISTANCE = 80  # Maximum pixels an object can move between frames

def match_centroids(prev_centroids, curr_centroids, max_dist):
    """Match current centroids to previous ones by nearest distance.

    Returns:
        matches: list of (prev_id, curr_index) pairs
        unmatched_curr: indices of new objects
    """
    matches = []
    used_prev = set()
    used_curr = set()

    # For each current centroid, find nearest previous centroid
    for ci, (cx, cy) in enumerate(curr_centroids):
        best_dist = max_dist
        best_pi = None
        for pi, (px, py) in prev_centroids.items():
            if pi in used_prev:
                continue
            dist = math.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi is not None:
            matches.append((best_pi, ci))
            used_prev.add(best_pi)
            used_curr.add(ci)

    unmatched_curr = [i for i in range(len(curr_centroids)) if i not in used_curr]
    return matches, unmatched_curr
```

| Parameter | Suggested Value | Meaning |
|---|---|---|
| `MAX_DISTANCE` | 50-100 pixels | Maximum distance an object can move between consecutive frames. Too low = lost tracking. Too high = identity swaps |

### Object ID Assignment

New centroids that do not match any previous centroid get a new ID:

```python
next_id = 0
tracked_objects = {}  # {id: (cx, cy)}

# After matching:
new_tracked = {}
for prev_id, curr_idx in matches:
    new_tracked[prev_id] = curr_centroids[curr_idx]

for curr_idx in unmatched_curr:
    new_tracked[next_id] = curr_centroids[curr_idx]
    next_id += 1

tracked_objects = new_tracked
```

## Line Crossing Detection

Place a virtual horizontal line across the frame. Count when a tracked object's centroid crosses it:

```python
LINE_Y = frame_height // 2  # Horizontal line at center

# For each tracked object, check if it crossed the line
for obj_id, (cx, cy) in tracked_objects.items():
    if obj_id in prev_positions:
        prev_y = prev_positions[obj_id][1]

        # Crossed downward (entered)
        if prev_y < LINE_Y and cy >= LINE_Y:
            count_down += 1

        # Crossed upward (exited)
        elif prev_y > LINE_Y and cy <= LINE_Y:
            count_up += 1
```

### Crossing Detection Logic

| Previous Y | Current Y | Line Y | Direction | Interpretation |
|---|---|---|---|---|
| above line | below line | mid | Downward crossing | Count as "entered" (down) |
| below line | above line | mid | Upward crossing | Count as "exited" (up) |
| above line | above line | mid | No crossing | No count |
| below line | below line | mid | No crossing | No count |

> **Critical:** You must compare the **previous** position to the **current** position of the **same object** (same ID). Comparing positions from different objects will produce garbage counts.

## Drawing the Counting Line and Stats

```python
# Draw counting line
cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 255, 255), 2)

# Draw counts
cv2.putText(frame, f"Down: {count_down}", (10, LINE_Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
cv2.putText(frame, f"Up: {count_up}", (10, LINE_Y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

## Tips & Common Mistakes

- **MOG2 needs a warm-up period.** The first 50-100 frames are used to learn the background. During this time, everything may appear as foreground. Skip counting during warm-up.
- The `varThreshold` parameter has the biggest impact on sensitivity. Start with 50 for indoor scenes and adjust down if objects are missed, or up if there are too many false detections.
- Shadow detection (`detectShadows=True`) is important. Without it, moving shadows trigger false detections. Always threshold at 200+ to remove shadow pixels.
- The maximum matching distance should be roughly 2x the expected per-frame movement. If people move 30 pixels per frame, set `MAX_DISTANCE` to 60-80.
- Objects that stop moving will gradually be absorbed into the background model. This is correct behavior for MOG2 — parked cars and stationary people become background. Increase `history` if this happens too quickly.
- The counting line position matters. Place it where objects are well-separated and moving at consistent speed — typically the middle of the frame, not near the edges where objects appear/disappear.
- For accurate counting, the camera should be positioned overhead looking down, or at an angle where people cross the line individually. Side-view cameras cause heavy occlusion.
- `cv2.createBackgroundSubtractorMOG2` creates a stateful object. Do not recreate it every frame — initialize it once and call `apply()` each frame.
- The `learningRate` parameter of `apply()` controls how fast the background adapts. Use -1 (automatic) unless you have a specific reason to override.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Wait for the warm-up period to complete (progress shown on screen) — the background model needs ~60 frames to stabilize before counting begins
- Walk across the camera view crossing the yellow "COUNTING LINE" in the middle — the Up/Down counters should increment depending on your direction
- Check that moving objects get green bounding boxes with numbered ID labels and red centroid dots
- Look at the small mask preview in the top-right corner to verify the foreground detection is clean

## Starter Code

```python
import cv2
import numpy as np
import math
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_times = deque(maxlen=30)

# --- Background subtractor ---
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)

# --- Morphological kernel ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- Counting configuration ---
LINE_Y = frame_height // 2
MIN_CONTOUR_AREA = 2000
MAX_MATCH_DISTANCE = 80

# --- Tracking state ---
next_id = 0
tracked = {}       # {id: (cx, cy)}
prev_tracked = {}  # Previous frame positions
count_up = 0
count_down = 0
counted_ids = set()  # IDs that have already been counted (avoid double counting)

WARMUP_FRAMES = 60
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Apply background subtraction ---
        fg_mask = bg_sub.apply(frame)

        # --- Remove shadows ---
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # --- Morphological cleanup ---
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # --- Find contours ---
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Extract centroids of valid contours ---
        curr_centroids = []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            curr_centroids.append((cx, cy))
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- Match centroids to tracked objects ---
        prev_tracked = dict(tracked)
        new_tracked = {}
        used_prev = set()
        used_curr = set()

        for ci, (cx, cy) in enumerate(curr_centroids):
            best_dist = MAX_MATCH_DISTANCE
            best_id = None
            for obj_id, (px, py) in tracked.items():
                if obj_id in used_prev:
                    continue
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            if best_id is not None:
                new_tracked[best_id] = (cx, cy)
                used_prev.add(best_id)
                used_curr.add(ci)

        # Assign new IDs to unmatched centroids
        for ci, (cx, cy) in enumerate(curr_centroids):
            if ci not in used_curr:
                new_tracked[next_id] = (cx, cy)
                next_id += 1

        tracked = new_tracked

        # --- Detect line crossings (skip warmup) ---
        if frame_count > WARMUP_FRAMES:
            for obj_id, (cx, cy) in tracked.items():
                if obj_id in prev_tracked and obj_id not in counted_ids:
                    prev_y = prev_tracked[obj_id][1]

                    if prev_y < LINE_Y and cy >= LINE_Y:
                        count_down += 1
                        counted_ids.add(obj_id)
                    elif prev_y > LINE_Y and cy <= LINE_Y:
                        count_up += 1
                        counted_ids.add(obj_id)

        # --- Draw centroids with IDs ---
        for obj_id, (cx, cy) in tracked.items():
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(obj_id), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Draw counting line ---
        cv2.line(frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (frame_width // 2 - 70, LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Show mask preview ---
        mask_small = cv2.resize(fg_mask, (160, 120))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[0:120, frame_width - 160:frame_width] = mask_bgr

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Counts display
        cv2.putText(frame, f"Down: {count_down}", (10, LINE_Y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Up: {count_up}", (10, LINE_Y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Total: {count_up + count_down}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if frame_count <= WARMUP_FRAMES:
            cv2.putText(frame, f"Warming up... {frame_count}/{WARMUP_FRAMES}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 150, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live People Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final counts — Down: {count_down}, Up: {count_up}, Total: {count_up + count_down}")
```
