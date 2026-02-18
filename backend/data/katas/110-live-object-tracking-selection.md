---
slug: 110-live-object-tracking-selection
title: Live Object Tracking with Selection
level: live
concepts: [cv2.TrackerCSRT, ROI selection, cv2.selectROI, real-time tracking, tracker re-initialization]
prerequisites: [100-live-camera-fps, 77-object-tracking-csrt]
---

## What Problem Are We Solving?

Color-based tracking only works when your object has a distinctive color. But what if you want to track an arbitrary object — a person's face, a car, a coffee mug, or anything else in the scene? You need a **general-purpose object tracker** that can follow any visual pattern you select. OpenCV provides several tracker algorithms that do exactly this: you draw a bounding box around the object you want to track, and the tracker follows it frame to frame using visual features like texture, edges, and spatial relationships.

This kata builds a complete interactive tracking system: press a key to pause the feed and select a region of interest (ROI), then track that object in real-time. When the tracker loses the object (occlusion, leaving the frame), the system reports the loss and allows you to re-select a new target. This is the same fundamental approach used in video surveillance, sports analysis, and drone follow-me modes.

The main challenge is choosing the right tracker. OpenCV offers several algorithms with different speed/accuracy trade-offs, and understanding when each is appropriate is a key skill.

## OpenCV Tracker Algorithms Compared

OpenCV provides multiple tracker implementations through its `cv2.Tracker*_create()` factory methods:

| Tracker | Speed | Accuracy | Occlusion Handling | Best For |
|---|---|---|---|---|
| **CSRT** | Slow (25-40 FPS) | Highest | Good | When accuracy matters more than speed |
| **KCF** | Fast (60-80 FPS) | Good | Poor | Real-time applications with simple motion |
| **MOSSE** | Fastest (100+ FPS) | Low | Poor | Extremely fast tracking, drone applications |
| **MIL** | Medium | Medium | Medium | Learning-based, handles appearance changes |
| **MedianFlow** | Fast | Medium | Detects failure | When you need failure detection |

### Creating a Tracker

```python
# CSRT — best accuracy, recommended default
tracker = cv2.TrackerCSRT_create()

# KCF — good balance of speed and accuracy
tracker = cv2.TrackerKCF_create()

# MOSSE — fastest, lowest accuracy
tracker = cv2.legacy.TrackerMOSSE_create()
```

> **Note:** In OpenCV 4.5+, some trackers moved to `cv2.legacy`. If `cv2.TrackerMOSSE_create()` fails, try `cv2.legacy.TrackerMOSSE_create()`.

## CSRT Tracker Internals

CSRT (Channel and Spatial Reliability Tracking) is the most accurate tracker bundled with OpenCV. It works by:

1. **Learning a discriminative correlation filter** on the selected ROI — this filter distinguishes the target from the background.
2. **Using channel reliability weights** — it evaluates which color channels (and HOG features) are most discriminative for this particular object and weights them accordingly.
3. **Applying a spatial reliability map** — it learns which parts of the bounding box reliably belong to the target (center) vs background (edges), handling non-rectangular objects better than other trackers.

The correlation filter approach means CSRT does not need to search the entire frame — it evaluates a local region around the predicted position, making it efficient despite its accuracy.

### CSRT Limitations

- **No re-detection**: If the object leaves the frame entirely and returns, CSRT will not find it again. You must re-initialize.
- **Scale drift**: Over long sequences, the bounding box may gradually grow or shrink.
- **Deformation**: Highly deformable objects (cloth, liquid) cause tracking failure.

## Selecting the ROI with cv2.selectROI

`cv2.selectROI` pauses the video feed and lets the user draw a bounding box:

```python
bbox = cv2.selectROI("Window Name", frame, fromCenter=False, showCrosshair=True)
```

| Parameter | Type | Meaning |
|---|---|---|
| `windowName` | str | Name of the display window |
| `frame` | ndarray | The frame to select ROI on |
| `fromCenter` | bool | If True, draw from center outward; if False, draw from corner |
| `showCrosshair` | bool | Show crosshair guides while selecting |
| **Returns** | tuple | `(x, y, w, h)` of selected rectangle, or `(0,0,0,0)` if cancelled |

The user draws by clicking and dragging. Press **Enter** or **Space** to confirm, **c** to cancel.

> **Important:** `selectROI` returns `(0, 0, 0, 0)` if the user cancels (presses 'c' or Escape). Always check for this before initializing the tracker.

## Tracker Initialization and Update

```python
# Initialize tracker with the selected bounding box
tracker.init(frame, bbox)

# In the main loop — update tracker
success, bbox = tracker.update(frame)
```

| Method | Parameters | Returns |
|---|---|---|
| `tracker.init(frame, bbox)` | First frame + bounding box tuple | None |
| `tracker.update(frame)` | Current frame | `(success, bbox)` where success is bool |

When `success` is `False`, the tracker has lost the object. The bounding box returned may be garbage — do not draw it.

## Re-initializing the Tracker

Once a tracker loses its target, it cannot recover. You must create a **new** tracker instance and re-initialize:

```python
if not success:
    # Tracker lost the object — wait for user to re-select
    status = "Lost"

# When user presses 's' to re-select:
tracker = cv2.TrackerCSRT_create()  # Must create a NEW instance
bbox = cv2.selectROI("Window", frame, fromCenter=False)
if bbox != (0, 0, 0, 0):
    tracker.init(frame, bbox)
```

> **Critical:** You cannot call `tracker.init()` again on the same tracker instance after it has been initialized. You must create a new `cv2.TrackerCSRT_create()` object. This is a common mistake that causes silent failures.

## Drawing the Tracking State

Clearly communicate whether tracking is active, lost, or waiting for selection:

```python
if success:
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, "Tracking", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    cv2.putText(frame, "Lost - press 's' to re-select", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

## Performance Tips

| Optimization | Effect | Implementation |
|---|---|---|
| Resize frame before tracking | 2-3x speedup | `small = cv2.resize(frame, (320, 240))` then scale bbox back |
| Use KCF instead of CSRT | 2x faster | Trade accuracy for speed |
| Use MOSSE for drones | 4-5x faster than CSRT | When speed is critical |
| Skip frames | Linear speedup | Track every 2nd frame, interpolate |

## Tips & Common Mistakes

- **Create a new tracker for each re-initialization.** Calling `init()` on a previously used tracker does not reset it properly and leads to erratic behavior.
- `cv2.selectROI` blocks the program until the user confirms or cancels. This is expected behavior, not a bug.
- If `selectROI` returns `(0, 0, 0, 0)`, the user cancelled. Do not pass this to `tracker.init()`.
- The CSRT tracker works best when the initial bounding box tightly fits the object. Too much background in the selection degrades performance.
- Tracking FPS depends on the bounding box size. A larger ROI means more computation per update.
- CSRT handles moderate scale changes but struggles with extreme zoom-in/zoom-out. Re-select if the object's apparent size changes dramatically.
- All OpenCV trackers are **single-object** trackers. For multi-object tracking, create multiple tracker instances (or use `cv2.MultiTracker`).
- The tracker does not know what it is tracking. It follows visual patterns, so if a similar-looking object crosses the target, the tracker may switch.
- On older OpenCV versions (< 4.5), trackers are in `cv2.TrackerCSRT_create()`. On newer versions, some moved to `cv2.legacy`.

## Starter Code

```python
import cv2
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

WINDOW_NAME = 'Live Object Tracking'

# --- Tracker state ---
tracker = None
tracking = False
bbox = None

def create_tracker():
    """Create a new CSRT tracker instance."""
    return cv2.TrackerCSRT_create()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Update tracker if active ---
        if tracking and tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                # Draw bounding box (green)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                # Draw center point
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # Show dimensions
                cv2.putText(frame, f"{w}x{h}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                tracking = False
                cv2.putText(frame, "Tracking LOST", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 's' to re-select target", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 1, cv2.LINE_AA)

        # --- Status display ---
        if not tracking:
            cv2.putText(frame, "Press 's' to select object to track",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        state_text = "TRACKING" if tracking else "IDLE"
        state_color = (0, 255, 0) if tracking else (100, 100, 100)
        cv2.putText(frame, f"State: {state_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "'s': select  'c': clear  'q': quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Pause and select ROI
            roi = cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                tracker = create_tracker()
                tracker.init(frame, roi)
                tracking = True
                bbox = roi
            else:
                print("Selection cancelled")
        elif key == ord('c'):
            # Clear tracker
            tracker = None
            tracking = False
            bbox = None

finally:
    cap.release()
    cv2.destroyAllWindows()
```
