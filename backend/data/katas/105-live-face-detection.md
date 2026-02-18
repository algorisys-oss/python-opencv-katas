---
slug: 105-live-face-detection
title: Live Face Detection & Counting
level: live
concepts: [cv2.CascadeClassifier, detectMultiScale, Haar cascades, face counting, real-time detection]
prerequisites: [100-live-camera-fps, 80-haar-cascade-face]
---

## What Problem Are We Solving?

Face detection is one of the most common real-time computer vision tasks — security cameras, video conferencing apps, phone unlock screens, and social media filters all need to find faces in a live video feed. The challenge is not just detecting faces **accurately**, but doing it **fast enough** to run smoothly at 20-30 FPS on a webcam feed.

OpenCV ships with pre-trained Haar cascade classifiers that can detect faces without any deep learning framework. While modern DNN-based detectors are more accurate, Haar cascades remain relevant because they are fast, lightweight, require no GPU, and work out of the box with zero setup. This kata teaches you to run face detection on a live camera feed, tune the detection parameters for your scene, and overlay face count and bounding boxes in real-time.

## How Haar Cascade Detection Works

A Haar cascade classifier is a machine-learning-based detector trained on thousands of positive (face) and negative (non-face) images. It works by sliding a window across the image at multiple scales and evaluating a cascade of simple features (edge patterns, line patterns, rectangle patterns) at each position.

The "cascade" part is key to performance: the classifier is organized as a series of stages, each containing a few weak classifiers. If a window fails any stage, it is immediately rejected without evaluating the remaining stages. Since most windows in an image are **not** faces, the cascade rejects them quickly — typically in the first 2-3 stages — making detection very fast.

```python
# Load the pre-trained cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

The `cv2.data.haarcascades` path points to the directory where OpenCV stores its bundled cascade XML files. This works across all platforms without hardcoding paths.

### Available Face Cascades

| Cascade File | Detects | Notes |
|---|---|---|
| `haarcascade_frontalface_default.xml` | Front-facing faces | Best general-purpose choice |
| `haarcascade_frontalface_alt.xml` | Front-facing faces | Fewer false positives, slightly slower |
| `haarcascade_frontalface_alt2.xml` | Front-facing faces | Good balance of speed and accuracy |
| `haarcascade_frontalface_alt_tree.xml` | Front-facing faces | Tree-based, different error profile |
| `haarcascade_profileface.xml` | Side-profile faces | For faces turned ~90 degrees |

## The detectMultiScale Function

This is the core detection call. It scans the image at multiple scales and returns bounding boxes for all detected faces:

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `image` | ndarray | (required) | Input image, **must be grayscale** |
| `scaleFactor` | float | 1.1 | How much the image is shrunk at each scale step. `1.1` means 10% reduction per step. Smaller values (e.g., `1.05`) are more thorough but slower |
| `minNeighbors` | int | 3 | How many overlapping detections are required to confirm a face. Higher values reduce false positives but may miss faces |
| `minSize` | tuple | (30, 30) | Minimum face size in pixels. Faces smaller than this are ignored |
| `maxSize` | tuple | None | Maximum face size. Useful to ignore very large close-up detections |
| `flags` | int | 0 | `cv2.CASCADE_SCALE_IMAGE` is the standard flag |

The return value `faces` is a NumPy array of shape `(N, 4)` where each row is `(x, y, w, h)` — the top-left corner and dimensions of a detected face. If no faces are found, it returns an empty tuple `()`.

### Tuning scaleFactor

`scaleFactor` controls the image pyramid — how much the image is downscaled between detection passes:

```
scaleFactor=1.01  → 1% reduction per step → very thorough, very slow
scaleFactor=1.1   → 10% reduction per step → good default
scaleFactor=1.3   → 30% reduction per step → fast but may miss faces between scales
scaleFactor=1.5   → 50% reduction per step → very fast, may miss many faces
```

Smaller values create more pyramid levels, so detection takes longer but is less likely to miss faces at intermediate sizes.

### Tuning minNeighbors

Each candidate window gets a "hit count" based on how many overlapping windows also detected a face at that location. `minNeighbors` sets the threshold:

```
minNeighbors=0  → Every candidate is returned → many false positives
minNeighbors=3  → Moderate filtering → some false positives
minNeighbors=5  → Strict filtering → fewer false positives, may miss faint faces
minNeighbors=8  → Very strict → only high-confidence detections
```

For live video, `minNeighbors=5` is a good starting point.

## Performance Optimization: Resize Before Detection

Running `detectMultiScale` on a full 1280x720 or 1920x1080 frame is slow. The standard optimization is to **downscale the frame before detection** and then scale the bounding box coordinates back up for drawing:

```python
# Downscale for detection
scale = 0.5
small = cv2.resize(gray, None, fx=scale, fy=scale)

# Detect on the smaller image
faces = face_cascade.detectMultiScale(small, 1.1, 5, minSize=(20, 20))

# Scale coordinates back to original size
for (x, y, w, h) in faces:
    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

Typical speedups:

| Frame Size | Detection Time | With 0.5x Resize |
|---|---|---|
| 1920x1080 | ~80-120 ms | ~20-30 ms |
| 1280x720 | ~40-60 ms | ~10-15 ms |
| 640x480 | ~15-25 ms | ~5-8 ms |

The trade-off: smaller detection image means smaller faces may be missed (because they fall below `minSize` after downscaling). Adjust `minSize` accordingly.

## Drawing Bounding Boxes and Face Count

For each detected face, draw a colored rectangle and label it:

```python
for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"Face {i+1}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Face count overlay
count = len(faces) if isinstance(faces, np.ndarray) else 0
cv2.putText(frame, f"Faces: {count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
```

> **Important:** `detectMultiScale` returns an empty tuple `()` when no faces are found, not an empty NumPy array. Always check with `len(faces)` or `isinstance(faces, np.ndarray)` before iterating.

## Tips & Common Mistakes

- **Always convert to grayscale** before calling `detectMultiScale`. Passing a color image causes an error or garbage results.
- Use `cv2.data.haarcascades + 'filename.xml'` to load cascades portably. Hardcoding paths like `/usr/share/opencv/...` breaks on other systems.
- If you get zero detections, try lowering `minNeighbors` to 3 or reducing `minSize`. The default parameters are conservative.
- If you get too many false positives (rectangles on walls, patterns), increase `minNeighbors` to 6-8.
- Haar cascades only detect **frontal** faces reliably. Tilted or profile faces need different cascades or a DNN detector.
- Resizing the frame before detection is the single biggest performance win. Always do it for frames larger than 640x480.
- The face cascade expects **upright** faces. If your camera is rotated, rotate the frame first.
- `detectMultiScale` is CPU-bound. On multi-core systems, OpenCV may use threading internally, but you cannot control it directly.
- Lighting matters significantly. Haar cascades struggle in very dark or heavily backlit scenes.

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Load Haar cascade ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("Error: Could not load face cascade")
    exit()

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)

# Detection parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)
DETECT_SCALE = 0.5  # Resize factor for faster detection

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Convert to grayscale and downscale for detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE)

        # --- Detect faces ---
        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=(int(MIN_FACE_SIZE[0] * DETECT_SCALE),
                     int(MIN_FACE_SIZE[1] * DETECT_SCALE))
        )

        # --- Draw bounding boxes (scale coordinates back up) ---
        face_count = 0
        if isinstance(faces, np.ndarray):
            face_count = len(faces)
            for i, (x, y, w, h) in enumerate(faces):
                # Scale back to original frame coordinates
                x = int(x / DETECT_SCALE)
                y = int(y / DETECT_SCALE)
                w = int(w / DETECT_SCALE)
                h = int(h / DETECT_SCALE)

                # Green rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Label each face
                cv2.putText(frame, f"Face {i + 1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {face_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
