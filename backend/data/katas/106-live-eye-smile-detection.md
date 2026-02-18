---
slug: 106-live-eye-smile-detection
title: Live Eye & Smile Detection
level: live
concepts: [cv2.CascadeClassifier, nested cascades, ROI detection, eye detection, smile detection]
prerequisites: [105-live-face-detection, 81-haar-cascade-eyes]
---

## What Problem Are We Solving?

Detecting a face is only the first step. Many applications need to find **features within a face** — eyes for gaze tracking, smiles for emotion detection, or mouth position for lip-reading. The naive approach of running eye and smile detectors on the entire frame is both slow (scanning the whole image) and inaccurate (false positives from patterns that look like eyes or mouths in the background).

The solution is **nested cascade detection**: first detect the face, then search for eyes and smiles **only within the face region of interest (ROI)**. This dramatically reduces both computation time and false positive rates, because you are searching a small, relevant area instead of the entire frame.

## Nested Detection Strategy

The key insight is that once you know where a face is, you know approximately where the eyes and mouth should be:

- **Eyes** are in the **upper half** of the face bounding box
- **Smiles/mouths** are in the **lower half** of the face bounding box

```
+-------------------+
|   Forehead area   |
|  +-----+ +-----+  |  ← Eyes are here (upper 60%)
|  | Eye | | Eye |  |
|  +-----+ +-----+  |
|                   |
|    +----------+   |  ← Smile is here (lower 50%)
|    |  Smile   |   |
|    +----------+   |
+-------------------+
```

By cropping the face ROI and further restricting the search region, you get faster and more accurate results.

## Loading Multiple Cascades

OpenCV bundles several cascade files for different facial features:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
```

| Cascade File | Detects | Typical Use |
|---|---|---|
| `haarcascade_eye.xml` | Open eyes | General eye detection |
| `haarcascade_eye_tree_eyeglasses.xml` | Eyes with/without glasses | Better for glasses wearers |
| `haarcascade_smile.xml` | Smiles/mouths | Smile detection within face ROI |
| `haarcascade_lefteye_2splits.xml` | Left eye specifically | When you need to distinguish L/R |
| `haarcascade_righteye_2splits.xml` | Right eye specifically | When you need to distinguish L/R |

> **Important:** Always verify each cascade loaded successfully with `cascade.empty()`. A typo in the filename returns an empty classifier that silently detects nothing.

## Extracting the Face ROI

When `detectMultiScale` returns a face at `(x, y, w, h)`, extract the grayscale ROI for nested detection:

```python
for (fx, fy, fw, fh) in faces:
    # Extract the face region from the grayscale image
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    face_color = frame[fy:fy+fh, fx:fx+fw]

    # Search for eyes in the upper 60% of the face
    eye_region_gray = face_gray[0:int(fh*0.6), :]
    eye_region_color = face_color[0:int(fh*0.6), :]

    # Search for smiles in the lower 50% of the face
    smile_region_gray = face_gray[int(fh*0.5):, :]
    smile_region_color = face_color[int(fh*0.5):, :]
```

This ROI extraction serves two purposes:
1. **Speed** — The eye detector scans a small region instead of the full frame
2. **Accuracy** — Background patterns that resemble eyes are excluded

## detectMultiScale Parameters for Eyes and Smiles

Each feature needs different tuning because of their different sizes and patterns:

### Eye Detection Parameters

```python
eyes = eye_cascade.detectMultiScale(
    eye_region_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20),
    maxSize=(fw // 3, fh // 4)
)
```

| Parameter | Recommended Value | Why |
|---|---|---|
| `scaleFactor` | 1.1 | Standard step size works well |
| `minNeighbors` | 5-7 | Eyes have many false positives; be strict |
| `minSize` | (20, 20) | Reject tiny false detections |
| `maxSize` | (fw//3, fh//4) | An eye cannot be larger than 1/3 of face width |

### Smile Detection Parameters

```python
smiles = smile_cascade.detectMultiScale(
    smile_region_gray,
    scaleFactor=1.7,
    minNeighbors=22,
    minSize=(25, 15)
)
```

| Parameter | Recommended Value | Why |
|---|---|---|
| `scaleFactor` | 1.5-1.8 | Smile cascade needs large scale jumps; smaller values produce too many false positives |
| `minNeighbors` | 20-25 | Smile detection is noisy; high threshold reduces false triggers |
| `minSize` | (25, 15) | Smiles are wide and short |

> **Why is the smile detector so different?** The smile cascade was trained on tightly cropped mouth regions and is inherently noisier than the eye cascade. It fires on many non-smile patterns. Compensate with aggressive `minNeighbors` filtering and restricted ROI.

## Drawing Annotations

Use distinct colors and shapes to differentiate features:

```python
# Face: green rectangle
cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

# Eyes: blue circles (centered on each detected eye)
for (ex, ey, ew, eh) in eyes:
    center_x = fx + ex + ew // 2
    center_y = fy + ey + eh // 2
    radius = max(ew, eh) // 2
    cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), 2)

# Smile: orange arc/rectangle in the lower face
for (sx, sy, sw, sh) in smiles:
    smile_y_offset = int(fh * 0.5)
    cv2.rectangle(frame, (fx + sx, fy + smile_y_offset + sy),
                  (fx + sx + sw, fy + smile_y_offset + sy + sh),
                  (0, 165, 255), 2)
```

> **Coordinate mapping:** Eye and smile coordinates are relative to the face ROI, not the full frame. You must add the face offset (`fx`, `fy`) plus any sub-region offset when drawing on the original frame.

## Performance Considerations

Nested detection multiplies the cost: for N faces, you run the eye and smile detectors N times each. Optimization strategies:

| Strategy | Speedup | Trade-off |
|---|---|---|
| Resize frame before face detection | 2-4x | May miss small faces |
| Restrict eye search to upper 60% of face | ~40% less area | None (eyes are always there) |
| Restrict smile search to lower 50% of face | ~50% less area | None |
| Increase `scaleFactor` for smile cascade | 2-3x faster | Slightly less precise |
| Skip detection every other frame | 2x | Detections lag by 1 frame |
| Limit to first face only | Linear with face count | Ignores additional faces |

## Tips & Common Mistakes

- **Coordinate offsets are the number one bug.** Eye coordinates are relative to the eye search region, which is relative to the face ROI, which is relative to the frame. Track each offset carefully.
- The smile cascade produces many false positives at low `minNeighbors`. Start with 20+ and reduce if smiles are not detected.
- `haarcascade_eye.xml` detects open eyes better than closed ones. For blink detection, consider tracking eye disappearance rather than closed-eye detection.
- If eyes are detected on eyebrows or nostrils, restrict the ROI more tightly or increase `minNeighbors`.
- The eye cascade may detect the nose bridge as two "eyes." Setting `maxSize` helps filter this out.
- For people wearing glasses, use `haarcascade_eye_tree_eyeglasses.xml` instead of the standard eye cascade.
- Drawing on the ROI slice directly modifies the original frame (NumPy views). This is fine for visualization but be aware if you need the unmodified frame later.
- Processing time scales with the number of faces. In a crowded scene, consider limiting detection to the first 3-4 faces.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Move your face into the frame — you should see a green rectangle around your face and blue circles on each detected eye
- Smile broadly — an orange "Smiling!" label and orange rectangle should appear around your mouth area
- Check the top-left counters for Faces, Eyes, and Smiles and verify they update as you change expression

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Load cascades ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

for name, cascade in [("Face", face_cascade), ("Eye", eye_cascade), ("Smile", smile_cascade)]:
    if cascade.empty():
        print(f"Error: Could not load {name} cascade")
        exit()

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Detect faces ---
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        eye_total = 0
        smile_total = 0

        if isinstance(faces, np.ndarray):
            for (fx, fy, fw, fh) in faces:
                # Draw face rectangle (green)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                # --- Eye detection in upper 60% of face ---
                eye_region_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(
                    eye_region_gray, 1.1, 7,
                    minSize=(20, 20),
                    maxSize=(fw // 3, fh // 4)
                )

                if isinstance(eyes, np.ndarray):
                    eye_total += len(eyes)
                    for (ex, ey, ew, eh) in eyes:
                        # Blue circle for each eye (map to frame coords)
                        cx = fx + ex + ew // 2
                        cy = fy + ey + eh // 2
                        radius = max(ew, eh) // 2
                        cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)

                # --- Smile detection in lower 50% of face ---
                smile_y_offset = int(fh * 0.5)
                smile_region_gray = gray[fy + smile_y_offset:fy + fh, fx:fx + fw]
                smiles = smile_cascade.detectMultiScale(
                    smile_region_gray, 1.7, 22,
                    minSize=(25, 15)
                )

                if isinstance(smiles, np.ndarray):
                    smile_total += len(smiles)
                    for (sx, sy, sw, sh) in smiles:
                        # Orange rectangle for smile (map to frame coords)
                        cv2.rectangle(
                            frame,
                            (fx + sx, fy + smile_y_offset + sy),
                            (fx + sx + sw, fy + smile_y_offset + sy + sh),
                            (0, 165, 255), 2
                        )
                        # Draw "Smiling!" label
                        cv2.putText(frame, "Smiling!", (fx, fy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 165, 255), 2, cv2.LINE_AA)

        # --- Overlays ---
        face_count = len(faces) if isinstance(faces, np.ndarray) else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {face_count}  Eyes: {eye_total}  Smiles: {smile_total}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Eye & Smile Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
