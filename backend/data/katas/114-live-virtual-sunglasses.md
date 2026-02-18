---
slug: 114-live-virtual-sunglasses
title: "Live Virtual Sunglasses"
level: live
concepts: [cv2.CascadeClassifier, eye detection, alpha blending, overlay positioning, image composition]
prerequisites: [106-live-eye-smile-detection, 13-image-blending]
---

## What Problem Are We Solving?

Augmented reality face filters -- sunglasses, hats, mustaches, animal ears -- are ubiquitous in modern video apps. The underlying technique is deceptively simple: detect facial landmarks (eyes, nose, mouth), compute where the virtual accessory should go, and composite it onto the frame with proper transparency. The challenging part is making it feel natural: the overlay must track the face smoothly, scale proportionally, and blend without hard edges.

This kata focuses on virtual sunglasses. You will detect eyes using a Haar cascade, compute the sunglasses geometry from the detected eye positions, create the sunglasses image programmatically using OpenCV drawing functions (no external file needed), and alpha-blend it onto the live camera feed. The same pipeline generalizes to any face accessory overlay.

Understanding alpha blending and coordinate mapping between ROI-relative and frame-absolute positions is essential for any overlay-based AR application. These skills transfer directly to more advanced pipelines that use DNN-based facial landmark detectors (like MediaPipe or dlib) instead of Haar cascades.

## Eye Detection with Haar Cascades

OpenCV ships with a pre-trained Haar cascade for eye detection. For best results, run eye detection **within a detected face region** rather than on the full frame. This dramatically reduces false positives (random textures being classified as eyes).

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
```

### Two-Stage Detection Pipeline

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

for (fx, fy, fw, fh) in faces:
    # Restrict eye search to upper 60% of face (eyes are never in the chin area)
    face_upper = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(face_upper, 1.1, 5, minSize=(20, 20))
```

### eye_cascade.detectMultiScale Parameters

| Parameter | Type | Description |
|---|---|---|
| `image` | ndarray | Grayscale image (face ROI for better accuracy) |
| `scaleFactor` | float | Image size reduction per pyramid level. `1.1` is the standard default |
| `minNeighbors` | int | Higher values reduce false positives but risk missing real eyes |
| `minSize` | tuple | Minimum eye size in pixels. `(20, 20)` filters out tiny false detections |
| `maxSize` | tuple | Maximum eye size. Prevents nostrils or mouth being detected as eyes |

**Critical:** Eye coordinates returned by `detectMultiScale` on the face ROI are **relative to that ROI**, not the full frame. You must add the face offset `(fx, fy)` before drawing on the full frame:

```python
for (ex, ey, ew, eh) in eyes:
    abs_x = fx + ex   # Convert to full-frame x
    abs_y = fy + ey   # Convert to full-frame y
```

### Why Restrict to the Upper 60%?

Eyes are always in the upper portion of a face bounding box. By cropping the face ROI to only the top 60%, you eliminate false detections from nostrils, mouth corners, and chin shadows that sometimes match the eye cascade pattern. This single optimization can cut false positives by more than half.

## Computing Sunglasses Geometry from Eye Positions

Once you have two detected eyes, you need to compute where the sunglasses should go and how large they should be:

```python
if len(eyes) >= 2:
    # Sort by x-coordinate: left eye first
    eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
    (e1x, e1y, e1w, e1h) = eyes_sorted[0]  # Left eye
    (e2x, e2y, e2w, e2h) = eyes_sorted[1]  # Right eye

    # Eye centers (relative to face ROI)
    c1x, c1y = e1x + e1w // 2, e1y + e1h // 2
    c2x, c2y = e2x + e2w // 2, e2y + e2h // 2

    # Sunglasses region with padding
    padding = int(e1w * 0.5)
    sg_x = fx + e1x - padding                         # Absolute left edge
    sg_y = fy + min(e1y, e2y) - padding // 2           # Absolute top edge
    sg_w = (e2x + e2w + padding) - (e1x - padding)    # Total width
    sg_h = max(e1h, e2h) + padding                     # Total height
```

### Geometry Breakdown

| Measurement | Formula | Purpose |
|---|---|---|
| Left edge | `e1x - padding` | Extends left beyond the left eye for a natural frame look |
| Right edge | `e2x + e2w + padding` | Extends right beyond the right eye |
| Total width | Right edge minus left edge | Spans both eyes plus padding on each side |
| Top edge | `min(e1y, e2y) - padding // 2` | Aligns to the higher eye with a small top margin |
| Height | `max(e1h, e2h) + padding` | Tall enough to cover both eyes fully |

The padding factor (50% of eye width) is empirically chosen to produce sunglasses that look proportional. Too little padding and the lenses barely cover the eyes; too much and the sunglasses look comically oversized.

## Creating Sunglasses Programmatically with OpenCV Drawing

Instead of loading an image file, we create the sunglasses from scratch using OpenCV drawing primitives. This keeps the kata self-contained and teaches BGRA image composition:

```python
def create_sunglasses(width, height):
    """Create a 4-channel (BGRA) sunglasses image."""
    sg = np.zeros((height, width, 4), dtype=np.uint8)

    lens_h = int(height * 0.65)
    lens_w = int(width * 0.38)
    y_off = int(height * 0.18)

    # Left lens: dark fill with semi-transparency
    lx = int(width * 0.06)
    cv2.ellipse(sg, (lx + lens_w // 2, y_off + lens_h // 2),
                (lens_w // 2, lens_h // 2), 0, 0, 360,
                (30, 30, 30, 210), -1)
    cv2.ellipse(sg, (lx + lens_w // 2, y_off + lens_h // 2),
                (lens_w // 2, lens_h // 2), 0, 0, 360,
                (10, 10, 10, 255), 2)

    # Right lens
    rx = width - lx - lens_w
    cv2.ellipse(sg, (rx + lens_w // 2, y_off + lens_h // 2),
                (lens_w // 2, lens_h // 2), 0, 0, 360,
                (30, 30, 30, 210), -1)
    cv2.ellipse(sg, (rx + lens_w // 2, y_off + lens_h // 2),
                (lens_w // 2, lens_h // 2), 0, 0, 360,
                (10, 10, 10, 255), 2)

    # Bridge connecting the lenses
    bridge_y = y_off + lens_h // 3
    cv2.line(sg, (lx + lens_w, bridge_y), (rx, bridge_y),
             (10, 10, 10, 255), max(2, height // 15))

    return sg
```

### Why BGRA (4 Channels)?

Standard OpenCV images use 3 channels (BGR). The fourth channel -- **alpha** -- controls per-pixel transparency:

| Alpha Value | Meaning |
|---|---|
| `255` | Fully opaque -- only the overlay pixel shows |
| `0` | Fully transparent -- only the background shows |
| `210` | Mostly opaque with slight see-through (the lens tint effect) |
| `128` | 50/50 blend of overlay and background |

By using `alpha = 210` for the lens fill, the dark sunglasses let a faint hint of the eyes show through, mimicking real tinted lenses. The frame outline uses `alpha = 255` for a solid border.

### Ellipse vs Rectangle Lenses

Using `cv2.ellipse` instead of `cv2.rectangle` produces rounded, aviator-style lenses that look more natural. The `cv2.ellipse` parameters:

| Parameter | Description |
|---|---|
| `center` | Center of the ellipse `(x, y)` |
| `axes` | Half-width and half-height `(rx, ry)` |
| `angle` | Rotation angle in degrees |
| `startAngle` / `endAngle` | Arc range (0 to 360 for full ellipse) |
| `color` | BGRA color tuple |
| `thickness` | `-1` for filled, positive for outline |

## Alpha Blending the Overlay onto the Frame

Alpha compositing merges the BGRA sunglasses onto the BGR camera frame using the alpha channel as a per-pixel mask:

```python
def overlay_rgba(bg, overlay, x, y):
    """Alpha-blend a BGRA overlay onto a BGR background at position (x, y)."""
    h, w = overlay.shape[:2]

    # Clip to frame boundaries (handle sunglasses extending past edges)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg.shape[1], x + w), min(bg.shape[0], y + h)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return

    cropped = overlay[oy1:oy2, ox1:ox2]
    alpha = cropped[:, :, 3] / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)

    roi = bg[y1:y2, x1:x2].astype(float)
    fg = cropped[:, :, :3].astype(float)
    bg[y1:y2, x1:x2] = (fg * alpha3 + roi * (1.0 - alpha3)).astype(np.uint8)
```

### The Blending Formula

For each pixel: `result = foreground * alpha + background * (1 - alpha)`

| Alpha | Result |
|---|---|
| 1.0 (fully opaque) | Only the overlay pixel is visible |
| 0.0 (fully transparent) | Only the background pixel is visible |
| 0.82 (210/255, the lens tint) | 82% overlay + 18% background, a dark tint effect |
| 0.5 | Equal mix, a ghostly overlay |

### Why Boundary Clipping Is Essential

When the person moves to the edge of the frame, the computed sunglasses region may extend beyond the image boundaries. Without clipping, NumPy array slicing would either silently produce a zero-size slice or crash. The clipping logic computes the valid overlap region in both the background and overlay coordinate systems.

## Coordinate System Summary

Understanding coordinate transforms is the most error-prone part of this pipeline:

```
Full Frame (640x480)
  |
  +-- face_cascade.detectMultiScale -> (fx, fy, fw, fh)
       |
       +-- face_upper = gray[fy:fy+int(fh*0.6), fx:fx+fw]
            |
            +-- eye_cascade.detectMultiScale -> (ex, ey, ew, eh)
                 |
                 +-- Absolute position: (fx + ex, fy + ey)
```

Every eye coordinate `(ex, ey)` is relative to `face_upper`, which starts at `(fx, fy)` in the full frame. Forgetting to add the face offset is the single most common bug in face-accessory overlays.

## Tips & Common Mistakes

- **Eye detection is noisy.** The cascade may detect 0, 1, 3, or more "eyes" per face. Always check `len(eyes) >= 2` and take only the first two after sorting by x-coordinate.
- **Filter false eyes** by restricting detection to the upper 60% of the face ROI. Eyes are never in the lower chin region, but nostrils and mouth corners often trigger the eye cascade.
- **Coordinate systems:** Eye positions from `detectMultiScale` on the face ROI are relative to the ROI. Always add `(fx, fy)` before drawing or placing overlays on the full frame.
- **Boundary clipping is essential.** If the sunglasses extend beyond the frame edges (person near the edge of camera), unclipped array slicing will crash or produce garbage.
- **Smooth jittery detection** by averaging eye positions across frames using an exponential moving average. This prevents the sunglasses from jumping around frame to frame.
- Use `cv2.INTER_AREA` when shrinking the sunglasses (higher quality downsampling) and `cv2.INTER_LINEAR` when enlarging.
- The bridge line thickness should scale with sunglasses size. A fixed thickness looks wrong on very large or very small detections.
- Creating the sunglasses image every frame is wasteful if the size has not changed. Cache the last size and only regenerate when dimensions change.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Face the camera straight-on so both eyes are visible — dark elliptical sunglasses with a bridge should appear over your eyes
- Move your head slowly side to side and up/down — the sunglasses should follow your face and scale with distance
- Check the FPS and face count displayed in the top-left corner to confirm real-time performance

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

def create_sunglasses(width, height):
    """Create a BGRA sunglasses image using ellipse lenses and a bridge."""
    sg = np.zeros((height, width, 4), dtype=np.uint8)
    lens_h = int(height * 0.65)
    lens_w = int(width * 0.38)
    y_off = int(height * 0.18)

    # Left lens (elliptical, dark with slight transparency)
    lx = int(width * 0.06)
    l_center = (lx + lens_w // 2, y_off + lens_h // 2)
    l_axes = (lens_w // 2, lens_h // 2)
    cv2.ellipse(sg, l_center, l_axes, 0, 0, 360, (30, 30, 30, 210), -1)
    cv2.ellipse(sg, l_center, l_axes, 0, 0, 360, (10, 10, 10, 255), 2)

    # Right lens
    rx = width - lx - lens_w
    r_center = (rx + lens_w // 2, y_off + lens_h // 2)
    r_axes = (lens_w // 2, lens_h // 2)
    cv2.ellipse(sg, r_center, r_axes, 0, 0, 360, (30, 30, 30, 210), -1)
    cv2.ellipse(sg, r_center, r_axes, 0, 0, 360, (10, 10, 10, 255), 2)

    # Bridge between lenses
    bridge_y = y_off + lens_h // 3
    cv2.line(sg, (lx + lens_w, bridge_y), (rx, bridge_y),
             (10, 10, 10, 255), max(2, height // 15))

    return sg

def overlay_rgba(bg, overlay, x, y):
    """Alpha-blend a BGRA overlay onto a BGR background."""
    h, w = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg.shape[1], x + w), min(bg.shape[0], y + h)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return

    cropped = overlay[oy1:oy2, ox1:ox2]
    alpha = cropped[:, :, 3] / 255.0
    alpha3 = np.stack([alpha] * 3, axis=-1)

    roi = bg[y1:y2, x1:x2].astype(float)
    fg = cropped[:, :, :3].astype(float)
    bg[y1:y2, x1:x2] = (fg * alpha3 + roi * (1.0 - alpha3)).astype(np.uint8)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)
cached_sg = None
cached_sg_size = (0, 0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (fx, fy, fw, fh) in faces:
            # Detect eyes in the upper 60% of the face
            face_upper = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(face_upper, 1.1, 5, minSize=(20, 20))

            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
                (e1x, e1y, e1w, e1h) = eyes_sorted[0]
                (e2x, e2y, e2w, e2h) = eyes_sorted[1]

                # Compute sunglasses position (absolute frame coordinates)
                padding = int(e1w * 0.5)
                sg_x = fx + e1x - padding
                sg_y = fy + min(e1y, e2y) - padding // 2
                sg_w = (e2x + e2w + padding) - (e1x - padding)
                sg_h = max(e1h, e2h) + padding

                if sg_w > 10 and sg_h > 5:
                    # Cache sunglasses image if size unchanged
                    if (sg_w, sg_h) != cached_sg_size:
                        cached_sg = create_sunglasses(sg_w, sg_h)
                        cached_sg_size = (sg_w, sg_h)
                    overlay_rgba(frame, cached_sg, sg_x, sg_y)

        # --- HUD overlay ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Virtual Sunglasses', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
