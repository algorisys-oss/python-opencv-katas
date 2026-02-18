---
slug: 113-live-face-blur
title: "Live Face Blur (Privacy Filter)"
level: live
concepts: [cv2.CascadeClassifier, cv2.GaussianBlur, pixel mosaic, ROI extraction, face anonymization]
prerequisites: [105-live-face-detection, 21-gaussian-blur]
---

## What Problem Are We Solving?

Privacy is a critical concern in any application that captures live video. Whether you are building a security camera dashboard, a live-streaming tool, or a data-collection pipeline for machine learning, you often need to **obscure faces** before storing or transmitting footage. Regulations like GDPR explicitly require anonymization of identifiable individuals in many contexts.

The two most common face-anonymization techniques are **Gaussian blur** and **pixelation (mosaic)**. Gaussian blur smooths the face region until features are unrecognizable, while pixelation downscales and then upscales the region to create the characteristic blocky mosaic look. Each has trade-offs: blur looks more natural but can sometimes be reversed with deconvolution algorithms; pixelation is harder to reverse and is the standard in broadcast TV.

This kata teaches you to detect faces in real-time using a Haar cascade classifier, extract the face region of interest (ROI), apply either blur or pixelation, and toggle between modes interactively.

## Haar Cascade Face Detection Recap

OpenCV ships with pre-trained Haar cascade XML files. The frontal face detector is the most commonly used:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

Detection is performed on a grayscale image:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

### detectMultiScale Parameters

| Parameter | Type | Description |
|---|---|---|
| `image` | ndarray | Input grayscale image |
| `scaleFactor` | float | Image size reduction at each scale (1.1 = 10% reduction). Smaller values are more thorough but slower |
| `minNeighbors` | int | Minimum number of neighbor rectangles required for a detection to be kept. Higher values reduce false positives |
| `minSize` | tuple | Minimum face size in pixels `(w, h)`. Faces smaller than this are ignored |
| `maxSize` | tuple | Maximum face size in pixels. Faces larger than this are ignored |

The function returns a list of `(x, y, w, h)` rectangles, one per detected face.

## Gaussian Blur on a Face ROI

To blur only the face region, extract it using NumPy slicing, blur it, and write it back:

```python
for (x, y, w, h) in faces:
    face_roi = frame[y:y+h, x:x+w]
    blurred_face = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
    frame[y:y+h, x:x+w] = blurred_face
```

### GaussianBlur Parameters

| Parameter | Type | Description |
|---|---|---|
| `src` | ndarray | Input image (the face ROI) |
| `ksize` | tuple | Kernel size `(width, height)` — must be positive and odd. Larger = more blur |
| `sigmaX` | float | Standard deviation in X. If 0, computed from kernel size |

**How large should the kernel be?** A good heuristic is to scale it with the face size:

```python
ksize = max(51, (w // 3) | 1)  # At least 51, always odd, proportional to face width
```

This ensures small faces get enough blur to be unrecognizable, and large faces get proportionally stronger blur.

## Pixelation (Mosaic) Effect

Pixelation is achieved by downscaling and then upscaling the ROI with nearest-neighbor interpolation:

```python
for (x, y, w, h) in faces:
    face_roi = frame[y:y+h, x:x+w]
    # Downscale to tiny size
    small = cv2.resize(face_roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    # Upscale back to original size with nearest-neighbor (blocky)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = pixelated
```

### Pixelation Size Parameter

| `pixel_size` Value | Effect |
|---|---|
| 3-5 | Extremely pixelated — barely any structure visible |
| 8-12 | Standard broadcast-style mosaic |
| 16-20 | Mild mosaic — features still partially visible |
| 25+ | Barely pixelated — not sufficient for anonymization |

Lower values mean fewer pixels in the downscaled version, which means larger blocks and stronger anonymization.

## Toggling Between Modes

Use a simple integer state variable and cycle through modes with a keypress:

```python
blur_mode = 0  # 0 = Gaussian blur, 1 = pixelation

key = cv2.waitKey(1) & 0xFF
if key == ord('b'):
    blur_mode = (blur_mode + 1) % 2
```

You can extend this to more modes (e.g., black rectangle, color fill) by increasing the modulus.

## Adjustable Blur Strength

Allow the user to increase or decrease blur strength with keyboard controls:

```python
blur_strength = 5  # Multiplier for kernel size or inverse of pixel_size

if key == ord('+') or key == ord('='):
    blur_strength = min(blur_strength + 1, 15)
elif key == ord('-'):
    blur_strength = max(blur_strength - 1, 1)
```

Map this strength to actual parameters:
- **Gaussian mode:** `ksize = blur_strength * 10 + 1` (ensures odd)
- **Pixelation mode:** `pixel_size = max(3, 16 - blur_strength)` (higher strength = fewer pixels = more blocky)

## Performance Considerations

| Operation | Typical Time (640x480) |
|---|---|
| `cvtColor` to grayscale | < 1 ms |
| `detectMultiScale` | 15-40 ms (depends on scaleFactor) |
| `GaussianBlur` on face ROI | < 1 ms per face |
| `resize` down + up (pixelation) | < 1 ms per face |

The bottleneck is always the cascade detection. To improve FPS:
- Use `scaleFactor=1.2` or `1.3` for faster (but less accurate) detection
- Set `minSize=(60, 60)` to skip scanning for tiny faces
- Process every Nth frame for detection, but apply blur every frame using cached face positions

## Tips & Common Mistakes

- **Always convert to grayscale** before calling `detectMultiScale` — passing a color image silently produces worse results or errors.
- Make the blur kernel size **proportional to face size**. A fixed small kernel on a large face leaves features recognizable.
- Pixelation with `INTER_NEAREST` on the upscale is critical — using `INTER_LINEAR` produces a blurry result instead of the characteristic blocky mosaic.
- Check that `faces` is not empty before iterating. `detectMultiScale` returns an empty tuple when no faces are found, which is safe to iterate but can cause confusion if you check `len()` on a tuple vs. ndarray.
- The ROI slice `frame[y:y+h, x:x+w]` directly references frame memory. Modifying it with `=` writes back in-place, which is exactly what we want.
- If false positives are a problem (random regions getting blurred), increase `minNeighbors` to 6 or 7.
- For very strong privacy guarantees, use pixelation with `pixel_size <= 6` — Gaussian blur with moderate kernels has been shown to be partially reversible.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Move your face into the camera view — it should be automatically blurred with a thin red border around the detected region
- Press **b** to toggle between Gaussian blur and pixelation (mosaic) modes — the current mode is shown in the HUD
- Press **+** / **-** to increase or decrease blur strength and verify the effect changes visibly

## Starter Code

```python
import cv2
import time
from collections import deque

# --- Load Haar cascade ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)
blur_mode = 0       # 0 = Gaussian blur, 1 = pixelation
blur_strength = 5   # 1-10 scale

mode_names = ["Gaussian Blur", "Pixelation"]

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Detect faces ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # --- Apply blur/pixelation to each face ---
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            if blur_mode == 0:
                # Gaussian blur — kernel proportional to face size and strength
                ksize = max(11, (blur_strength * w // 5) | 1)
                blurred = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
                frame[y:y+h, x:x+w] = blurred
            else:
                # Pixelation — fewer pixels = more anonymized
                pixel_size = max(3, 18 - blur_strength * 2)
                small = cv2.resize(face_roi, (pixel_size, pixel_size),
                                   interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = pixelated

            # Draw subtle border around blurred region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        # --- HUD overlay ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {mode_names[blur_mode]} | Strength: {blur_strength}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "'b'=toggle mode  '+'/'-'=strength  'q'=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Face Blur', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            blur_mode = (blur_mode + 1) % 2
        elif key == ord('+') or key == ord('='):
            blur_strength = min(blur_strength + 1, 10)
        elif key == ord('-'):
            blur_strength = max(blur_strength - 1, 1)

finally:
    cap.release()
    cv2.destroyAllWindows()
```
