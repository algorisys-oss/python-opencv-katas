---
slug: 71-webcam-capture
title: Webcam Capture
level: advanced
concepts: [cv2.VideoCapture device, live feed, frame processing]
prerequisites: [70-reading-video-files]
---

## What Problem Are We Solving?

Reading from a video file gives you pre-recorded footage, but many computer vision applications need to process a **live camera feed** in real time — think face detection, gesture recognition, or augmented reality. OpenCV makes it easy to capture frames from a webcam using the same `cv2.VideoCapture` interface, just with a **device index** instead of a file path.

## Opening a Webcam

Pass an integer device index (typically `0` for the default camera) instead of a filename:

```python
cap = cv2.VideoCapture(0)
```

If you have multiple cameras, use `1`, `2`, etc. to select different devices.

On some systems, you may want to specify the capture backend:

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # Linux
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Windows
```

## The Live Feed Loop

The standard pattern for a webcam application is a loop that reads, processes, and displays frames:

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Live', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Key details:
- `cv2.waitKey(1)` waits 1 ms, keeping the display responsive.
- The `& 0xFF` mask ensures cross-platform compatibility for key codes.
- Press `q` to exit the loop cleanly.

## Setting Camera Properties

You can adjust camera resolution and other settings:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

Not all cameras support all settings. Check `cap.get()` to verify actual values:

```python
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
```

## Processing Each Frame

The power of the webcam loop is that you can apply **any** OpenCV operation to each frame in real time:

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Apply edge detection to live feed
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Flipping the Feed

Webcam images are often mirrored. Use `cv2.flip()` to correct this:

```python
frame = cv2.flip(frame, 1)  # 1 = horizontal flip (mirror)
```

## Tips & Common Mistakes

- Always check `cap.isOpened()` before entering the loop. If the camera is in use by another application, it may fail silently.
- Use `cv2.waitKey(1)` (not `cv2.waitKey(0)`) in the loop. A value of 0 blocks forever waiting for a key press.
- Release the camera with `cap.release()` when done. If you don't, the camera may stay locked.
- Processing too slowly will cause frame lag. If your per-frame operation is heavy, consider processing every Nth frame or resizing frames before processing.
- On laptops, device index `0` is usually the built-in camera. External USB cameras are typically `1` or higher.
- Some cameras take a few frames to auto-adjust exposure and white balance. The first few frames may be dark or oddly colored.

## Starter Code

```python
import cv2
import numpy as np

# --- Simulate a webcam feed with synthetic frames ---
# In a real scenario you would use:
#   cap = cv2.VideoCapture(0)
# Here we create frames that mimic a live camera feed.

num_frames = 8
frame_h, frame_w = 200, 300

frames = []
processed_frames = []

for i in range(num_frames):
    # Simulate a "webcam" frame with a face-like shape
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (50, 40, 35)

    # Draw a moving "face" (circle + eyes)
    cx = frame_w // 2 + int(30 * np.sin(i * 0.8))
    cy = frame_h // 2 + int(15 * np.cos(i * 0.6))
    cv2.circle(frame, (cx, cy), 50, (180, 200, 220), -1)       # Face
    cv2.circle(frame, (cx - 18, cy - 12), 7, (40, 40, 40), -1) # Left eye
    cv2.circle(frame, (cx + 18, cy - 12), 7, (40, 40, 40), -1) # Right eye
    cv2.ellipse(frame, (cx, cy + 15), (15, 8), 0, 0, 180,
                (40, 40, 40), 2)  # Mouth

    cv2.putText(frame, f'Live Frame {i}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    frames.append(frame)

    # --- Process each frame (simulating real-time processing) ---
    # Apply edge detection, as you might in a live feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(edges_bgr, f'Edges {i}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    processed_frames.append(edges_bgr)

# --- Build a composite display ---
# Top row: raw "webcam" frames
# Bottom row: processed (edge detection) frames
top_row = np.hstack(frames[:4])
mid_row = np.hstack(frames[4:])
bot_row1 = np.hstack(processed_frames[:4])
bot_row2 = np.hstack(processed_frames[4:])

# Add section labels
label_bar_top = np.zeros((30, top_row.shape[1], 3), dtype=np.uint8)
cv2.putText(label_bar_top, 'Simulated Webcam Feed (raw frames)', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

label_bar_bot = np.zeros((30, top_row.shape[1], 3), dtype=np.uint8)
cv2.putText(label_bar_bot, 'Real-Time Processing (Canny edges)', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

result = np.vstack([label_bar_top, top_row, mid_row,
                    label_bar_bot, bot_row1, bot_row2])

print('Simulated webcam capture with real-time edge detection')
print(f'Frame size: {frame_w}x{frame_h}, Total frames: {num_frames}')

cv2.imshow('Webcam Capture', result)
```
