---
slug: 125-live-gesture-drawing
title: Live Gesture-Controlled Drawing
level: live
concepts: [color tracking, persistent canvas, drawing commands, gesture recognition, cv2.line]
prerequisites: [107-live-color-object-tracking]
---

## What Problem Are We Solving?

Traditional drawing applications require a mouse or stylus. But what if you could draw in the air by waving a colored object in front of your webcam? **Gesture-controlled drawing** uses computer vision to turn a physical object -- a bright marker cap, a colored ball, or even a fingertip -- into a virtual pen.

The core challenge is maintaining a **persistent canvas layer** that accumulates the drawing across frames, while the camera feed itself refreshes every frame. You track the object's centroid, connect consecutive positions with lines on the canvas, and blend the canvas with the live camera feed. This creates the illusion of drawing in mid-air.

This technique is used in interactive whiteboards, educational tools, AR drawing apps, and accessibility interfaces for people who cannot use traditional input devices.

## Architecture: Separate Canvas Layer

The key architectural insight is keeping the drawing on a **separate image** from the camera feed:

```python
canvas = np.zeros((height, width, 3), dtype=np.uint8)  # Persistent drawing layer
```

Each frame:
1. Capture a fresh camera frame (changes every frame)
2. Track the pen object and draw new line segments on the **canvas** (persists)
3. Combine: `output = cv2.add(frame, canvas)`

Why not draw directly on the camera frame? Because the camera frame is replaced every iteration. Any drawing on it would vanish after one frame.

## The Drawing Pipeline

```
Camera Frame (fresh)  +  Canvas (persistent)  =  Display Output
        |                       |
   Detect pen object      Draw line segments
   Find centroid          Keep all previous strokes
```

Each frame performs these steps:

1. **Capture** the frame and detect the pen object via color segmentation
2. **Find centroid** of the detected region
3. If the pen was also detected in the previous frame, **draw a line** from the previous centroid to the current centroid on the canvas
4. **Overlay** the canvas on the camera frame using `cv2.add`
5. Display the combined result

## Color Palette System

A good drawing tool needs multiple colors. Implement a palette by defining a set of colors and switching between them with number keys:

```python
colors = {
    '1': (0, 0, 255),     # Red
    '2': (0, 255, 0),     # Green
    '3': (255, 0, 0),     # Blue
    '4': (0, 255, 255),   # Yellow
    '5': (255, 0, 255),   # Magenta
    '6': (255, 255, 0),   # Cyan
    '7': (255, 255, 255), # White
}
```

Display the palette on screen so the user knows which color is active:

```python
for i, (key, color) in enumerate(colors.items()):
    x = 10 + i * 35
    cv2.rectangle(frame, (x, y), (x + 25, y + 25), color, -1)
    if key == active_key:
        cv2.rectangle(frame, (x - 2, y - 2), (x + 27, y + 27), (255, 255, 255), 2)
```

## Thickness Control

Line thickness can be tied to the detected object's area -- a larger detected region (object closer to camera) draws thicker lines:

```python
area = M["m00"]
thickness = max(2, min(15, int(area / 3000)))
```

Or use fixed thickness levels controlled by keys:

| Key | Thickness | Use Case |
|---|---|---|
| `-` | 2 px | Fine detail |
| `=` | 5 px | Normal writing |
| `+` | 10 px | Bold strokes |

## Clear and Undo Functionality

**Clear** resets the entire canvas:

```python
if key == ord('c'):
    canvas[:] = 0  # Wipe everything
```

**Undo** requires storing canvas snapshots. Each time a new stroke starts (pen detected after being absent), save the current canvas state:

```python
undo_stack = []
max_undo = 20

# When a new stroke begins:
undo_stack.append(canvas.copy())
if len(undo_stack) > max_undo:
    undo_stack.pop(0)

# On undo keypress:
if key == ord('u') and undo_stack:
    canvas = undo_stack.pop()
```

**Why copy?** NumPy arrays are references. Without `.copy()`, the undo stack would contain references to the same array, and all entries would show the current state.

## Combining Canvas and Frame with cv2.add

`cv2.add` performs saturating addition -- values are capped at 255 rather than wrapping around:

```python
output = cv2.add(frame, canvas)
```

| Function | Behavior | Result on (200 + 100) |
|---|---|---|
| `cv2.add(a, b)` | Saturating add | 255 (capped) |
| `a + b` (NumPy) | Wrapping add | 44 (overflow!) |
| `cv2.addWeighted` | Weighted blend | Depends on weights |

`cv2.add` works well when the canvas is mostly black (zeros) -- only the drawn lines add color to the frame. For semi-transparent overlays, use `cv2.addWeighted` instead.

## Handling Detection Gaps

When the pen object leaves the frame or is occluded, detection fails. Without proper handling, the next detection would draw a line from the last known position to the new position -- creating a long unwanted stroke across the canvas.

The solution: only draw a line when the pen was detected in **both** the current and previous frames:

```python
if centroid is not None and prev_centroid is not None:
    cv2.line(canvas, prev_centroid, centroid, draw_color, draw_thickness)

prev_centroid = centroid  # None if not detected, coordinates if detected
```

When the pen disappears, `prev_centroid` becomes `None`. When it reappears, the first frame has `prev_centroid = None`, so no line is drawn. The second frame draws from the new starting position.

## Tips & Common Mistakes

- Use `cv2.add` (not NumPy `+`) to combine the canvas and frame. NumPy addition wraps around at 255, creating dark artifacts where drawn lines overlap bright parts of the frame.
- Clean the HSV mask thoroughly with erosion and dilation before computing the centroid. A noisy mask causes the centroid to jitter, making drawn lines wobbly.
- Set a minimum contour area threshold (e.g., 500 pixels) to avoid tracking noise blobs as the pen.
- The canvas must be the **same size and dtype** as the camera frame. Mismatched sizes cause OpenCV errors; mismatched dtypes cause silent corruption.
- Undo snapshots consume memory. Limit the stack size (20 is usually enough) and only save when a new stroke begins, not every frame.
- If your drawing looks laggy, the bottleneck is usually the color detection step. Resize the frame before HSV conversion and mask computation, then scale the centroid coordinates back up.
- Mirror the frame (`cv2.flip(frame, 1)`) so the drawing feels natural -- moving your hand right draws right. Without flipping, the motion is reversed like looking in a mirror.

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

# --- HSV range for pen object (bright green by default) ---
lower_hsv = np.array([35, 100, 100])
upper_hsv = np.array([85, 255, 255])

# --- Drawing state ---
canvas = np.zeros((height, width, 3), dtype=np.uint8)
prev_centroid = None
undo_stack = []
max_undo = 20
stroke_active = False  # True while pen is currently drawing

# --- Color palette ---
palette = [
    ("Red",     (0, 0, 255)),
    ("Green",   (0, 255, 0)),
    ("Blue",    (255, 0, 0)),
    ("Yellow",  (0, 255, 255)),
    ("Magenta", (255, 0, 255)),
    ("Cyan",    (255, 255, 0)),
    ("White",   (255, 255, 255)),
]
active_color_idx = 0
draw_thickness = 4

# --- FPS tracking ---
frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Mirror frame for natural drawing
        frame = cv2.flip(frame, 1)

        frame_times.append(time.time())
        if len(frame_times) > 1:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps = 0.0

        # --- Detect pen object ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # --- Find centroid ---
        M = cv2.moments(mask)
        centroid = None

        if M["m00"] > 500:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            # Highlight detected pen position
            cv2.circle(frame, centroid, 8, palette[active_color_idx][1], -1)
            cv2.circle(frame, centroid, 10, (255, 255, 255), 2)

        # --- Drawing logic ---
        if centroid is not None and prev_centroid is not None:
            # Draw line segment on the persistent canvas
            cv2.line(canvas, prev_centroid, centroid,
                     palette[active_color_idx][1], draw_thickness, cv2.LINE_AA)
            stroke_active = True
        elif centroid is not None and prev_centroid is None:
            # New stroke starting -- save undo snapshot
            if stroke_active or cv2.countNonZero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)) > 0:
                undo_stack.append(canvas.copy())
                if len(undo_stack) > max_undo:
                    undo_stack.pop(0)
            stroke_active = True
        else:
            stroke_active = False

        prev_centroid = centroid

        # --- Combine canvas with camera frame ---
        output = cv2.add(frame, canvas)

        # --- Draw color palette ---
        palette_y = 10
        for i, (name, color) in enumerate(palette):
            px = 10 + i * 35
            cv2.rectangle(output, (px, palette_y), (px + 25, palette_y + 25), color, -1)
            if i == active_color_idx:
                cv2.rectangle(output, (px - 2, palette_y - 2),
                              (px + 27, palette_y + 27), (255, 255, 255), 2)

        # --- Draw HUD ---
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"Color: {palette[active_color_idx][0]}  Thick: {draw_thickness}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        status = "DRAWING" if stroke_active else "PEN UP"
        status_color = (0, 255, 0) if stroke_active else (100, 100, 100)
        cv2.putText(output, status, (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        cv2.putText(output, "1-7=color  +/-=thickness  c=clear  u=undo  q=quit",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Gesture Drawing', output)

        # --- Handle keypresses ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            undo_stack.append(canvas.copy())
            canvas[:] = 0
            print("Canvas cleared")
        elif key == ord('u'):
            if undo_stack:
                canvas = undo_stack.pop()
                print(f"Undo ({len(undo_stack)} states remaining)")
            else:
                print("Nothing to undo")
        elif ord('1') <= key <= ord('7'):
            active_color_idx = key - ord('1')
            print(f"Color: {palette[active_color_idx][0]}")
        elif key == ord('=') or key == ord('+'):
            draw_thickness = min(20, draw_thickness + 2)
            print(f"Thickness: {draw_thickness}")
        elif key == ord('-'):
            draw_thickness = max(1, draw_thickness - 2)
            print(f"Thickness: {draw_thickness}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
