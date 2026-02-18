---
slug: 102-live-color-picker
title: Live Color Picker
level: live
concepts: [cv2.setMouseCallback, cv2.cvtColor, BGR to HSV, pixel sampling]
prerequisites: [100-live-camera-fps, 02-color-spaces]
---

## What Problem Are We Solving?

When building color-based detection systems (tracking a red ball, detecting skin tone, isolating a green screen), you need to know the **exact HSV range** of the color you're targeting. But colors look different under different lighting conditions — the "red" of a ball under fluorescent light has completely different HSV values than under sunlight.

A **live color picker** lets you point your camera at an object, click on it, and instantly see its BGR and HSV values. This is an indispensable development tool for any color-based computer vision pipeline.

## Mouse Callbacks in OpenCV

OpenCV lets you register a function that gets called whenever the mouse does something in a window:

```python
def mouse_callback(event, x, y, flags, param):
    # event: what happened (click, move, etc.)
    # x, y: pixel coordinates of the mouse
    # flags: modifier keys (Ctrl, Shift, etc.)
    # param: user data passed during registration

cv2.setMouseCallback('window_name', mouse_callback)
```

### Mouse Events

| Event | When It Fires |
|---|---|
| `cv2.EVENT_MOUSEMOVE` | Mouse moves over the window |
| `cv2.EVENT_LBUTTONDOWN` | Left button pressed |
| `cv2.EVENT_LBUTTONUP` | Left button released |
| `cv2.EVENT_RBUTTONDOWN` | Right button pressed |
| `cv2.EVENT_LBUTTONDBLCLK` | Left button double-clicked |

```python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
    elif event == cv2.EVENT_MOUSEMOVE:
        # fires continuously as mouse moves
        pass
```

### Sharing Data Between Callback and Main Loop

The callback runs in a different context than your main loop. To share data (like the selected pixel color), use a mutable container:

```python
# Use a dictionary as shared state
state = {"color_bgr": None, "color_hsv": None, "pos": None}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param  # We'll pass the current frame as param
        # Bounds check to avoid index errors
        h, w = frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            state["color_bgr"] = frame[y, x].tolist()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            state["color_hsv"] = hsv_frame[y, x].tolist()
            state["pos"] = (x, y)
```

> **Critical detail:** Array indexing is `frame[y, x]` not `frame[x, y]`. Mouse coordinates are `(x, y)` but NumPy arrays are `[row, col]` = `[y, x]`.

## Sampling a Region Instead of a Single Pixel

A single pixel is noisy — it might not represent the "true" color of the object. Sampling a small neighborhood gives you a more stable reading:

```python
def sample_region(frame, x, y, radius=5):
    """Sample the average color in a square region around (x, y)."""
    h, w = frame.shape[:2]
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    region = frame[y1:y2, x1:x2]
    return region.mean(axis=(0, 1)).astype(int)  # Average BGR
```

This returns the **average color** of an 11x11 pixel region (radius=5), which is much more representative of the object's actual color.

## BGR to HSV Conversion

To convert a single pixel from BGR to HSV:

```python
bgr_pixel = frame[y, x]
# Reshape to (1, 1, 3) — cvtColor needs at least a 2D image
bgr_array = np.uint8([[bgr_pixel]])
hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
hsv_pixel = hsv_array[0, 0]
```

Or convert the entire frame once:

```python
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv_pixel = hsv_frame[y, x]
```

### OpenCV HSV Ranges (Reminder)

| Channel | Range | Meaning |
|---|---|---|
| H (Hue) | 0–179 | Color angle (0=Red, 60=Green, 120=Blue) |
| S (Saturation) | 0–255 | Color purity (0=gray, 255=vivid) |
| V (Value) | 0–255 | Brightness (0=black, 255=bright) |

> **OpenCV uses H: 0–179** (not 0–360). This is because `uint8` can only hold 0–255, and OpenCV halves the hue to fit.

## Drawing a Color Swatch

To visually display the picked color, draw a filled rectangle with that color:

```python
# Create a color swatch showing the picked color
swatch = np.zeros((60, 200, 3), dtype=np.uint8)
swatch[:] = color_bgr  # Fill with the picked color
cv2.putText(swatch, f"BGR: {color_bgr}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(swatch, f"HSV: {color_hsv}", (5, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
```

## Building an HSV Range from Picked Colors

Once you pick a color, you can compute an `inRange` mask for detection:

```python
# Define a tolerance around the picked HSV value
h, s, v = color_hsv
tolerance_h, tolerance_s, tolerance_v = 10, 40, 40

lower = np.array([max(0, h - tolerance_h), max(0, s - tolerance_s), max(0, v - tolerance_v)])
upper = np.array([min(179, h + tolerance_h), min(255, s + tolerance_s), min(255, v + tolerance_v)])

mask = cv2.inRange(hsv_frame, lower, upper)
```

## Tips & Common Mistakes

- Mouse coordinates are `(x, y)` but array access is `[y, x]`. Mixing these up is the #1 source of bugs.
- Always bounds-check before indexing: `if 0 <= x < w and 0 <= y < h`.
- `cv2.setMouseCallback` must be called **after** `cv2.namedWindow` or `cv2.imshow` — the window must exist first.
- The callback fires on mouse events only — it doesn't have access to the "current" frame unless you pass it somehow.
- For continuous color readout (hover), use `EVENT_MOUSEMOVE` instead of `EVENT_LBUTTONDOWN`.
- Sampling a region (5-10px radius) is much more reliable than a single pixel.
- HSV hue wraps around: red appears at both H≈0 and H≈179. If your detected color is red, you may need two `inRange` calls.
- The color swatch should use BGR values directly (OpenCV's native format), not RGB.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Move your mouse over the camera feed — a crosshair should follow, and the BGR/HSV values at the bottom should update live
- Click on a colored object — a yellow circle marks the picked point and a color swatch with BGR/HSV values appears in the top-right corner
- Click on several different colored objects and confirm the swatch updates each time with accurate color values

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Shared state for mouse callback ---
pick = {
    "bgr": None, "hsv": None,
    "pos": None, "hover_bgr": None, "hover_hsv": None, "hover_pos": None
}

current_frame = [None]  # Mutable container to share frame with callback

def mouse_callback(event, x, y, flags, param):
    frame = current_frame[0]
    if frame is None:
        return
    h_img, w_img = frame.shape[:2]
    if not (0 <= x < w_img and 0 <= y < h_img):
        return

    # Sample a 5px radius region for stability
    radius = 5
    y1, y2 = max(0, y - radius), min(h_img, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w_img, x + radius + 1)
    region_bgr = frame[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.uint8)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    region_hsv = hsv_frame[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.uint8)

    if event == cv2.EVENT_MOUSEMOVE:
        pick["hover_bgr"] = region_bgr.tolist()
        pick["hover_hsv"] = region_hsv.tolist()
        pick["hover_pos"] = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        pick["bgr"] = region_bgr.tolist()
        pick["hsv"] = region_hsv.tolist()
        pick["pos"] = (x, y)
        print(f"Picked at ({x},{y}): BGR={pick['bgr']}  HSV={pick['hsv']}")

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('Live Color Picker')
cv2.setMouseCallback('Live Color Picker', mouse_callback)

frame_times = deque(maxlen=30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame[0] = frame.copy()

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        display = frame.copy()

        # --- Draw crosshair at hover position ---
        if pick["hover_pos"]:
            hx, hy = pick["hover_pos"]
            cv2.drawMarker(display, (hx, hy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

            # Show hover color info
            if pick["hover_bgr"]:
                b, g, r = pick["hover_bgr"]
                h, s, v = pick["hover_hsv"]
                cv2.putText(display, f"BGR:({b},{g},{r}) HSV:({h},{s},{v})",
                            (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Draw picked color swatch ---
        if pick["bgr"]:
            px, py = pick["pos"]
            # Draw marker at picked location
            cv2.circle(display, (px, py), 8, (0, 255, 255), 2)

            # Draw color swatch in top-right corner
            swatch_w, swatch_h = 180, 80
            sx = display.shape[1] - swatch_w - 10
            sy = 10

            # Background
            cv2.rectangle(display, (sx, sy), (sx + swatch_w, sy + swatch_h), (0, 0, 0), -1)
            cv2.rectangle(display, (sx, sy), (sx + swatch_w, sy + swatch_h), (255, 255, 255), 1)

            # Color fill
            cv2.rectangle(display, (sx + 5, sy + 5), (sx + 45, sy + 45),
                          tuple(pick["bgr"]), -1)
            cv2.rectangle(display, (sx + 5, sy + 5), (sx + 45, sy + 45),
                          (255, 255, 255), 1)

            # Text info
            b, g, r = pick["bgr"]
            h, s, v = pick["hsv"]
            cv2.putText(display, f"BGR:({b},{g},{r})", (sx + 50, sy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"HSV:({h},{s},{v})", (sx + 50, sy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(display, f"at ({px},{py})", (sx + 50, sy + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        # --- FPS and instructions ---
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, "Click to pick color | 'q' to quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Live Color Picker', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
