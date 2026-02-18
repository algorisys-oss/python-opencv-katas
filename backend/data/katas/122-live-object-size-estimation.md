---
slug: 122-live-object-size-estimation
title: Live Object Size Estimation
level: live
concepts: [reference object calibration, pixels-per-metric, cv2.minAreaRect, real-world measurement]
prerequisites: [100-live-camera-fps, 45-bounding-shapes, 39-contour-properties]
---

## What Problem Are We Solving?

Measuring real-world dimensions of objects from a camera image is one of the most practical applications of computer vision. Manufacturing quality control systems measure parts to verify they are within tolerance. Agriculture uses overhead cameras to estimate fruit sizes for sorting. Shipping companies photograph packages to estimate dimensions. Even home improvement projects benefit from quick camera-based measurements.

The core problem is that a camera image gives you **pixels**, not centimeters. A 200-pixel-wide object could be a 2cm coin close to the camera or a 2-meter table far away. To convert pixels to real-world units, you need a **reference object** of known size in the same image at the same distance. This reference establishes a **pixels-per-metric** ratio that translates pixel measurements into centimeters or inches.

This kata builds a live measurement tool. The user places a reference object of known dimensions (such as a credit card, which is 8.56 cm x 5.398 cm) in the camera's view, presses a key to calibrate, and then the system measures other objects in real-time. The calibration, the measurement process, and the limitations of this approach (especially regarding perspective) are all covered.

## The Pixels-Per-Metric Ratio

The fundamental concept is simple:

```
pixels_per_cm = reference_width_in_pixels / reference_width_in_cm
```

Once you know how many pixels correspond to 1 cm at that distance and angle, you can measure anything else in the frame:

```
object_width_cm = object_width_in_pixels / pixels_per_cm
```

### Example with a Credit Card

A standard credit card is 8.56 cm wide. If it appears 300 pixels wide in the frame:

```python
KNOWN_WIDTH_CM = 8.56
ref_pixels = 300

pixels_per_cm = ref_pixels / KNOWN_WIDTH_CM  # = 35.05 px/cm

# Now measure another object that is 180 pixels wide:
object_cm = 180 / pixels_per_cm  # = 5.14 cm
```

### Common Reference Objects

| Object | Width (cm) | Height (cm) | Notes |
|---|---|---|---|
| Credit/debit card | 8.56 | 5.398 | ISO/IEC 7810 ID-1 standard |
| US dollar bill | 15.6 | 6.6 | Slightly larger than a card |
| A4 paper | 29.7 | 21.0 | Large reference, good for big objects |
| US Letter paper | 27.9 | 21.6 | Common in North America |
| US quarter coin | 2.43 (diameter) | -- | Circular reference |

## Using cv2.minAreaRect for Measurement

`cv2.minAreaRect` returns the smallest rotated rectangle that encloses a contour. This gives the most accurate width and height for objects at any orientation:

```python
rect = cv2.minAreaRect(contour)
# rect = ((center_x, center_y), (width, height), angle)
```

| Component | Type | Meaning |
|---|---|---|
| `rect[0]` | `(float, float)` | Center point `(cx, cy)` |
| `rect[1]` | `(float, float)` | Size `(width, height)` in pixels |
| `rect[2]` | `float` | Rotation angle in degrees |

### Why minAreaRect Instead of boundingRect?

`cv2.boundingRect` returns an axis-aligned rectangle. For rotated objects, this rectangle is larger than the actual object, inflating the measurement. `cv2.minAreaRect` fits tightly regardless of rotation.

```python
# Axis-aligned (inaccurate for rotated objects)
x, y, w, h = cv2.boundingRect(contour)

# Minimum area rotated (accurate for any orientation)
rect = cv2.minAreaRect(contour)
(cx, cy), (w, h), angle = rect
```

For consistent measurement, always use the **longer** side as "width" and the **shorter** side as "height" (or vice versa, as long as you are consistent):

```python
(cx, cy), (w, h), angle = cv2.minAreaRect(contour)
obj_w = max(w, h)
obj_h = min(w, h)
```

### Drawing the Rotated Rectangle

```python
box = cv2.boxPoints(rect)    # Get 4 corner points
box = np.int32(box)          # Convert to integers
cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
```

## The Calibration Process

Calibration establishes the pixels-per-metric ratio from a reference object:

1. Place the reference object on a flat surface in the camera's view.
2. Detect the reference object's contour.
3. Compute `cv2.minAreaRect` to get its pixel dimensions.
4. Divide pixel width by known real-world width to get `pixels_per_cm`.

```python
# During calibration (user presses 'c'):
ref_rect = cv2.minAreaRect(reference_contour)
(cx, cy), (rw, rh), angle = ref_rect
ref_pixel_width = max(rw, rh)

KNOWN_WIDTH_CM = 8.56  # Credit card width
pixels_per_cm = ref_pixel_width / KNOWN_WIDTH_CM
```

### Selecting the Reference Object

The reference object should be the **leftmost** or otherwise identifiable object. A simple heuristic is to use the leftmost contour (smallest x-coordinate of the bounding rectangle) as the reference, assuming the user places the reference card on the left side of the scene.

Alternatively, use the contour closest to a specific aspect ratio. A credit card has an aspect ratio of approximately 8.56 / 5.398 = 1.586. Filter contours by this ratio to identify the card automatically:

```python
TARGET_ASPECT_RATIO = 8.56 / 5.398  # ~1.586
ASPECT_TOLERANCE = 0.3

for contour in contours:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if min(w, h) == 0:
        continue
    ar = max(w, h) / min(w, h)
    if abs(ar - TARGET_ASPECT_RATIO) < ASPECT_TOLERANCE:
        # Likely the credit card
        reference_contour = contour
        break
```

## Contour Detection for Objects

To measure objects, you first need clean contour detection. A typical pipeline:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150)

# Close gaps
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Filter out small contours (noise) by area:

```python
MIN_AREA = 1000  # pixels^2
contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
```

## Perspective Limitations

The biggest caveat of this approach is that **the pixels-per-metric ratio is only valid at the same distance and plane as the reference object**. Objects closer to the camera appear larger (more pixels per cm); objects farther away appear smaller.

| Scenario | Accuracy |
|---|---|
| Object and reference on the same flat surface, camera looking straight down | Best -- consistent pixels-per-cm across the surface |
| Object and reference at the same distance from camera, camera angled | Good -- minor distortion at edges |
| Object farther/closer than reference | Poor -- measurement will be over/underestimated |
| Camera at steep angle to the surface | Poor -- foreshortening distorts measurements |

For the most accurate results:
1. Mount the camera directly above the surface (bird's-eye view).
2. Keep all objects on the same flat plane.
3. Place the reference object near the center of the frame.

## Drawing Dimension Labels

Label each object with its measured dimensions:

```python
def draw_dimension(frame, pt_a, pt_b, dim_cm, color=(0, 255, 255)):
    """Draw a dimension line with measurement between two points."""
    pt_a = tuple(map(int, pt_a))
    pt_b = tuple(map(int, pt_b))

    cv2.line(frame, pt_a, pt_b, color, 2)
    cv2.circle(frame, pt_a, 4, color, -1)
    cv2.circle(frame, pt_b, 4, color, -1)

    mid_x = (pt_a[0] + pt_b[0]) // 2
    mid_y = (pt_a[1] + pt_b[1]) // 2
    cv2.putText(frame, f"{dim_cm:.1f} cm", (mid_x - 25, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
```

For a minimum area rect, the four corner points from `cv2.boxPoints` can be used to draw dimension lines along the width and height edges:

```python
box = cv2.boxPoints(rect)
# box[0]-box[1] is one edge, box[1]-box[2] is the adjacent edge
edge1_len = np.linalg.norm(box[0] - box[1])
edge2_len = np.linalg.norm(box[1] - box[2])

# The longer edge is the "width"
if edge1_len > edge2_len:
    width_pts = (box[0], box[1])
    height_pts = (box[1], box[2])
else:
    width_pts = (box[1], box[2])
    height_pts = (box[0], box[1])
```

## Tips & Common Mistakes

- Always calibrate before measuring. Without calibration, pixel measurements are meaningless in real-world units.
- `cv2.minAreaRect` returns `(width, height)` where width is not necessarily the longer side. Always use `max()` and `min()` to get consistent long and short dimensions.
- `cv2.boxPoints()` returns corners as float. Convert to int with `np.int32(box)` before drawing.
- The pixels-per-metric ratio is only valid at the calibration distance. Moving the camera closer or farther invalidates the calibration -- recalibrate after any camera movement.
- Use `cv2.RETR_EXTERNAL` for `findContours` to only get outer contours. Internal holes and nested contours would create false measurements.
- Filter contours by minimum area (e.g., 1000 pixels) to ignore noise. The exact threshold depends on camera resolution and scene.
- For objects touching the frame border, `minAreaRect` will give incorrect measurements because the contour is truncated. Ignore contours that touch the image edges.
- A bird's-eye (top-down) camera angle gives the most accurate measurements. At oblique angles, objects closer to the camera appear larger, breaking the constant pixels-per-cm assumption.
- Gaussian blur before Canny edge detection reduces noise that creates false contours. A `(7, 7)` kernel is a good starting point for measurement applications.
- If the reference object is not detected reliably, try adjusting the Canny thresholds or placing it on a contrasting background (white card on dark surface, or vice versa).
- Store the calibration value across frames. Recalibrating every frame would cause the measurements to fluctuate with detection noise.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Place a credit card (8.56 cm wide) on the left side of a flat surface in view — it should be highlighted in yellow and labeled "REFERENCE?"
- Press **c** to calibrate — the status should change to "Calibrated" with a px/cm value, and other objects in view should display their dimensions in centimeters
- Press **r** to reset calibration and verify it returns to "NOT CALIBRATED" mode
- Check the edge detection preview in the top-right corner to confirm clean contour detection

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

frame_times = deque(maxlen=30)

# --- Calibration state ---
pixels_per_cm = None
KNOWN_WIDTH_CM = 8.56   # Credit card width in cm
KNOWN_HEIGHT_CM = 5.398  # Credit card height in cm
MIN_CONTOUR_AREA = 1500

# --- Edge detection parameters ---
CANNY_LOW = 50
CANNY_HIGH = 150


def order_box_points(box):
    """Order box points: top-left, top-right, bottom-right, bottom-left."""
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]
    ordered[2] = box[np.argmax(s)]
    ordered[1] = box[np.argmin(d)]
    ordered[3] = box[np.argmax(d)]
    return ordered


def midpoint(pt_a, pt_b):
    """Return the midpoint between two points."""
    return ((pt_a[0] + pt_b[0]) / 2, (pt_a[1] + pt_b[1]) / 2)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]

        # --- Edge detection pipeline ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

        # --- Find contours ---
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area and sort left-to-right
        valid_contours = []
        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            # Skip contours touching frame borders
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            valid_contours.append(c)

        # Sort by x-position (leftmost first)
        valid_contours = sorted(valid_contours,
                                key=lambda c: cv2.boundingRect(c)[0])

        # --- Process each contour ---
        for i, contour in enumerate(valid_contours):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box)
            ordered = order_box_points(box)

            # Compute edge lengths
            edge_top = np.linalg.norm(ordered[1] - ordered[0])
            edge_right = np.linalg.norm(ordered[2] - ordered[1])
            obj_w_px = max(edge_top, edge_right)
            obj_h_px = min(edge_top, edge_right)

            # Draw the bounding box
            draw_box = np.int32(box)

            if pixels_per_cm is not None:
                # --- Measurement mode ---
                obj_w_cm = obj_w_px / pixels_per_cm
                obj_h_cm = obj_h_px / pixels_per_cm

                cv2.drawContours(frame, [draw_box], 0, (0, 255, 0), 2)

                # Draw dimension along the longer edge
                if edge_top >= edge_right:
                    mid_w = midpoint(ordered[0], ordered[1])
                    mid_w2 = midpoint(ordered[3], ordered[2])
                    mid_h = midpoint(ordered[1], ordered[2])
                    mid_h2 = midpoint(ordered[0], ordered[3])
                else:
                    mid_w = midpoint(ordered[1], ordered[2])
                    mid_w2 = midpoint(ordered[0], ordered[3])
                    mid_h = midpoint(ordered[0], ordered[1])
                    mid_h2 = midpoint(ordered[3], ordered[2])

                # Width label
                mid_w_center = midpoint(mid_w, mid_w2)
                cv2.putText(frame, f"{obj_w_cm:.1f}cm",
                            (int(mid_w_center[0]) - 20, int(mid_w_center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                # Height label
                mid_h_center = midpoint(mid_h, mid_h2)
                cv2.putText(frame, f"{obj_h_cm:.1f}cm",
                            (int(mid_h_center[0]) + 10, int(mid_h_center[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                # Corner dots
                for pt in ordered:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 200, 255), -1)
            else:
                # --- Pre-calibration mode: highlight leftmost as reference candidate ---
                if i == 0:
                    cv2.drawContours(frame, [draw_box], 0, (0, 255, 255), 3)
                    cx, cy_r = int(rect[0][0]), int(rect[0][1])
                    cv2.putText(frame, "REFERENCE?", (cx - 40, cy_r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{obj_w_px:.0f}x{obj_h_px:.0f} px",
                                (cx - 35, cy_r + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                else:
                    cv2.drawContours(frame, [draw_box], 0, (100, 100, 100), 1)

        # --- Show edge preview ---
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        preview_w, preview_h = 160, 120
        edge_small = cv2.resize(edge_bgr, (preview_w, preview_h))
        frame[0:preview_h, w - preview_w:w] = edge_small
        cv2.putText(frame, "Edges", (w - preview_w + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if pixels_per_cm is not None:
            cv2.putText(frame, f"Calibrated: {pixels_per_cm:.1f} px/cm", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Objects: {len(valid_contours)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "NOT CALIBRATED", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Place reference ({KNOWN_WIDTH_CM}cm card) on LEFT",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'c' to calibrate", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "Press 'q' to quit | 'c' calibrate | 'r' reset",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Object Size Estimation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Calibrate using the leftmost contour
            if valid_contours:
                ref_contour = valid_contours[0]
                ref_rect = cv2.minAreaRect(ref_contour)
                rw, rh = ref_rect[1]
                ref_pixel_width = max(rw, rh)
                pixels_per_cm = ref_pixel_width / KNOWN_WIDTH_CM
                print(f"Calibrated! Reference: {ref_pixel_width:.0f} px = {KNOWN_WIDTH_CM} cm")
                print(f"Pixels per cm: {pixels_per_cm:.2f}")
            else:
                print("No contour detected for calibration. Place the reference object in view.")
        elif key == ord('r'):
            pixels_per_cm = None
            print("Calibration reset.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
