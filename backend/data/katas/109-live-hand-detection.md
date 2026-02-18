---
slug: 109-live-hand-detection
title: Live Hand Detection
level: live
concepts: [skin color segmentation, cv2.convexHull, convexity defects, finger counting, YCrCb]
prerequisites: [100-live-camera-fps, 41-convex-hull]
---

## What Problem Are We Solving?

Hand detection and finger counting is a gateway to gesture-based interfaces — controlling a robot with hand signals, navigating a presentation by showing numbers, or building sign language recognition systems. Unlike face detection, there is no pre-trained Haar cascade for hands that works reliably across skin tones and orientations. Instead, we use a fundamentally different approach: **skin color segmentation**.

The idea is to isolate skin-colored pixels, find the hand contour, compute its convex hull, and then analyze **convexity defects** — the gaps between fingers — to count how many fingers are raised. This approach works without machine learning models and runs in real-time on any hardware, making it an excellent introduction to contour-based gesture recognition.

The main challenge is robustness. Skin color varies across individuals and lighting conditions. The **YCrCb color space** provides more consistent skin detection than BGR or HSV because it separates luminance (Y) from chrominance (Cr, Cb), and research has shown that skin pixels cluster tightly in the Cr-Cb plane regardless of skin tone.

## Skin Color Segmentation in YCrCb

The YCrCb color space splits an image into luminance (brightness) and two chrominance channels:

```python
ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
```

| Channel | Range | Meaning |
|---|---|---|
| **Y** | 0-255 | Luminance (brightness) |
| **Cr** | 0-255 | Red chrominance — how much red the pixel deviates from gray |
| **Cb** | 0-255 | Blue chrominance — how much blue the pixel deviates from gray |

### Why YCrCb for Skin Detection?

Research on skin color modeling has shown that skin pixels of all ethnicities cluster in a narrow band of the Cr-Cb plane:

- **Cr range:** approximately 133-173
- **Cb range:** approximately 77-127

This clustering is tighter than in HSV or BGR because the Y (brightness) channel absorbs most of the lighting variation, leaving Cr and Cb relatively stable.

```python
lower_skin = np.array([0, 133, 77])
upper_skin = np.array([255, 173, 127])
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
```

### YCrCb Skin Detection Ranges

| Skin Tone | Y Low | Y High | Cr Low | Cr High | Cb Low | Cb High |
|---|---|---|---|---|---|---|
| General (all tones) | 0 | 255 | 133 | 173 | 77 | 127 |
| Light skin (refined) | 0 | 255 | 140 | 170 | 80 | 120 |
| Dark skin (wider) | 0 | 255 | 130 | 175 | 75 | 130 |

> **Tip:** Start with the general range and adjust Cr/Cb bounds if your environment produces too many false positives (background detected as skin) or false negatives (hand not detected).

### Alternative: HSV Skin Detection

HSV can also detect skin, though less reliably:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_skin_hsv = np.array([0, 30, 60])
upper_skin_hsv = np.array([20, 150, 255])
mask = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
```

HSV skin detection is more affected by lighting than YCrCb, but can be useful as a secondary confirmation.

## Cleaning the Skin Mask

Raw skin masks are noisy — speckled with false detections from wood furniture, brown backgrounds, etc. Clean aggressively:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# Fill holes
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
# Smooth edges
mask = cv2.GaussianBlur(mask, (5, 5), 0)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
```

## Finding the Hand Contour

After masking, find the largest contour (assumed to be the hand):

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) > 5000:  # Minimum area for a hand
        # Process hand contour
        ...
```

The minimum area threshold (5000 pixels at 640x480) filters out small skin-colored noise blobs. A hand typically occupies 5000-50000 pixels depending on distance from the camera.

## Convex Hull and Convexity Defects

The convex hull is the smallest convex polygon that encloses the contour. For a hand with spread fingers, the convex hull wraps around the fingertips, and the **convexity defects** are the valleys between the fingers:

```python
hull = cv2.convexHull(hand)
cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
```

### Computing Convexity Defects

To get defects, you need the hull computed with `returnPoints=False` (returns indices instead of points):

```python
hull_indices = cv2.convexHull(hand, returnPoints=False)
defects = cv2.convexityDefects(hand, hull_indices)
```

Each defect is a 4-element array:

| Index | Name | Meaning |
|---|---|---|
| `[0]` | `start_index` | Index of the start point on the contour (fingertip) |
| `[1]` | `end_index` | Index of the end point on the contour (next fingertip) |
| `[2]` | `far_index` | Index of the farthest point from the hull (valley between fingers) |
| `[3]` | `distance` | Distance of the farthest point from the hull edge, **multiplied by 256** |

> **Critical:** The `distance` value is fixed-point scaled by 256. Divide by 256.0 to get the actual pixel distance.

## Counting Fingers with Angle Filtering

Not every convexity defect is a gap between fingers. Defects at the wrist or between tightly held fingers have wide angles. Filter defects by the angle formed at the valley point:

```python
import math

finger_count = 0

if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(hand[s][0])
        end = tuple(hand[e][0])
        far = tuple(hand[f][0])

        # Calculate the angle at the defect point
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # Cosine rule: angle = arccos((b^2 + c^2 - a^2) / (2*b*c))
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c + 1e-6))
        angle_deg = math.degrees(angle)

        # Filter: finger gaps have angles roughly < 90 degrees
        if angle_deg < 90 and (d / 256.0) > 20:
            finger_count += 1
            cv2.circle(frame, far, 5, (0, 0, 255), -1)

# Defects count gaps between fingers, so add 1 for the finger count
finger_count = min(finger_count + 1, 5)
```

### Angle Filtering Logic

| Angle at Valley | Interpretation | Action |
|---|---|---|
| < 60 degrees | Tight gap between adjacent spread fingers | Count as finger gap |
| 60-90 degrees | Moderate gap — likely a finger gap | Count as finger gap |
| > 90 degrees | Wide gap — wrist edge or palm side | Ignore |

### Distance Filtering

The defect distance (depth of the valley) also helps:

| Distance (pixels) | Interpretation |
|---|---|
| < 20 | Very shallow — noise or tightly held fingers |
| 20-100 | Typical finger gap depth |
| > 100 | Deep valley — likely the wrist area at close range |

## Tips & Common Mistakes

- **YCrCb skin detection works better than HSV** for most skin tones because luminance is separated from chrominance. Start with YCrCb.
- The face is also skin-colored. If the face is in frame, it creates a large competing contour. Consider detecting and excluding the face ROI, or position the hand closer to the camera than the face.
- Convexity defects count the **gaps** between fingers, not the fingers themselves. For N defects passing the angle filter, the finger count is N+1 (up to 5).
- Always check `defects is not None` before iterating. A convex contour (e.g., a fist) has no defects.
- The `distance` field in defects is multiplied by 256 (fixed-point encoding). Divide by 256.0 before comparing to pixel thresholds.
- A closed fist produces 0 valid defects, giving a finger count of 1 (the fist itself). Distinguish "fist" from "one finger" by checking the aspect ratio or area ratio of the contour vs its convex hull.
- Lighting is the biggest enemy. Direct sunlight creates harsh shadows that break skin segmentation. Diffuse, even lighting works best.
- The wrist area produces large defects with wide angles. The angle filter (< 90 degrees) is essential to avoid counting the wrist as finger gaps.
- Background objects matching skin color (wooden desk, beige wall) will confuse the detector. Use a non-skin-colored background or add a region of interest.

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

frame_times = deque(maxlen=30)

# --- Skin detection parameters (YCrCb) ---
lower_skin = np.array([0, 133, 77])
upper_skin = np.array([255, 173, 127])

# --- Morphological kernel ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

MIN_HAND_AREA = 5000
ANGLE_THRESHOLD = 90  # degrees
DEPTH_THRESHOLD = 20  # pixels (after dividing by 256)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Flip for mirror effect ---
        frame = cv2.flip(frame, 1)

        # --- Convert to YCrCb and segment skin ---
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # --- Clean mask ---
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # --- Find contours ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        finger_count = 0
        hand_detected = False

        if contours:
            hand = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand)

            if area > MIN_HAND_AREA:
                hand_detected = True

                # Draw hand contour
                cv2.drawContours(frame, [hand], -1, (0, 255, 0), 2)

                # Convex hull (for drawing)
                hull_draw = cv2.convexHull(hand)
                cv2.drawContours(frame, [hull_draw], -1, (0, 255, 255), 2)

                # Convex hull (indices, for defects)
                hull_indices = cv2.convexHull(hand, returnPoints=False)

                # Convexity defects
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(hand, hull_indices)

                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(hand[s][0])
                            end = tuple(hand[e][0])
                            far = tuple(hand[f][0])

                            # Calculate angle at defect point (cosine rule)
                            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                            denom = 2 * b * c
                            if denom > 0:
                                cos_angle = (b**2 + c**2 - a**2) / denom
                                cos_angle = max(-1, min(1, cos_angle))
                                angle_deg = math.degrees(math.acos(cos_angle))
                            else:
                                angle_deg = 180

                            depth = d / 256.0

                            # Filter: finger gaps have small angles and sufficient depth
                            if angle_deg < ANGLE_THRESHOLD and depth > DEPTH_THRESHOLD:
                                finger_count += 1
                                cv2.circle(frame, far, 6, (0, 0, 255), -1)
                                cv2.line(frame, start, far, (255, 0, 0), 1)
                                cv2.line(frame, end, far, (255, 0, 0), 1)

                # Defects count gaps; fingers = gaps + 1
                finger_count = min(finger_count + 1, 5)

                # Draw bounding box and area info
                x, y, w, h = cv2.boundingRect(hand)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 1)

        # --- Show mask preview ---
        mask_small = cv2.resize(mask, (160, 120))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[0:120, frame.shape[1] - 160:frame.shape[1]] = mask_bgr

        # --- Draw large finger count ---
        if hand_detected:
            cv2.putText(frame, str(finger_count), (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, "fingers", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        status = "Hand Detected" if hand_detected else "No Hand"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
```
