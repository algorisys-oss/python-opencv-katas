---
slug: 112-live-multi-color-tracker
title: Live Multi-Object Color Tracker
level: live
concepts: [multiple HSV ranges, cv2.inRange, contour detection, multi-object tracking, color assignment]
prerequisites: [107-live-color-object-tracking]
---

## What Problem Are We Solving?

In the single-color tracker (kata 107), we tracked one colored object at a time. But many real-world scenarios require tracking **multiple objects simultaneously**, each identified by a different color. Think of a tabletop game where red, green, and blue tokens are tracked for an augmented reality overlay, a robotics competition where each team's robot has a colored marker, or an interactive art installation where multiple participants hold different colored wands.

The challenge is managing **multiple HSV masks in parallel**: each color needs its own range, its own contour detection, its own centroid computation, and its own persistent trail. The processing pipeline multiplies by the number of colors being tracked, so efficiency matters — you need to keep the per-color overhead low to maintain real-time frame rates.

This kata teaches you to build a multi-color tracking system where each tracked color has its own label, bounding box, centroid, and fading trail, all running simultaneously on a live webcam feed.

## Defining Multiple Color Targets

Organize each target color as a dictionary with its HSV range, display color, label, and trail:

```python
from collections import deque

color_targets = [
    {
        "name": "Red",
        "lower1": (0, 120, 70),     # Red wraps: need two ranges
        "upper1": (10, 255, 255),
        "lower2": (170, 120, 70),
        "upper2": (180, 255, 255),
        "bgr": (0, 0, 255),         # Drawing color in BGR
        "trail": deque(maxlen=64),
    },
    {
        "name": "Green",
        "lower1": (40, 80, 80),
        "upper1": (80, 255, 255),
        "lower2": None,              # No second range needed
        "upper2": None,
        "bgr": (0, 255, 0),
        "trail": deque(maxlen=64),
    },
    {
        "name": "Blue",
        "lower1": (100, 120, 70),
        "upper1": (130, 255, 255),
        "lower2": None,
        "upper2": None,
        "bgr": (255, 0, 0),
        "trail": deque(maxlen=64),
    },
]
```

### HSV Ranges for Common Tracking Colors

| Color | H Low 1 | H High 1 | H Low 2 | H High 2 | S Low | S High | V Low | V High |
|---|---|---|---|---|---|---|---|---|
| Red | 0 | 10 | 170 | 180 | 120 | 255 | 70 | 255 |
| Orange | 10 | 25 | - | - | 100 | 255 | 100 | 255 |
| Yellow | 25 | 35 | - | - | 100 | 255 | 100 | 255 |
| Green | 40 | 80 | - | - | 80 | 255 | 80 | 255 |
| Cyan | 80 | 100 | - | - | 100 | 255 | 80 | 255 |
| Blue | 100 | 130 | - | - | 120 | 255 | 70 | 255 |
| Purple | 130 | 160 | - | - | 80 | 255 | 50 | 255 |
| Pink | 160 | 175 | - | - | 50 | 255 | 100 | 255 |

> **Note:** Red requires two ranges because it wraps around hue 0/180. All other colors use a single contiguous range.

## Processing Multiple Masks

For each color target, create a mask, clean it, and find the largest contour:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

for target in color_targets:
    # Create mask (handle red's wrap-around)
    mask = cv2.inRange(hsv,
                       np.array(target["lower1"]),
                       np.array(target["upper1"]))
    if target["lower2"] is not None:
        mask2 = cv2.inRange(hsv,
                            np.array(target["lower2"]),
                            np.array(target["upper2"]))
        mask = cv2.bitwise_or(mask, mask2)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # ... process contours ...
```

### Performance Considerations

Each additional color target adds one or two `cv2.inRange` calls plus morphological operations and contour detection. Here is the approximate cost breakdown:

| Operation | Time per Color (640x480) | Notes |
|---|---|---|
| `cv2.inRange` | ~0.5 ms | Very fast, pixel-level comparison |
| `cv2.morphologyEx` (x2) | ~1-2 ms | Depends on kernel size and iterations |
| `cv2.findContours` | ~0.5-1 ms | Depends on mask complexity |
| Centroid computation | < 0.1 ms | Negligible |
| **Total per color** | **~2-4 ms** | |

With 3 colors: ~6-12 ms total, leaving plenty of time for 30 FPS operation. With 6+ colors, consider optimizing by reducing resolution or kernel size.

## Per-Color Centroid and Trail

Each color target maintains its own deque-based trail:

```python
centroid = None
if contours:
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) > MIN_AREA:
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

target["trail"].appendleft(centroid)
```

### Drawing Labeled Trails

Each color's trail uses the target's BGR color for drawing:

```python
trail = target["trail"]
color = target["bgr"]
name = target["name"]

# Draw trail
for i in range(1, len(trail)):
    if trail[i - 1] is None or trail[i] is None:
        continue
    thickness = max(1, int((len(trail) - i) / len(trail) * 4))
    cv2.line(frame, trail[i - 1], trail[i], color, thickness)

# Draw centroid and label
if centroid:
    cv2.circle(frame, centroid, 6, color, -1)
    cv2.putText(frame, name, (centroid[0] + 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
```

## Handling Overlapping Colors

When two tracked objects overlap or are very close, their contours may merge or interfere. Strategies to handle this:

| Strategy | Implementation | When to Use |
|---|---|---|
| Largest contour only | `max(contours, key=cv2.contourArea)` | Default — tracks one object per color |
| All contours above threshold | Iterate all contours with `area > MIN_AREA` | When multiple objects of the same color exist |
| Minimum distance filter | Skip contours too close to another color's centroid | When colors are physically close |

## Building a Status Dashboard

Display tracking status for all colors in a compact format:

```python
y_offset = 30
for target in color_targets:
    trail = target["trail"]
    active = len(trail) > 0 and trail[0] is not None
    status = "Active" if active else "Lost"
    color = target["bgr"] if active else (100, 100, 100)
    cv2.putText(frame, f"{target['name']}: {status}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    y_offset += 25
```

## Tips & Common Mistakes

- **Pre-compute the HSV image once.** Convert `frame` to HSV once before the color loop. Do not convert inside the loop — it wastes 3-5ms per redundant conversion.
- Blur the frame before HSV conversion to reduce noise. A single `cv2.GaussianBlur(frame, (11, 11), 0)` benefits all color channels.
- If two target colors have overlapping HSV ranges (e.g., orange overlaps with both red and yellow), tighten the ranges or use narrower Saturation/Value bounds.
- Red always needs two `cv2.inRange` calls because hue wraps at 0/180. Do not try to use a single range like `(170, 10)`.
- Each color target should have its own independent trail deque. Do not share trails between colors.
- The morphological kernel can be shared across all colors (create it once before the loop).
- For debugging, show each color's mask as a small preview in the corner of the frame. Cycle through them with a keypress.
- If one color is significantly slower to detect than others, its HSV range may be too broad, capturing too many pixels and creating complex masks.
- When tracking fails for a color (no contour above threshold), append `None` to the trail. This creates gaps in the drawn trail rather than false connections.
- Test each color individually first by temporarily commenting out the others. Once each works reliably, enable all of them together.

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

# --- Define color targets ---
color_targets = [
    {
        "name": "Red",
        "lower1": np.array([0, 120, 70]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 120, 70]),
        "upper2": np.array([180, 255, 255]),
        "bgr": (0, 0, 255),
        "trail": deque(maxlen=64),
    },
    {
        "name": "Green",
        "lower1": np.array([40, 80, 80]),
        "upper1": np.array([80, 255, 255]),
        "lower2": None,
        "upper2": None,
        "bgr": (0, 255, 0),
        "trail": deque(maxlen=64),
    },
    {
        "name": "Blue",
        "lower1": np.array([100, 120, 70]),
        "upper1": np.array([130, 255, 255]),
        "lower2": None,
        "upper2": None,
        "bgr": (255, 0, 0),
        "trail": deque(maxlen=64),
    },
]

# --- Morphological kernel (shared) ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MIN_CONTOUR_AREA = 500

# --- Mask preview state ---
show_mask_idx = 0  # Which color's mask to show in preview

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Pre-process: blur + HSV (ONCE) ---
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        masks = []  # Store masks for preview

        # --- Process each color target ---
        for target in color_targets:
            # Create mask
            mask = cv2.inRange(hsv, target["lower1"], target["upper1"])
            if target["lower2"] is not None:
                mask2 = cv2.inRange(hsv, target["lower2"], target["upper2"])
                mask = cv2.bitwise_or(mask, mask2)

            # Clean mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            masks.append(mask)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centroid = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > MIN_CONTOUR_AREA:
                    # Draw contour
                    cv2.drawContours(frame, [largest], -1, target["bgr"], 2)

                    # Bounding box
                    x, y, w, h = cv2.boundingRect(largest)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), target["bgr"], 1)

                    # Centroid
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centroid = (cx, cy)

                        # Draw centroid and label
                        cv2.circle(frame, centroid, 6, target["bgr"], -1)
                        cv2.putText(frame, target["name"],
                                    (cx + 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    target["bgr"], 2, cv2.LINE_AA)

            # Update trail
            target["trail"].appendleft(centroid)

            # Draw trail with fading
            trail = target["trail"]
            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                thickness = max(1, int((len(trail) - i) / len(trail) * 4))
                alpha = 1.0 - (i / len(trail))
                b, g, r = target["bgr"]
                faded = (int(b * alpha), int(g * alpha), int(r * alpha))
                cv2.line(frame, trail[i - 1], trail[i], faded, thickness)

        # --- Show selected mask preview ---
        if masks:
            idx = show_mask_idx % len(masks)
            preview = cv2.resize(masks[idx], (160, 120))
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            frame[0:120, frame.shape[1] - 160:frame.shape[1]] = preview_bgr
            cv2.putText(frame, f"Mask: {color_targets[idx]['name']}",
                        (frame.shape[1] - 155, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color_targets[idx]["bgr"], 1, cv2.LINE_AA)

        # --- Status dashboard ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        y_offset = 60
        for target in color_targets:
            active = len(target["trail"]) > 0 and target["trail"][0] is not None
            status = "Tracking" if active else "Lost"
            color = target["bgr"] if active else (100, 100, 100)
            cv2.putText(frame, f"{target['name']}: {status}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            y_offset += 22

        cv2.putText(frame, "'m': cycle mask  'q': quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Multi-Color Tracker', frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask_idx = (show_mask_idx + 1) % len(color_targets)

finally:
    cap.release()
    cv2.destroyAllWindows()
```
