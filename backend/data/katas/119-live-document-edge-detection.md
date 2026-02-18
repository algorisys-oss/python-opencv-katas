---
slug: 119-live-document-edge-detection
title: Live Document Edge Detection
level: live
concepts: [contour detection, polygon approximation, perspective guide, cv2.approxPolyDP, document scanning]
prerequisites: [100-live-camera-fps, 38-finding-contours, 42-contour-approximation, 90-document-scanner]
---

## What Problem Are We Solving?

Mobile document scanning apps like CamScanner and Adobe Scan detect the edges of a document in real-time as you hold your phone over it. A green outline snaps to the document boundary, and when the detection is stable, the app signals "ready to capture." This real-time edge detection is the critical first step before the perspective warp that produces a clean, flat scan.

Building this live requires solving several challenges that do not exist in single-image scanning: the document may be partially visible, the camera may be moving, lighting may change between frames, and the detected contour may jump around erratically. A production-quality scanner needs **stability detection** -- it should only signal "ready" when the same four corners have been consistent across multiple consecutive frames, indicating the camera is steady and the detection is reliable.

This kata combines edge detection, contour finding, polygon approximation, corner ordering, and frame-to-frame stability tracking into a live camera pipeline. The result is a real-time document boundary detector with a perspective guide overlay that tells the user when the detection is stable enough to capture.

## The Detection Pipeline Per Frame

Each frame goes through a multi-step pipeline to find the document boundary:

```
Frame -> Grayscale -> Blur -> Canny Edges -> Dilate -> Find Contours
     -> Sort by Area -> Approximate to Polygon -> Filter for 4 Points
     -> Order Corners -> Draw Overlay
```

### Step 1: Preprocessing

Convert to grayscale and apply Gaussian blur to suppress noise before edge detection:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

The blur kernel size affects sensitivity. A `(5, 5)` kernel is a good default. Larger kernels (7x7, 9x9) suppress more noise but may blur thin document edges.

### Step 2: Canny Edge Detection

```python
edges = cv2.Canny(blurred, 50, 150)
```

| Parameter | Value | Effect |
|---|---|---|
| `threshold1` (low) | 50 | Edges below this gradient magnitude are discarded |
| `threshold2` (high) | 150 | Edges above this are always kept; between low and high, kept only if connected to strong edges |

Lower thresholds detect more edges (including noise). Higher thresholds are stricter. A 1:3 ratio between low and high thresholds is a standard starting point.

### Step 3: Dilation

Close small gaps in the edge map so that the document boundary forms a complete, closed contour:

```python
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
```

Without dilation, small breaks in the edge (from low contrast at one side, or a rounded corner) can prevent `findContours` from seeing the document as a single closed contour.

## Finding the Largest 4-Point Contour

After edge detection, find all contours and look for the largest one that approximates to a quadrilateral:

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Top 5 by area

doc_contour = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc_contour = approx
        break
```

### Why Sort by Area?

The document should be the largest object in the scene (or at least the largest quadrilateral). Sorting by area and checking the top candidates first avoids wasting time on small, irrelevant contours.

### Understanding cv2.approxPolyDP

`cv2.approxPolyDP` simplifies a contour by removing points that are close to the line between their neighbors, controlled by an epsilon tolerance:

```python
approx = cv2.approxPolyDP(contour, epsilon, closed)
```

| Parameter | Meaning |
|---|---|
| `contour` | Input contour (array of points) |
| `epsilon` | Maximum distance a point can be from the simplified line to be removed |
| `closed` | `True` if the contour is a closed shape |

The epsilon is typically set as a fraction of the contour's perimeter:

| Epsilon | Effect |
|---|---|
| `0.01 * perimeter` | Tight approximation -- keeps more points, may not simplify to 4 |
| `0.02 * perimeter` | Standard -- good balance for document detection |
| `0.05 * perimeter` | Aggressive -- simplifies heavily, may collapse valid corners |
| `0.10 * perimeter` | Very aggressive -- rarely useful, distorts the shape |

If `0.02` does not produce a 4-point result, you can iterate over a range of epsilon values:

```python
for eps_mult in [0.02, 0.03, 0.04, 0.05]:
    approx = cv2.approxPolyDP(c, eps_mult * peri, True)
    if len(approx) == 4:
        doc_contour = approx
        break
```

## Ordering the Four Corners

The four corners from `approxPolyDP` come in an arbitrary order. For a perspective transform, you need them in a consistent order: top-left, top-right, bottom-right, bottom-left. The sum-and-difference method is robust:

```python
def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)        # x + y
    d = np.diff(pts, axis=1).flatten()  # y - x

    ordered[0] = pts[np.argmin(s)]   # Top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # Bottom-right: largest x+y
    ordered[1] = pts[np.argmin(d)]   # Top-right: smallest y-x
    ordered[3] = pts[np.argmax(d)]   # Bottom-left: largest y-x

    return ordered
```

| Corner | Heuristic | Why |
|---|---|---|
| Top-left | Smallest `x + y` | Closest to origin (0, 0) |
| Bottom-right | Largest `x + y` | Farthest from origin |
| Top-right | Smallest `y - x` | High x, low y |
| Bottom-left | Largest `y - x` | Low x, high y |

## Stability Detection

In a live feed, the detected contour jumps around due to noise, camera shake, and auto-exposure adjustments. A "ready to capture" signal should only appear when the detection has been consistent for several frames.

Track stability by comparing the current corners to the previous frame's corners:

```python
def corners_are_stable(current, previous, threshold=10.0):
    """Check if corners moved less than threshold pixels."""
    if previous is None:
        return False
    diff = np.linalg.norm(current - previous, axis=1)
    return np.all(diff < threshold)
```

Count consecutive stable frames:

```python
stable_count = 0
STABLE_THRESHOLD = 15  # Frames needed to signal "ready"

# In the loop:
if doc_contour is not None:
    ordered = order_points(doc_contour)
    if corners_are_stable(ordered, prev_corners, threshold=10.0):
        stable_count += 1
    else:
        stable_count = 0
    prev_corners = ordered
else:
    stable_count = 0
    prev_corners = None

is_ready = stable_count >= STABLE_THRESHOLD
```

| Parameter | Typical Value | Effect |
|---|---|---|
| `threshold` (pixels) | 8-15 | How much corner movement is tolerated per frame |
| `STABLE_THRESHOLD` (frames) | 10-20 | How many stable frames before signaling "ready" |

Lower pixel threshold = stricter stability requirement. Higher frame count = longer wait but more confident result.

## Drawing the Perspective Guide Overlay

When a document is detected, draw visual feedback showing the detected boundary and stability status:

```python
# Color based on stability
if is_ready:
    color = (0, 255, 0)     # Green = ready to capture
elif doc_contour is not None:
    color = (0, 255, 255)   # Yellow = detected but not stable
else:
    color = (0, 0, 200)     # Red = nothing found

# Draw filled semi-transparent polygon
if doc_contour is not None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [doc_contour], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [doc_contour], True, color, 3)

    # Draw corner markers
    for pt in doc_contour.reshape(-1, 2):
        cv2.circle(frame, tuple(pt), 8, color, -1)
```

## Tips & Common Mistakes

- The `0.02 * perimeter` epsilon for `approxPolyDP` is a starting point. If documents with rounded corners are not detected, try `0.03` or `0.04`.
- Always sort contours by area largest-first. The document should dominate the scene, so checking the largest contours first is both correct and efficient.
- Dilation of the edge map is critical. Without it, small breaks in the document boundary cause `findContours` to see multiple open contours instead of one closed quadrilateral.
- The corner ordering method (sum/difference) assumes the camera is roughly upright. If the camera is rotated 90 degrees, the "top-left" will be wrong -- but the perspective warp will still produce a correct result as long as the ordering is consistent.
- Stability detection prevents the "ready" signal from firing during camera movement. Without it, a single good frame during a pan would trigger a capture.
- Set a minimum contour area threshold (e.g., 10% of frame area) to ignore small quadrilaterals that are not the document.
- Canny thresholds may need adjustment for different lighting conditions. Brighter scenes may need higher thresholds (75, 200); darker scenes may need lower (30, 100).
- If the document has a dark border on a dark background (e.g., a dark phone screen on a dark desk), Canny will not find the edge. Ensure contrast between the document and the surface.
- Gaussian blur kernel size must be odd. Use `(5, 5)` as default, increase to `(7, 7)` for noisier cameras.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Place a document (paper, book, or card) on a contrasting surface in front of the camera — a colored polygon overlay should snap to its edges
- Hold the camera steady — the overlay changes from yellow ("Hold steady...") to green ("READY") as the stability progress bar fills up
- When the overlay turns green, press **c** to capture — a new window shows the perspective-corrected document scan

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

# --- Stability tracking ---
prev_corners = None
stable_count = 0
STABLE_FRAMES_NEEDED = 15
CORNER_MOVE_THRESHOLD = 12.0  # pixels
MIN_AREA_RATIO = 0.05  # Document must be at least 5% of frame area


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def corners_stable(current, previous, threshold):
    """Check if all corners moved less than threshold pixels."""
    if previous is None:
        return False
    diffs = np.linalg.norm(current - previous, axis=1)
    return np.all(diffs < threshold)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]
        frame_area = h * w

        # --- Step 1: Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- Step 2: Canny edge detection ---
        edges = cv2.Canny(blurred, 50, 150)

        # --- Step 3: Dilate to close gaps ---
        dilate_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)

        # --- Step 4: Find contours, sort by area ---
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # --- Step 5: Find largest 4-point contour ---
        doc_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * MIN_AREA_RATIO:
                continue
            peri = cv2.arcLength(c, True)
            # Try multiple epsilon values for robustness
            for eps_mult in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(c, eps_mult * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break
            if doc_contour is not None:
                break

        # --- Step 6: Stability detection ---
        is_ready = False
        if doc_contour is not None:
            ordered = order_points(doc_contour)
            if corners_stable(ordered, prev_corners, CORNER_MOVE_THRESHOLD):
                stable_count += 1
            else:
                stable_count = 0
            prev_corners = ordered.copy()
            is_ready = stable_count >= STABLE_FRAMES_NEEDED
        else:
            stable_count = 0
            prev_corners = None

        # --- Step 7: Draw overlay ---
        if doc_contour is not None:
            pts = doc_contour.reshape(-1, 2).astype(int)

            if is_ready:
                color = (0, 255, 0)       # Green = stable, ready
                label = "READY - Press 'c' to capture"
                label_color = (0, 255, 0)
            else:
                color = (0, 200, 255)     # Yellow = detected, not stable
                label = "Hold steady..."
                label_color = (0, 200, 255)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

            # Polygon outline
            cv2.polylines(frame, [pts], True, color, 3)

            # Corner markers with labels
            corner_labels = ["TL", "TR", "BR", "BL"]
            ordered_int = ordered.astype(int)
            for i, (cx, cy) in enumerate(ordered_int):
                cv2.circle(frame, (cx, cy), 7, color, -1)
                cv2.putText(frame, corner_labels[i], (cx + 10, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            # Status label
            cv2.putText(frame, label, (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

            # Stability progress bar
            bar_w = 200
            bar_h = 10
            bar_x = 10
            bar_y = h - 35
            progress = min(stable_count / STABLE_FRAMES_NEEDED, 1.0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
        else:
            cv2.putText(frame, "No document detected", (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)

        # --- Show edge detection preview (small, in corner) ---
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        preview_w, preview_h = 160, 120
        edge_small = cv2.resize(edge_bgr, (preview_w, preview_h))
        frame[0:preview_h, w - preview_w:w] = edge_small
        cv2.putText(frame, "Edges", (w - preview_w + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # --- FPS and info overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Document Edge Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and is_ready:
            # Capture: perform perspective warp
            ordered_pts = order_points(doc_contour)
            w1 = np.linalg.norm(ordered_pts[1] - ordered_pts[0])
            w2 = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
            out_w = int(max(w1, w2))
            h1 = np.linalg.norm(ordered_pts[3] - ordered_pts[0])
            h2 = np.linalg.norm(ordered_pts[2] - ordered_pts[1])
            out_h = int(max(h1, h2))

            target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered_pts, target)
            warped = cv2.warpPerspective(frame, M, (out_w, out_h))

            cv2.imshow('Captured Document', warped)
            print(f"Document captured! Size: {out_w}x{out_h}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
