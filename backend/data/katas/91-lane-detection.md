---
slug: 91-lane-detection
title: Lane Detection
level: advanced
concepts: [ROI masking, Canny edges, Hough lines, line overlay]
prerequisites: [32-canny-edge-detection, 15-creating-masks]
---

## What Problem Are We Solving?

Self-driving cars and advanced driver assistance systems (ADAS) need to know where the lane markings are on the road. A **lane detection pipeline** takes a camera frame looking forward from a vehicle and identifies the lane boundaries by finding line-like structures in the lower portion of the image.

This pipeline combines region-of-interest masking (to focus on the road), Canny edge detection (to find edges), and the Hough Line Transform (to detect straight lines from those edges). The detected lines are then overlaid on the original frame.

## Step 1: Create a Synthetic Road Image

We simulate a road scene with a gray road surface, dashed lane markings, and a sky/horizon:

```python
road = np.zeros((400, 600, 3), dtype=np.uint8)
road[0:200] = (180, 140, 100)   # Sky
road[200:400] = (80, 80, 80)    # Road
```

White lane lines are drawn converging toward a vanishing point to simulate perspective.

## Step 2: Define a Region of Interest (ROI)

Lane markings only appear in the lower portion of the image. Searching the entire frame wastes computation and introduces false positives from trees, buildings, and signs. An ROI mask limits our search to a trapezoidal region covering the road:

```python
mask = np.zeros_like(edges)
roi_vertices = np.array([[
    (50, height), (250, 220), (350, 220), (550, height)
]], dtype=np.int32)
cv2.fillPoly(mask, roi_vertices, 255)
masked_edges = cv2.bitwise_and(edges, mask)
```

The trapezoid is wider at the bottom (close to the car) and narrower at the top (near the horizon), matching the perspective shape of the road.

## Step 3: Canny Edge Detection

Before applying Hough, we detect edges using the Canny algorithm. A Gaussian blur first reduces noise:

```python
gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
```

## Step 4: Hough Line Transform with cv2.HoughLinesP

`cv2.HoughLinesP` is the **probabilistic Hough Line Transform**. Unlike the standard Hough Transform that returns infinite lines in polar coordinates, `HoughLinesP` returns actual line **segments** with start and end points, making it more practical for lane detection:

```python
lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180,
                         threshold=30, minLineLength=40, maxLineGap=150)
```

| Parameter | Meaning |
|---|---|
| `rho` | Distance resolution in pixels (1 pixel) |
| `theta` | Angle resolution in radians (1 degree = pi/180) |
| `threshold` | Minimum votes (intersections in Hough space) to detect a line |
| `minLineLength` | Minimum length of a line segment to be accepted |
| `maxLineGap` | Maximum gap between segments to treat them as one line |

Each detected line is returned as `[x1, y1, x2, y2]` -- the start and end coordinates.

## Step 5: Separate Left and Right Lanes

Lines are classified as left or right lane based on their slope. Left lane lines have a negative slope (going up-left to down-right in image coordinates), and right lane lines have a positive slope:

```python
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1 + 1e-6)  # avoid division by zero
    if slope < -0.3:
        left_lines.append(line[0])
    elif slope > 0.3:
        right_lines.append(line[0])
```

The threshold of 0.3 filters out nearly-horizontal lines that are unlikely to be lane markings.

## Step 6: Draw and Overlay

Detected lane lines are drawn onto a blank overlay image, which is then blended with the original using `cv2.addWeighted`:

```python
line_image = np.zeros_like(road)
cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
result = cv2.addWeighted(road, 0.8, line_image, 1.0, 0)
```

## The Complete Pipeline

1. **Input**: Forward-facing road image
2. **Grayscale + Blur**: Prepare for edge detection
3. **Canny Edge Detection**: Find all edges
4. **ROI Mask**: Keep only edges in the road region
5. **HoughLinesP**: Detect line segments
6. **Classify Lines**: Separate left vs right lane
7. **Draw + Overlay**: Visualize detected lanes on the original image

## Tips & Common Mistakes

- The ROI vertices must match your image dimensions and road position. If the image size changes, the ROI must be recalculated.
- `minLineLength` and `maxLineGap` in `HoughLinesP` are critical tuning parameters. Too strict and you miss dashed lines; too loose and you pick up noise.
- Filtering by slope is essential. Without it, horizontal edges from the horizon or road cracks appear as lane candidates.
- The `rho` and `theta` resolution affect accuracy. Higher resolution (smaller values) gives more precise lines but is slower.
- In real-world applications, you would average multiple detected segments into a single lane line per side, rather than drawing every detected segment.
- Adding a slight dilation to the Canny output can help bridge small gaps in dashed lane markings.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create a synthetic road scene
# =============================================================
height, width = 400, 600
road = np.zeros((height, width, 3), dtype=np.uint8)

# Sky gradient (top half)
for y in range(200):
    blue = 180 - int(y * 0.3)
    road[y] = (blue, int(blue * 0.8), int(blue * 0.5))

# Road surface (bottom half)
road[200:] = (80, 80, 80)

# Road shoulders
pts_left_shoulder = np.array([[0, height], [0, 250], [140, 200], [50, height]], np.int32)
pts_right_shoulder = np.array([[width, height], [width, 250], [460, 200], [550, height]], np.int32)
cv2.fillPoly(road, [pts_left_shoulder], (50, 100, 50))
cv2.fillPoly(road, [pts_right_shoulder], (50, 100, 50))

# Vanishing point
vp_x, vp_y = 300, 200

# Left lane line (white dashes from vanishing point down-left)
for i in range(5):
    t1 = 0.2 + i * 0.15
    t2 = t1 + 0.08
    x1 = int(vp_x + (120 - vp_x) * t1)
    y1 = int(vp_y + (height - vp_y) * t1)
    x2 = int(vp_x + (120 - vp_x) * t2)
    y2 = int(vp_y + (height - vp_y) * t2)
    cv2.line(road, (x1, y1), (x2, y2), (220, 220, 220), 3)

# Right lane line (white dashes from vanishing point down-right)
for i in range(5):
    t1 = 0.2 + i * 0.15
    t2 = t1 + 0.08
    x1 = int(vp_x + (480 - vp_x) * t1)
    y1 = int(vp_y + (height - vp_y) * t1)
    x2 = int(vp_x + (480 - vp_x) * t2)
    y2 = int(vp_y + (height - vp_y) * t2)
    cv2.line(road, (x1, y1), (x2, y2), (220, 220, 220), 3)

# Center dashed yellow line
for i in range(6):
    t1 = 0.15 + i * 0.13
    t2 = t1 + 0.06
    x1 = int(vp_x + (300 - vp_x) * t1)
    y1 = int(vp_y + (height - vp_y) * t1)
    x2 = int(vp_x + (300 - vp_x) * t2)
    y2 = int(vp_y + (height - vp_y) * t2)
    cv2.line(road, (x1, y1), (x2, y2), (0, 200, 220), 2)

# =============================================================
# Step 2: Convert to grayscale, blur, and detect edges
# =============================================================
gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# =============================================================
# Step 3: Apply ROI mask (trapezoidal region over the road)
# =============================================================
roi_mask = np.zeros_like(edges)
roi_vertices = np.array([[
    (50, height),        # bottom-left
    (240, 210),          # top-left
    (360, 210),          # top-right
    (550, height)        # bottom-right
]], dtype=np.int32)
cv2.fillPoly(roi_mask, roi_vertices, 255)
masked_edges = cv2.bitwise_and(edges, roi_mask)

# =============================================================
# Step 4: Detect lines using HoughLinesP
# =============================================================
lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180,
                         threshold=25, minLineLength=30, maxLineGap=150)

# =============================================================
# Step 5: Classify lines as left or right lane
# =============================================================
left_lines = []
right_lines = []
line_image = np.zeros_like(road)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # skip vertical lines
        slope = (y2 - y1) / (x2 - x1)

        if slope < -0.3:  # Left lane (negative slope in image coords)
            left_lines.append((x1, y1, x2, y2))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        elif slope > 0.3:  # Right lane (positive slope)
            right_lines.append((x1, y1, x2, y2))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

    print(f'Total lines detected: {len(lines)}')
    print(f'Left lane segments: {len(left_lines)}')
    print(f'Right lane segments: {len(right_lines)}')
else:
    print('No lines detected')

# =============================================================
# Step 6: Overlay detected lanes on original image
# =============================================================
lane_overlay = cv2.addWeighted(road, 0.8, line_image, 1.0, 0)

# Draw the ROI boundary for visualization
roi_display = road.copy()
cv2.polylines(roi_display, roi_vertices, True, (0, 255, 255), 2)
cv2.putText(roi_display, 'ROI', (260, 350), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2)

# Convert edges to BGR for display
edges_bgr = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(roi_display, 'ROI Region', (10, 25), font, 0.6, (0, 255, 255), 2)
cv2.putText(edges_bgr, 'Masked Edges', (10, 25), font, 0.6, (0, 255, 0), 2)
cv2.putText(lane_overlay, 'Detected Lanes', (10, 25), font, 0.6, (0, 255, 0), 2)

# Build result grid: top row = ROI + edges, bottom = overlay
# Resize if needed to match
top_row = np.hstack([roi_display, edges_bgr])
# Create a centered display of the lane overlay at same width
lane_padded = np.zeros((height, width * 2, 3), dtype=np.uint8)
x_offset = (width * 2 - width) // 2
lane_padded[:, x_offset:x_offset + width] = lane_overlay
cv2.putText(lane_padded, 'Red=Left Lane  Blue=Right Lane', (150, height - 15),
            font, 0.5, (200, 200, 200), 1)

result = np.vstack([top_row, lane_padded])

cv2.imshow('Lane Detection', result)
```
