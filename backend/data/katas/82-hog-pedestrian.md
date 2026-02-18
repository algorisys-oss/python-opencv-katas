---
slug: 82-hog-pedestrian
title: HOG Pedestrian Detection
level: advanced
concepts: [cv2.HOGDescriptor, detectMultiScale, SVM]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

Detecting people walking in images and video is critical for surveillance systems, autonomous vehicles, and robotics. **HOG (Histogram of Oriented Gradients)** combined with a linear **SVM (Support Vector Machine)** is a classic approach for pedestrian detection. OpenCV provides a built-in pre-trained pedestrian detector via `cv2.HOGDescriptor` that works out of the box — no external model files needed.

## How HOG Features Work

HOG captures the **shape and structure** of objects by computing gradient orientations in localized regions:

1. **Compute gradients**: For each pixel, calculate the magnitude and direction of the intensity gradient.
2. **Divide into cells**: Split the image into small cells (typically 8x8 pixels).
3. **Build histograms**: For each cell, create a histogram of gradient orientations (typically 9 bins covering 0-180 degrees).
4. **Normalize over blocks**: Group cells into blocks (typically 2x2 cells) and normalize the histograms. This provides invariance to lighting changes.
5. **Concatenate**: The final HOG descriptor is the concatenation of all block histograms.

```
Image -> Gradients -> Cell Histograms -> Block Normalization -> Feature Vector -> SVM
```

The resulting feature vector captures the "shape signature" of the detection window, which is then classified by an SVM as pedestrian or non-pedestrian.

## Creating a HOG Descriptor

OpenCV's `cv2.HOGDescriptor()` creates the descriptor with default parameters optimized for pedestrian detection:

```python
hog = cv2.HOGDescriptor()
```

The default HOG parameters are:

| Parameter | Default | Meaning |
|---|---|
| `winSize` | (64, 128) | Detection window size |
| `blockSize` | (16, 16) | Block size for normalization |
| `blockStride` | (8, 8) | Stride between blocks |
| `cellSize` | (8, 8) | Cell size for histograms |
| `nbins` | 9 | Number of orientation bins |

## Loading the Built-in People Detector

The key advantage of OpenCV's HOG pedestrian detector is the **built-in SVM coefficients**. No external model file is needed:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

`cv2.HOGDescriptor_getDefaultPeopleDetector()` returns the pre-trained SVM coefficients for the Dalal-Triggs pedestrian detector. This is trained on 64x128 pedestrian images.

There is also a smaller "Daimler" detector:

```python
# Alternative: Daimler people detector (48x96 window, faster but less accurate)
hog_daimler = cv2.HOGDescriptor((48, 96), (16, 16), (8, 8), (8, 8), 9)
hog_daimler.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
```

## Detecting Pedestrians with detectMultiScale

Once the SVM detector is set, use `detectMultiScale()` to find people:

```python
boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),     # Step size for sliding window
    padding=(4, 4),       # Padding around detection window
    scale=1.05,           # Scale factor for image pyramid
    useMeanshiftGrouping=False
)
```

| Parameter | Meaning |
|---|---|
| `winStride` | Sliding window step. Smaller = more thorough, slower |
| `padding` | Padding around the detection window |
| `scale` | Image pyramid scale factor (like scaleFactor in cascades) |
| `useMeanshiftGrouping` | Use mean shift to merge overlapping detections |

The method returns:
- `boxes`: Array of `(x, y, w, h)` bounding boxes
- `weights`: Confidence scores for each detection

## Filtering and Drawing Detections

```python
boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(4, 4), scale=1.05)

# Filter by confidence
for (x, y, w, h), weight in zip(boxes, weights):
    if weight > 0.5:  # Confidence threshold
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{weight:.2f}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

## Non-Maximum Suppression (NMS)

HOG often produces overlapping detections. You can apply NMS to keep only the best:

```python
# Simple NMS using OpenCV's groupRectangles (built-in)
# Or manual overlap-based suppression:
def non_max_suppression(boxes, weights, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = np.argsort(weights.flatten())[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[1:]]
        inds = np.where(overlap <= overlap_thresh)[0]
        order = order[inds + 1]
    return keep
```

## Tips & Common Mistakes

- The default people detector expects a **64x128 detection window**. Very small or very distant pedestrians may be missed.
- `winStride` dramatically affects speed. `(4,4)` is thorough but slow; `(8,8)` is a good balance; `(16,16)` is fast but may miss people.
- HOG works on **any image type** (grayscale or color). It internally computes gradients.
- The detector returns many overlapping boxes. Always apply NMS or confidence filtering.
- Resize large images before detection for better performance. A width of 400-800 pixels is usually sufficient.
- `useMeanshiftGrouping=True` provides built-in grouping but can be slower. The default `False` with manual NMS often works better.
- HOG pedestrian detection is a **classic** method. For production use, DNN-based detectors (YOLO, SSD) are more accurate but HOG requires no model files.

## Starter Code

```python
import cv2
import numpy as np

# --- Initialize HOG People Detector (built-in, no model file needed) ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- Create a synthetic scene with pedestrian-like shapes ---
canvas = np.zeros((500, 800, 3), dtype=np.uint8)

# Background: simple outdoor scene
cv2.rectangle(canvas, (0, 300), (800, 500), (60, 100, 60), -1)    # Ground
canvas[0:300] = (180, 140, 100)                                    # Sky

# Draw stick-figure "pedestrians" (synthetic shapes)
def draw_pedestrian_shape(img, cx, ground_y, height, color):
    """Draw a simplified pedestrian-like shape."""
    w = height // 3
    top_y = ground_y - height
    # Body rectangle
    cv2.rectangle(img, (cx - w // 2, top_y + height // 5),
                  (cx + w // 2, ground_y), color, -1)
    # Head circle
    head_r = height // 7
    cv2.circle(img, (cx, top_y + head_r), head_r, color, -1)
    # Legs
    cv2.line(img, (cx - w // 4, ground_y), (cx - w // 3, ground_y + height // 6), color, 3)
    cv2.line(img, (cx + w // 4, ground_y), (cx + w // 3, ground_y + height // 6), color, 3)
    return (cx - w, top_y, w * 2, height + height // 6)

ped1_box = draw_pedestrian_shape(canvas, 150, 380, 160, (80, 80, 120))
ped2_box = draw_pedestrian_shape(canvas, 400, 400, 200, (100, 90, 80))
ped3_box = draw_pedestrian_shape(canvas, 600, 370, 140, (90, 100, 110))

# Add some non-pedestrian objects
cv2.rectangle(canvas, (700, 250), (780, 350), (100, 120, 80), -1)  # Building
cv2.circle(canvas, (50, 100), 40, (0, 200, 255), -1)               # Sun

# --- Run HOG detection on the synthetic image ---
# Note: HOG is trained on real pedestrian images, so it may not detect
# these simplified shapes. This demonstrates the API pattern.
boxes, weights = hog.detectMultiScale(
    canvas,
    winStride=(8, 8),
    padding=(4, 4),
    scale=1.05
)

print(f'HOG detected {len(boxes)} regions')

# --- Visualize results ---
result = canvas.copy()

# Draw actual HOG detections (if any)
for (x, y, w, h), weight in zip(boxes, weights):
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(result, f'{weight[0]:.2f}', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Also show the synthetic pedestrian regions for reference
for i, box in enumerate([ped1_box, ped2_box, ped3_box]):
    bx, by, bw, bh = box
    cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 200, 255), 1)
    cv2.putText(result, f'Shape {i+1}', (bx, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

# --- Info panel ---
info = np.zeros((180, 800, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'HOG Pedestrian Detection', (10, 25),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'API: hog = cv2.HOGDescriptor()', (20, 50),
            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())', (20, 73),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'boxes, weights = hog.detectMultiScale(img, winStride, padding, scale)', (20, 96),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, f'Detections: {len(boxes)} (on synthetic data)', (20, 125),
            font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(info, 'Note: HOG is trained on real photos; synthetic shapes may not trigger', (20, 150),
            font, 0.4, (150, 150, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'detection. With real pedestrian images, boxes would appear around people.', (20, 170),
            font, 0.4, (150, 150, 255), 1, cv2.LINE_AA)

# Legend
cv2.rectangle(info, (580, 10), (600, 25), (0, 255, 0), -1)
cv2.putText(info, 'HOG Detection', (605, 24), font, 0.35, (200, 200, 200), 1)
cv2.rectangle(info, (580, 30), (600, 45), (0, 200, 255), -1)
cv2.putText(info, 'Synthetic Shape', (605, 44), font, 0.35, (200, 200, 200), 1)

cv2.putText(result, 'HOG Pedestrian Detection', (10, 25),
            font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

output = np.vstack([result, info])

print('\nHOG Descriptor Parameters:')
print(f'  Window size: {hog.winSize}')
print(f'  Block size: {hog.blockSize}')
print(f'  Block stride: {hog.blockStride}')
print(f'  Cell size: {hog.cellSize}')
print(f'  Nbins: {hog.nbins}')

cv2.imshow('HOG Pedestrian Detection', output)
```
