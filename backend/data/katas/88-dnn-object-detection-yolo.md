---
slug: 88-dnn-object-detection-yolo
title: "DNN: Object Detection (YOLO)"
level: advanced
concepts: [YOLO architecture, NMS, cv2.dnn.NMSBoxes]
prerequisites: [87-dnn-object-detection-ssd]
---

## What Problem Are We Solving?

**YOLO (You Only Look Once)** is one of the fastest and most popular object detection architectures. Unlike SSD's pre-formatted output, YOLO outputs raw detection tensors that require more post-processing: parsing class probabilities, extracting bounding boxes, and applying **Non-Maximum Suppression (NMS)** to eliminate duplicate detections. Understanding YOLO's output format and the NMS step is essential for using YOLO with OpenCV's DNN module.

## How YOLO Differs from SSD

Both are single-shot detectors, but their output formats differ significantly:

| Aspect | SSD | YOLO |
|---|---|---|
| Output format | `(1, 1, N, 7)` — pre-filtered | Raw grid predictions — needs parsing |
| Coordinates | Normalized (0-1) directly | Center x, y, width, height (normalized) |
| NMS | Done internally | Must be applied manually |
| Class scores | Single confidence per detection | Per-class probabilities |

## Loading YOLO in OpenCV

YOLO uses the Darknet framework format (`.cfg` + `.weights`):

```python
# In production, load YOLO model:
# net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Or for YOLOv4:
# net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Or YOLOv3-tiny (faster, less accurate):
# net = cv2.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')
```

## Getting YOLO Output Layer Names

YOLO has multiple output layers (one per detection scale). You need to identify them:

```python
# Get all layer names
# layer_names = net.getLayerNames()

# Get output layer names (unconnected layers)
# output_layers = net.getUnconnectedOutLayersNames()
# For YOLOv3: ['yolo_82', 'yolo_94', 'yolo_106'] (3 scales)
```

## YOLO Preprocessing

```python
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,   # Normalize to [0, 1]
    size=(416, 416),        # Common sizes: 320, 416, 608
    mean=(0, 0, 0),         # No mean subtraction
    swapRB=True,            # BGR to RGB
    crop=False
)
# net.setInput(blob)
# outputs = net.forward(output_layers)
```

Input size affects speed vs accuracy:
- **320x320**: Fast, lower accuracy
- **416x416**: Balanced (default)
- **608x608**: Slow, highest accuracy

## Understanding YOLO's Output Format

Each output layer returns a 2D array of shape `(num_detections, 5 + num_classes)`:

```python
# For COCO (80 classes), each row has 85 values:
# [center_x, center_y, width, height, objectness, class1_prob, class2_prob, ..., class80_prob]

# outputs is a list of arrays, one per scale
# outputs[0].shape might be (507, 85) for a 416x416 input
# outputs[1].shape might be (2028, 85)
# outputs[2].shape might be (8112, 85)
```

The first 5 values per detection:
- `center_x`: Center X of bounding box (normalized 0-1)
- `center_y`: Center Y of bounding box (normalized 0-1)
- `width`: Box width (normalized 0-1)
- `height`: Box height (normalized 0-1)
- `objectness`: Confidence that an object exists in this cell

## Parsing YOLO Detections

The complete parsing pipeline:

```python
def parse_yolo_output(outputs, img_w, img_h, conf_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]          # Class probabilities
            class_id = np.argmax(scores)
            class_confidence = scores[class_id]
            objectness = detection[4]

            # Final confidence = objectness * class probability
            confidence = objectness * class_confidence

            if confidence > conf_threshold:
                # Convert from center format to corner format
                center_x = int(detection[0] * img_w)
                center_y = int(detection[1] * img_h)
                w = int(detection[2] * img_w)
                h = int(detection[3] * img_h)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids
```

## Non-Maximum Suppression with cv2.dnn.NMSBoxes

YOLO produces many overlapping detections. **NMS** keeps only the best box for each object:

```python
indices = cv2.dnn.NMSBoxes(
    bboxes=boxes,           # List of [x, y, w, h]
    scores=confidences,     # List of confidence scores
    score_threshold=0.5,    # Minimum confidence to consider
    nms_threshold=0.4       # IoU threshold for suppression
)
```

How NMS works:
1. Sort all detections by confidence (highest first)
2. Take the highest-confidence box as a kept detection
3. Remove all remaining boxes with IoU > `nms_threshold` with the kept box
4. Repeat until no boxes remain

The `nms_threshold` (IoU threshold) controls how much overlap is allowed:
- **0.3**: Aggressive suppression — fewer overlapping boxes
- **0.4**: Balanced default
- **0.5**: Lenient — allows more overlapping detections

```python
# Draw only NMS-surviving detections
for i in indices:
    x, y, w, h = boxes[i]
    label = f'{class_names[class_ids[i]]}: {confidences[i]:.2f}'
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

## COCO Class Names

YOLO models trained on COCO have 80 classes:

```python
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
```

## Tips & Common Mistakes

- YOLO output coordinates use **center format** `(cx, cy, w, h)`, not corner format. Convert before drawing: `x = cx - w/2`, `y = cy - h/2`.
- Always apply NMS after parsing. Without NMS, each object will have dozens of overlapping boxes.
- The confidence is `objectness * class_probability`, not just one of them. Some implementations forget to multiply.
- `cv2.dnn.NMSBoxes` expects boxes as `[x, y, w, h]` (top-left corner + size), not `[x1, y1, x2, y2]`.
- YOLO input size must be a multiple of 32. The default 416x416 works well; 608x608 gives better accuracy.
- Larger input size = slower but more accurate, especially for small objects.
- YOLOv3-tiny is 10x faster than YOLOv3 but significantly less accurate.
- The `indices` returned by `NMSBoxes` may be a numpy array or a tuple depending on OpenCV version. Use `indices.flatten()` to be safe.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic scene ---
img = np.zeros((500, 700, 3), dtype=np.uint8)
img[:] = (190, 200, 190)

# Ground and sky
img[0:250] = (200, 170, 140)  # Sky
img[350:500] = (80, 130, 80)  # Grass
cv2.rectangle(img, (0, 250), (700, 350), (140, 160, 140), -1)  # Road

# Draw objects
# Car
cv2.rectangle(img, (50, 260), (180, 330), (200, 60, 60), -1)
cv2.rectangle(img, (75, 230), (155, 260), (200, 60, 60), -1)
cv2.circle(img, (80, 335), 15, (30, 30, 30), -1)
cv2.circle(img, (155, 335), 15, (30, 30, 30), -1)

# Person
cv2.circle(img, (300, 210), 18, (200, 170, 150), -1)
cv2.rectangle(img, (285, 228), (315, 310), (50, 50, 150), -1)
cv2.rectangle(img, (288, 310), (300, 360), (70, 70, 70), -1)
cv2.rectangle(img, (302, 310), (314, 360), (70, 70, 70), -1)

# Dog
cv2.ellipse(img, (470, 320), (40, 22), 0, 0, 360, (130, 100, 60), -1)
cv2.circle(img, (510, 308), 13, (130, 100, 60), -1)

# Bicycle
cv2.circle(img, (580, 310), 25, (100, 100, 100), 2)
cv2.circle(img, (640, 310), 25, (100, 100, 100), 2)
cv2.line(img, (580, 310), (610, 270), (100, 100, 100), 2)
cv2.line(img, (640, 310), (610, 270), (100, 100, 100), 2)

# --- Demonstrate YOLO preprocessing ---
blob = cv2.dnn.blobFromImage(
    img, scalefactor=1/255.0, size=(416, 416),
    mean=(0, 0, 0), swapRB=True, crop=False
)
print(f'YOLO input blob shape: {blob.shape}')

# In production:
# net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net.setInput(blob)
# output_layers = net.getUnconnectedOutLayersNames()
# outputs = net.forward(output_layers)

# --- Simulate YOLO output (raw format) ---
# YOLO output: list of arrays, each row = [cx, cy, w, h, objectness, class_probs...]
# Using 80 COCO classes
num_classes = 80
img_h, img_w = img.shape[:2]

# Simulate raw detections (including duplicates that NMS will handle)
raw_detections = []

# Car detections (class 2) - multiple overlapping boxes
for dx, dy, conf in [(0, 0, 0.91), (5, 3, 0.85), (-3, -2, 0.78)]:
    det = np.zeros(5 + num_classes, dtype=np.float32)
    det[0] = (115 + dx) / img_w   # center_x
    det[1] = (280 + dy) / img_h   # center_y
    det[2] = 130 / img_w           # width
    det[3] = 100 / img_h           # height
    det[4] = conf                   # objectness
    det[5 + 2] = 0.95              # car class probability
    raw_detections.append(det)

# Person detections (class 0) - multiple overlapping
for dx, dy, conf in [(0, 0, 0.88), (4, 2, 0.80)]:
    det = np.zeros(5 + num_classes, dtype=np.float32)
    det[0] = (300 + dx) / img_w
    det[1] = (280 + dy) / img_h
    det[2] = 35 / img_w
    det[3] = 155 / img_h
    det[4] = conf
    det[5 + 0] = 0.92
    raw_detections.append(det)

# Dog detection (class 16)
det = np.zeros(5 + num_classes, dtype=np.float32)
det[0] = 485 / img_w
det[1] = 315 / img_h
det[2] = 80 / img_w
det[3] = 50 / img_h
det[4] = 0.72
det[5 + 16] = 0.88
raw_detections.append(det)

# Bicycle detection (class 1)
det = np.zeros(5 + num_classes, dtype=np.float32)
det[0] = 610 / img_w
det[1] = 295 / img_h
det[2] = 80 / img_w
det[3] = 60 / img_h
det[4] = 0.68
det[5 + 1] = 0.85
raw_detections.append(det)

# Package as YOLO output format (list of arrays)
simulated_outputs = [np.array(raw_detections)]

# --- Parse YOLO output ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
]  # First 20 of 80

conf_threshold = 0.5
nms_threshold = 0.4

boxes = []
confidences = []
class_ids = []

for output in simulated_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        class_conf = scores[class_id]
        objectness = detection[4]
        confidence = objectness * class_conf

        if confidence > conf_threshold:
            cx = int(detection[0] * img_w)
            cy = int(detection[1] * img_h)
            w = int(detection[2] * img_w)
            h = int(detection[3] * img_h)
            x = cx - w // 2
            y = cy - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(f'\nBefore NMS: {len(boxes)} detections')

# --- Apply Non-Maximum Suppression ---
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print(f'After NMS:  {len(indices)} detections')

# --- Draw results ---
result = img.copy()

# Colors per class
np.random.seed(42)
colors = {i: tuple(int(c) for c in np.random.randint(80, 255, 3)) for i in range(num_classes)}

for i in indices:
    x, y, w, h = boxes[i]
    class_id = class_ids[i]
    conf = confidences[i]
    color = colors[class_id]

    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'class_{class_id}'
    label = f'{class_name}: {conf:.2f}'

    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result, (x, y - th - 8), (x + tw + 4, y), color, -1)
    cv2.putText(result, label, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# --- Show NMS before/after comparison ---
before_nms = img.copy()
for i, (box, conf, cid) in enumerate(zip(boxes, confidences, class_ids)):
    x, y, w, h = box
    cv2.rectangle(before_nms, (x, y), (x + w, y + h), (0, 150, 255), 1)

cv2.putText(before_nms, f'Before NMS: {len(boxes)} boxes', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)
cv2.putText(result, f'After NMS: {len(indices)} boxes', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# Info panel
info = np.zeros((110, 700, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'YOLO Output Format: [cx, cy, w, h, objectness, class_probs...]', (10, 25),
            font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'Confidence = objectness * max(class_probs)', (10, 48),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, f'NMS: cv2.dnn.NMSBoxes(boxes, scores, {conf_threshold}, {nms_threshold})', (10, 71),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, f'Reduced {len(boxes)} -> {len(indices)} detections by removing overlaps', (10, 98),
            font, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

top_row = np.hstack([before_nms, result])
# Resize info to match
info_resized = cv2.resize(info, (top_row.shape[1], 110))
output = np.vstack([top_row, info_resized])

print('\nFinal detections:')
for i in indices:
    cname = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else f'class_{class_ids[i]}'
    print(f'  {cname}: {confidences[i]:.2f} at {boxes[i]}')

cv2.imshow('DNN Object Detection (YOLO)', output)
```
