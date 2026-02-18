---
slug: 87-dnn-object-detection-ssd
title: "DNN: Object Detection (SSD)"
level: advanced
concepts: [SSD architecture, bounding boxes, confidence filtering]
prerequisites: [85-dnn-loading-models]
---

## What Problem Are We Solving?

Unlike classification (which answers "what is in this image?"), **object detection** answers "what objects are where?" It produces **bounding boxes** around each detected object along with a class label and confidence score. **SSD (Single Shot MultiBox Detector)** is a popular architecture that performs detection in a single forward pass, making it fast enough for real-time use. OpenCV's DNN module can load pre-trained SSD models from TensorFlow or Caffe.

## How SSD Works

SSD processes the image at multiple scales simultaneously using **feature maps** of decreasing resolution:

```
Input Image (300x300)
    |
CNN Backbone (e.g., MobileNet, VGG)
    |
Multi-scale Feature Maps:
    38x38 -> detects small objects
    19x19 -> detects medium objects
    10x10 -> detects medium-large objects
    5x5   -> detects large objects
    3x3   -> detects very large objects
    1x1   -> detects image-filling objects
    |
Each cell predicts: [class_scores, box_offsets] x num_anchors
    |
Non-Maximum Suppression
    |
Final Detections: (classId, confidence, x1, y1, x2, y2)
```

The "Single Shot" means detection happens in **one forward pass** (unlike R-CNN which uses a separate region proposal step).

## Loading an SSD Model

Common SSD models for OpenCV:

```python
# MobileNet-SSD (Caffe) — fast, lightweight
# net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
#                                 'MobileNetSSD_deploy.caffemodel')

# SSD Inception V2 (TensorFlow) — more accurate
# net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
#                                      'ssd_inception_v2_coco.pbtxt')

# Universal loader
# net = cv2.dnn.readNet('model_file', 'config_file')
```

## The SSD Detection Pipeline

```python
# 1. Preprocess
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/127.5,
    size=(300, 300),
    mean=(127.5, 127.5, 127.5),
    swapRB=True
)

# 2. Forward pass
# net.setInput(blob)
# detections = net.forward()
# detections shape: (1, 1, N, 7)
# Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
```

## Interpreting the SSD Output

The output tensor has shape `(1, 1, N, 7)` where N is the number of detections. Each detection is a 7-element vector:

```python
# detections[0, 0, i] = [batch_id, class_id, confidence, x1, y1, x2, y2]
# Coordinates are NORMALIZED (0.0 to 1.0) — multiply by image dimensions

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])
        # Scale normalized coordinates to image size
        x1 = int(detections[0, 0, i, 3] * img_w)
        y1 = int(detections[0, 0, i, 4] * img_h)
        x2 = int(detections[0, 0, i, 5] * img_w)
        y2 = int(detections[0, 0, i, 6] * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

## Confidence Filtering

Not all detections are valid. Filter by confidence threshold:

```python
confidence_threshold = 0.5

valid_detections = []
for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])
    if confidence > confidence_threshold:
        class_id = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7]
        valid_detections.append((class_id, confidence, box))
```

Typical thresholds:
- **0.3**: Catches more objects but more false positives
- **0.5**: Good balance (most common default)
- **0.7**: Conservative, fewer false positives

## Drawing Detection Results

A complete drawing function with class labels and confidence:

```python
# COCO class names (for models trained on COCO dataset)
COCO_CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow']  # ... 80 total

def draw_detections(img, detections, threshold=0.5):
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            class_id = int(detections[0, 0, i, 1])
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            label = f'{COCO_CLASSES[class_id]}: {confidence:.2f}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

## MobileNet-SSD Class Labels

The MobileNet-SSD model trained on VOC has 21 classes:

```python
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]
```

## Tips & Common Mistakes

- SSD output coordinates are **normalized** (0 to 1). You must multiply by image width/height to get pixel coordinates.
- The `class_id` is an integer index into the class label list. Index 0 is typically "background".
- Always filter by confidence. Without filtering, you get hundreds of low-confidence garbage detections.
- Different SSD models need different preprocessing. MobileNet-SSD (Caffe) uses `mean=(127.5, 127.5, 127.5), scale=1/127.5`. Always check the model's documentation.
- The output shape `(1, 1, N, 7)` is specific to SSD. Other architectures (YOLO, Faster R-CNN) have different output formats.
- SSD is faster than two-stage detectors (Faster R-CNN) but typically less accurate on small objects.
- In a sandboxed environment, you can demonstrate the full pipeline using synthetic detection output to understand the data format.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic scene with objects ---
img = np.zeros((500, 700, 3), dtype=np.uint8)
img[:] = (180, 200, 180)

# Draw ground
cv2.rectangle(img, (0, 350), (700, 500), (80, 140, 80), -1)

# Draw a "car" (simplified)
cv2.rectangle(img, (50, 280), (200, 350), (180, 50, 50), -1)    # Body
cv2.rectangle(img, (80, 240), (170, 280), (180, 50, 50), -1)    # Top
cv2.circle(img, (90, 355), 18, (40, 40, 40), -1)                 # Wheel
cv2.circle(img, (170, 355), 18, (40, 40, 40), -1)                # Wheel

# Draw a "person" (simplified)
cv2.circle(img, (350, 220), 20, (200, 170, 150), -1)             # Head
cv2.rectangle(img, (335, 240), (365, 330), (60, 60, 160), -1)    # Body
cv2.rectangle(img, (338, 330), (352, 380), (80, 80, 80), -1)     # Left leg
cv2.rectangle(img, (353, 330), (367, 380), (80, 80, 80), -1)     # Right leg

# Draw a "dog" (simplified)
cv2.ellipse(img, (530, 325), (45, 25), 0, 0, 360, (140, 110, 70), -1)  # Body
cv2.circle(img, (575, 310), 15, (140, 110, 70), -1)                      # Head
cv2.rectangle(img, (490, 335), (500, 370), (140, 110, 70), -1)           # Legs
cv2.rectangle(img, (555, 335), (565, 370), (140, 110, 70), -1)

# Draw a "bottle" on a table
cv2.rectangle(img, (620, 200), (680, 260), (120, 100, 80), -1)   # Table top
cv2.rectangle(img, (640, 140), (660, 200), (200, 200, 200), -1)  # Bottle
cv2.rectangle(img, (643, 130), (657, 140), (200, 200, 200), -1)  # Cap

# --- Demonstrate SSD preprocessing ---
blob = cv2.dnn.blobFromImage(
    img, scalefactor=1/127.5, size=(300, 300),
    mean=(127.5, 127.5, 127.5), swapRB=True
)
print(f'SSD input blob shape: {blob.shape}')

# In production:
# net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt',
#                                 'MobileNetSSD_deploy.caffemodel')
# net.setInput(blob)
# detections = net.forward()

# --- Simulate SSD detection output ---
# Format: (1, 1, N, 7) where each row is [batch, class_id, conf, x1, y1, x2, y2]
# Coordinates are normalized (0 to 1)
img_h, img_w = img.shape[:2]

simulated_detections = np.zeros((1, 1, 6, 7), dtype=np.float32)
# Detection 0: car (class 7 in VOC)
simulated_detections[0, 0, 0] = [0, 7, 0.92, 50/img_w, 240/img_h, 200/img_w, 370/img_h]
# Detection 1: person (class 15 in VOC)
simulated_detections[0, 0, 1] = [0, 15, 0.88, 325/img_w, 200/img_h, 375/img_w, 385/img_h]
# Detection 2: dog (class 12 in VOC)
simulated_detections[0, 0, 2] = [0, 12, 0.75, 480/img_w, 290/img_h, 590/img_w, 375/img_h]
# Detection 3: bottle (class 5 in VOC)
simulated_detections[0, 0, 3] = [0, 5, 0.65, 635/img_w, 125/img_h, 665/img_w, 205/img_h]
# Detection 4: low confidence false positive
simulated_detections[0, 0, 4] = [0, 9, 0.15, 0.1, 0.1, 0.3, 0.3]
# Detection 5: another low confidence
simulated_detections[0, 0, 5] = [0, 2, 0.08, 0.5, 0.6, 0.7, 0.8]

# VOC class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# Color per class (for visualization)
np.random.seed(10)
colors = {i: tuple(int(c) for c in np.random.randint(80, 255, 3)) for i in range(21)}

# --- Filter and draw detections ---
result = img.copy()
confidence_threshold = 0.5

print(f'\nSSD Output shape: {simulated_detections.shape}')
print(f'Total raw detections: {simulated_detections.shape[2]}')
print(f'Confidence threshold: {confidence_threshold}')
print(f'\nDetections:')

kept = 0
for i in range(simulated_detections.shape[2]):
    batch_id = simulated_detections[0, 0, i, 0]
    class_id = int(simulated_detections[0, 0, i, 1])
    confidence = simulated_detections[0, 0, i, 2]
    # Normalized coordinates -> pixel coordinates
    x1 = int(simulated_detections[0, 0, i, 3] * img_w)
    y1 = int(simulated_detections[0, 0, i, 4] * img_h)
    x2 = int(simulated_detections[0, 0, i, 5] * img_w)
    y2 = int(simulated_detections[0, 0, i, 6] * img_h)

    status = 'KEPT' if confidence > confidence_threshold else 'filtered'
    class_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f'class_{class_id}'
    print(f'  [{i}] {class_name}: {confidence:.2f} ({x1},{y1})-({x2},{y2}) -> {status}')

    if confidence > confidence_threshold:
        kept += 1
        color = colors.get(class_id, (0, 255, 0))
        # Bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        # Label background
        label = f'{class_name}: {confidence:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

print(f'\nKept {kept} of {simulated_detections.shape[2]} detections')

# --- Info panel ---
info = np.zeros((140, 700, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'SSD Object Detection Pipeline:', (10, 25),
            font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'Output: (1, 1, N, 7) = [batch, classId, conf, x1, y1, x2, y2]', (20, 50),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'Coordinates are NORMALIZED (0-1) -> multiply by image size', (20, 70),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, f'Filter by confidence > {confidence_threshold}: {kept} detections kept', (20, 95),
            font, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(info, 'blob = blobFromImage(img, 1/127.5, (300,300), (127.5,127.5,127.5))', (20, 120),
            font, 0.35, (150, 150, 255), 1, cv2.LINE_AA)
cv2.putText(info, f'Classes: {", ".join(VOC_CLASSES[1:8])}...', (20, 137),
            font, 0.33, (150, 150, 255), 1, cv2.LINE_AA)

cv2.putText(result, 'SSD Object Detection (Simulated)', (10, 25),
            font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

output = np.vstack([result, info])

cv2.imshow('DNN Object Detection (SSD)', output)
```
