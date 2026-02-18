---
slug: 84-text-detection-east
title: Text Detection with EAST
level: advanced
concepts: [cv2.dnn, EAST model, text bounding boxes, NMS]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

Detecting text in natural scene images (signs, license plates, product labels) is fundamentally different from OCR on clean documents. Text in the wild can be at any angle, scale, or lighting condition. The **EAST (Efficient and Accurate Scene Text) detector** is a deep learning model that produces word-level or line-level bounding boxes for text regions. OpenCV's DNN module can load and run the EAST model to detect text locations in images.

## How EAST Works

EAST is a fully convolutional neural network that directly predicts text regions without complex multi-step pipelines:

1. **Input**: An image resized to a multiple of 32 (e.g., 320x320, 640x640)
2. **Output**: Two tensors:
   - **Scores** (confidence map): Per-pixel probability that a pixel belongs to text
   - **Geometry**: Per-pixel bounding box parameters (either rotated rectangles or quadrilaterals)
3. **Post-processing**: Non-Maximum Suppression (NMS) to merge overlapping detections

```
Input Image -> CNN Feature Extraction -> Score Map + Geometry Map -> NMS -> Text Boxes
```

## Loading the EAST Model

The EAST model is a `.pb` (TensorFlow frozen graph) file loaded via OpenCV's DNN module:

```python
# In production, load the EAST model:
# net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# The model requires input dimensions that are multiples of 32
input_width = 320
input_height = 320
```

The model file `frozen_east_text_detection.pb` must be downloaded separately (it is not bundled with OpenCV).

## Creating the Input Blob

EAST requires specific preprocessing:

```python
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0,        # No scaling
    size=(320, 320),         # Must be multiple of 32
    mean=(123.68, 116.78, 103.94),  # ImageNet mean subtraction
    swapRB=True,             # BGR to RGB
    crop=False               # Resize without cropping
)
net.setInput(blob)
```

## Forward Pass and Output Layers

EAST has two output layers that you request by name:

```python
output_layers = [
    'feature_fusion/Conv_7/Sigmoid',  # Confidence scores
    'feature_fusion/concat_3'          # Geometry (bounding box parameters)
]
# scores, geometry = net.forward(output_layers)
```

The outputs are:
- **scores**: Shape `(1, 1, H/4, W/4)` — confidence that each 4x4 region contains text
- **geometry**: Shape `(1, 5, H/4, W/4)` — 5 values per pixel: distances to top, right, bottom, left edges of the bounding box, plus the rotation angle

## Interpreting the Geometry Output

Each pixel in the geometry output encodes a rotated bounding box:

```python
# geometry shape: (1, 5, H, W)
# Channel 0: distance to top edge
# Channel 1: distance to right edge
# Channel 2: distance to bottom edge
# Channel 3: distance to left edge
# Channel 4: rotation angle

def decode_predictions(scores, geometry, min_confidence=0.5):
    (num_rows, num_cols) = scores.shape[2:4]
    boxes = []
    confidences = []

    for y in range(num_rows):
        scores_row = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]  # top
        x_data1 = geometry[0, 1, y]  # right
        x_data2 = geometry[0, 2, y]  # bottom
        x_data3 = geometry[0, 3, y]  # left
        angles = geometry[0, 4, y]   # angle

        for x in range(num_cols):
            if scores_row[x] < min_confidence:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + cos_a * x_data1[x] + sin_a * x_data2[x])
            end_y = int(offset_y - sin_a * x_data1[x] + cos_a * x_data2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_row[x]))

    return boxes, confidences
```

## Applying Non-Maximum Suppression

Multiple overlapping detections are common. Use NMS to keep only the best:

```python
# Using OpenCV's built-in NMS
indices = cv2.dnn.NMSBoxes(
    bboxes=[(x, y, w - x, h - y) for (x, y, w, h) in boxes],
    scores=confidences,
    score_threshold=0.5,
    nms_threshold=0.4
)

# Draw surviving boxes
for i in indices:
    (start_x, start_y, end_x, end_y) = boxes[i]
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
```

## Scaling Boxes Back to Original Image

Since the input is resized to 320x320 (or similar), you must scale the detected boxes back:

```python
orig_h, orig_w = original_img.shape[:2]
ratio_w = orig_w / 320.0
ratio_h = orig_h / 320.0

for (start_x, start_y, end_x, end_y) in final_boxes:
    start_x = int(start_x * ratio_w)
    start_y = int(start_y * ratio_h)
    end_x = int(end_x * ratio_w)
    end_y = int(end_y * ratio_h)
    cv2.rectangle(original_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
```

## Tips & Common Mistakes

- Input dimensions **must be multiples of 32**. Common choices: 320x320, 640x640, 1280x1280.
- Larger input = better detection of small text but much slower processing.
- The mean subtraction values `(123.68, 116.78, 103.94)` are the ImageNet mean — they must be exact.
- EAST detects text **location only**. It does not perform OCR (text recognition). Use Tesseract or another OCR engine on the detected regions.
- The confidence threshold (0.5) and NMS threshold (0.4) are good defaults but may need tuning.
- The geometry output uses a specific coordinate system. The `decode_predictions` function handles the trigonometry.
- The EAST model file (`frozen_east_text_detection.pb`) is approximately 95 MB and must be downloaded separately.
- In a sandboxed playground, you can demonstrate the blob creation and output parsing pipeline with synthetic data.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic image with text-like regions ---
canvas = np.ones((500, 700, 3), dtype=np.uint8) * 230

# Draw text regions to simulate what EAST would detect
cv2.putText(canvas, 'WELCOME TO', (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3, cv2.LINE_AA)
cv2.putText(canvas, 'OPENCV TEXT', (50, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3, cv2.LINE_AA)
cv2.putText(canvas, 'DETECTION', (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
cv2.putText(canvas, 'EAST Model', (400, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2, cv2.LINE_AA)
cv2.putText(canvas, 'Scene Text', (50, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
cv2.putText(canvas, 'EXIT 42', (450, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 150), 2, cv2.LINE_AA)

# --- Demonstrate the EAST pipeline with synthetic data ---

# Step 1: Prepare input blob (this works without the model file)
input_w, input_h = 320, 320
blob = cv2.dnn.blobFromImage(
    canvas,
    scalefactor=1.0,
    size=(input_w, input_h),
    mean=(123.68, 116.78, 103.94),
    swapRB=True,
    crop=False
)
print(f'Input blob shape: {blob.shape}')  # (1, 3, 320, 320)

# Step 2: In production, run forward pass:
# net = cv2.dnn.readNet('frozen_east_text_detection.pb')
# net.setInput(blob)
# output_layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
# scores, geometry = net.forward(output_layers)
# print(f'Scores shape: {scores.shape}')      # (1, 1, 80, 80)
# print(f'Geometry shape: {geometry.shape}')   # (1, 5, 80, 80)

# Step 3: Simulate EAST output for demonstration
# Create synthetic score and geometry maps
score_h, score_w = input_h // 4, input_w // 4  # 80x80
sim_scores = np.zeros((1, 1, score_h, score_w), dtype=np.float32)
sim_geometry = np.zeros((1, 5, score_h, score_w), dtype=np.float32)

# Simulated text detections (what EAST would find)
# Scale factors for mapping to original image
ratio_w = canvas.shape[1] / input_w
ratio_h = canvas.shape[0] / input_h

sim_detections = [
    (40, 45, 380, 95, 0.92, 'WELCOME TO'),
    (40, 100, 400, 155, 0.88, 'OPENCV TEXT'),
    (85, 165, 340, 210, 0.85, 'DETECTION'),
    (380, 315, 600, 360, 0.78, 'EAST Model'),
    (35, 370, 230, 410, 0.72, 'Scene Text'),
    (430, 68, 640, 110, 0.81, 'EXIT 42'),
]

# Step 4: Apply NMS on simulated boxes
boxes_for_nms = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _, _) in sim_detections]
scores_for_nms = [conf for (_, _, _, _, conf, _) in sim_detections]

indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms,
                            score_threshold=0.5, nms_threshold=0.4)
print(f'Detections before NMS: {len(sim_detections)}')
print(f'Detections after NMS: {len(indices)}')

# Step 5: Draw results
result = canvas.copy()
for idx in indices:
    x1, y1, x2, y2, conf, text = sim_detections[idx]
    # Draw bounding box
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw confidence
    cv2.putText(result, f'{conf:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)

# --- Info panel ---
info = np.zeros((190, 700, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'EAST Text Detection Pipeline:', (10, 25),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, '1. blob = cv2.dnn.blobFromImage(img, 1.0, (320,320), mean)', (20, 50),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '2. net.setInput(blob)', (20, 70),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '3. scores, geometry = net.forward(output_layers)', (20, 90),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '4. Decode: scores -> confidence, geometry -> bounding boxes', (20, 110),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '5. NMS: cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)', (20, 130),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '6. Scale boxes back to original image size', (20, 150),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, f'Output: scores (1,1,80,80) + geometry (1,5,80,80) for 320x320 input', (20, 178),
            font, 0.38, (150, 150, 255), 1, cv2.LINE_AA)

cv2.putText(result, 'EAST Text Detection (Simulated)', (10, 475),
            font, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

output = np.vstack([result, info])

print('\nEAST Pipeline:')
print(f'  Input blob: {blob.shape}')
print(f'  Score map would be: (1, 1, {score_h}, {score_w})')
print(f'  Geometry map would be: (1, 5, {score_h}, {score_w})')
print(f'  Output layers: feature_fusion/Conv_7/Sigmoid, feature_fusion/concat_3')

cv2.imshow('EAST Text Detection', output)
```
