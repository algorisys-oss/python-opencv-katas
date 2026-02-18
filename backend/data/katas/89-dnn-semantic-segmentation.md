---
slug: 89-dnn-semantic-segmentation
title: "DNN: Semantic Segmentation"
level: advanced
concepts: [pixel-wise classification, argmax, color mapping]
prerequisites: [85-dnn-loading-models]
---

## What Problem Are We Solving?

While object detection draws **boxes** around objects, **semantic segmentation** classifies **every single pixel** in the image. Each pixel is assigned a class label — person, car, road, sky, building, etc. This produces a dense, pixel-level understanding of the scene, which is critical for autonomous driving, medical imaging, and scene understanding. OpenCV's DNN module can run pre-trained segmentation models and the main challenge is interpreting and visualizing the output.

## How Semantic Segmentation Works

A segmentation model takes an image and outputs a **class probability map** for every pixel:

```
Input Image (H x W x 3)
    |
Encoder-Decoder Network (e.g., FCN, DeepLab, ENet)
    |
Output: (1, num_classes, H, W) — per-pixel class probabilities
    |
argmax across classes
    |
Class Map: (H, W) — class ID per pixel
    |
Color Mapping
    |
Colored Segmentation Visualization
```

## Common Segmentation Models

| Model | Speed | Accuracy | Classes | Use Case |
|---|---|---|---|---|
| ENet | Very Fast | Moderate | 20 (Cityscapes) | Real-time, autonomous driving |
| FCN-8s | Slow | Good | 21 (VOC) | General purpose |
| DeepLabV3 | Medium | Very Good | 21 (VOC) / 150 (ADE20K) | High-quality segmentation |

## Loading a Segmentation Model

```python
# ENet (Cityscapes, 20 classes)
# net = cv2.dnn.readNet('enet-model-best.net')

# FCN-8s (Pascal VOC, 21 classes)
# net = cv2.dnn.readNetFromCaffe('fcn8s-heavy-pascal.prototxt',
#                                 'fcn8s-heavy-pascal.caffemodel')

# DeepLabV3 (TensorFlow)
# net = cv2.dnn.readNetFromTensorflow('deeplabv3_mnv2_pascal_train_aug.pb')
```

## Preprocessing for Segmentation

```python
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,
    size=(512, 512),          # Model-specific input size
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)
# net.setInput(blob)
# output = net.forward()
```

## Understanding the Output

The output shape is `(1, num_classes, H, W)`:

```python
# output.shape = (1, 21, 512, 512) for VOC with 21 classes
# output[0, c, y, x] = probability that pixel (y, x) belongs to class c

# To get the class ID per pixel, use argmax:
class_map = np.argmax(output[0], axis=0)  # Shape: (H, W)
# class_map[y, x] = class index with highest probability at pixel (y, x)
```

The `argmax` operation across the class dimension gives you a single integer per pixel representing the most likely class.

## Color Mapping for Visualization

Each class needs a distinct color for visualization:

```python
# Pascal VOC color palette (21 classes)
VOC_COLORS = np.array([
    [0, 0, 0],         # 0: background
    [128, 0, 0],       # 1: aeroplane
    [0, 128, 0],       # 2: bicycle
    [128, 128, 0],     # 3: bird
    [0, 0, 128],       # 4: boat
    [128, 0, 128],     # 5: bottle
    [0, 128, 128],     # 6: bus
    [128, 128, 128],   # 7: car
    [64, 0, 0],        # 8: cat
    [192, 0, 0],       # 9: chair
    [64, 128, 0],      # 10: cow
    [192, 128, 0],     # 11: dining table
    [64, 0, 128],      # 12: dog
    [192, 0, 128],     # 13: horse
    [64, 128, 128],    # 14: motorbike
    [192, 128, 128],   # 15: person
    [0, 64, 0],        # 16: potted plant
    [128, 64, 0],      # 17: sheep
    [0, 192, 0],       # 18: sofa
    [128, 192, 0],     # 19: train
    [0, 64, 128],      # 20: tv/monitor
], dtype=np.uint8)

def colorize_segmentation(class_map, colors):
    """Convert class map to colored image."""
    h, w = class_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        mask = (class_map == class_id)
        colored[mask] = colors[class_id]
    return colored

# Faster vectorized version:
def colorize_fast(class_map, colors):
    """Fast color mapping using array indexing."""
    return colors[class_map]
```

## Blending Segmentation with Original Image

To see the segmentation overlaid on the original image:

```python
# Resize segmentation to match original image
seg_resized = cv2.resize(colored_seg, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)

# Blend original and segmentation
alpha = 0.5
overlay = cv2.addWeighted(original_img, 1 - alpha, seg_resized, alpha, 0)
```

Use `INTER_NEAREST` interpolation when resizing class maps to avoid creating invalid intermediate class values.

## Extracting Individual Class Masks

You can extract binary masks for specific classes:

```python
# Get mask for 'person' class (class 15 in VOC)
person_mask = (class_map == 15).astype(np.uint8) * 255

# Apply mask to original image
person_only = cv2.bitwise_and(img, img, mask=person_mask)
```

## Cityscapes Classes (for ENet)

```python
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]
```

## Tips & Common Mistakes

- The output resolution may be smaller than the input. Always resize the class map back to the original image size for overlay.
- Use `cv2.INTER_NEAREST` when resizing class maps. Bilinear or bicubic interpolation creates fractional values that produce invalid class IDs.
- `argmax` on axis 0 of `output[0]` gives per-pixel class IDs. Make sure you use axis 0 (the class axis), not axis 1 or 2.
- Segmentation models are memory-intensive. Large input sizes can consume several GB of RAM.
- The color palette must match the number of classes in the model. Using a wrong palette produces misleading colors.
- Some models output logits (before softmax), others output probabilities. For argmax, it does not matter — argmax is the same either way.
- Semantic segmentation assigns one class per pixel. It does not distinguish between individual instances (two people = same class). For that, use **instance segmentation** (e.g., Mask R-CNN).
- In a sandboxed playground, you can simulate the segmentation output and demonstrate the argmax + color mapping pipeline.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic scene for segmentation demonstration ---
img = np.zeros((400, 600, 3), dtype=np.uint8)

# Sky (class: 10 = sky-like)
img[0:150] = (200, 170, 140)

# Building (class: 2)
cv2.rectangle(img, (350, 50), (500, 250), (160, 160, 170), -1)
cv2.rectangle(img, (370, 70), (395, 110), (140, 180, 200), -1)   # Window
cv2.rectangle(img, (420, 70), (445, 110), (140, 180, 200), -1)   # Window
cv2.rectangle(img, (370, 130), (395, 170), (140, 180, 200), -1)  # Window
cv2.rectangle(img, (420, 130), (445, 170), (140, 180, 200), -1)  # Window

# Road (class: 0)
cv2.rectangle(img, (0, 250), (600, 400), (100, 100, 100), -1)
# Lane markings
for x in range(50, 600, 80):
    cv2.rectangle(img, (x, 320), (x + 40, 328), (220, 220, 220), -1)

# Car (class: 7)
cv2.rectangle(img, (80, 270), (200, 340), (200, 50, 50), -1)
cv2.rectangle(img, (105, 250), (175, 270), (200, 50, 50), -1)
cv2.circle(img, (105, 345), 14, (40, 40, 40), -1)
cv2.circle(img, (175, 345), 14, (40, 40, 40), -1)

# Person (class: 15)
cv2.circle(img, (300, 230), 15, (200, 170, 150), -1)
cv2.rectangle(img, (288, 245), (312, 320), (50, 50, 140), -1)
cv2.rectangle(img, (290, 320), (300, 360), (60, 60, 60), -1)
cv2.rectangle(img, (302, 320), (312, 360), (60, 60, 60), -1)

# Tree / vegetation (class: 8)
cv2.circle(img, (550, 180), 50, (40, 130, 40), -1)
cv2.circle(img, (530, 160), 40, (50, 140, 50), -1)
cv2.circle(img, (570, 170), 35, (45, 135, 45), -1)
cv2.rectangle(img, (545, 220), (555, 280), (80, 60, 40), -1)

# --- Simulate segmentation model output ---
# In production:
# net = cv2.dnn.readNet('enet-model-best.net')
# blob = cv2.dnn.blobFromImage(img, 1/255.0, (512, 512), (0,0,0), swapRB=True)
# net.setInput(blob)
# output = net.forward()  # shape: (1, num_classes, H, W)

# Create a simulated class map (what argmax of model output would give)
h, w = img.shape[:2]
class_map = np.zeros((h, w), dtype=np.int32)

# Assign classes to regions
class_map[0:150, :] = 10          # Sky
class_map[250:400, :] = 0         # Road
class_map[150:250, :] = 10        # More sky (default)

# Building region
class_map[50:250, 350:500] = 2    # Building

# Car region
class_map[250:345, 80:200] = 7    # Car

# Person region
class_map[215:360, 285:315] = 15  # Person

# Vegetation region
cv2_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(cv2_mask, (550, 180), 50, 1, -1)
cv2.circle(cv2_mask, (530, 160), 40, 1, -1)
cv2.circle(cv2_mask, (570, 170), 35, 1, -1)
cv2.rectangle(cv2_mask, (545, 220), (555, 280), 1, -1)
class_map[cv2_mask == 1] = 8      # Vegetation

print(f'Class map shape: {class_map.shape}')
print(f'Unique classes: {np.unique(class_map)}')

# --- Color mapping ---
# VOC-style color palette (21 classes)
VOC_COLORS = np.array([
    [0, 0, 0],         # 0: background / road
    [128, 0, 0],       # 1: aeroplane
    [0, 128, 0],       # 2: bicycle -> building here
    [128, 128, 0],     # 3: bird
    [0, 0, 128],       # 4: boat
    [128, 0, 128],     # 5: bottle
    [0, 128, 128],     # 6: bus
    [128, 128, 128],   # 7: car
    [64, 128, 0],      # 8: cat -> vegetation here
    [192, 0, 0],       # 9: chair
    [135, 206, 235],   # 10: cow -> sky here
    [192, 128, 0],     # 11: dining table
    [64, 0, 128],      # 12: dog
    [192, 0, 128],     # 13: horse
    [64, 128, 128],    # 14: motorbike
    [192, 128, 128],   # 15: person
    [0, 64, 0],        # 16: potted plant
    [128, 64, 0],      # 17: sheep
    [0, 192, 0],       # 18: sofa
    [128, 192, 0],     # 19: train
    [0, 64, 128],      # 20: tv/monitor
], dtype=np.uint8)

# Custom labels for our scene
SCENE_LABELS = {
    0: 'road', 2: 'building', 7: 'car', 8: 'vegetation',
    10: 'sky', 15: 'person'
}

# Fast color mapping using array indexing
colored_seg = VOC_COLORS[class_map]

# --- Blend with original image ---
alpha = 0.5
overlay = cv2.addWeighted(img, 1 - alpha, colored_seg, alpha, 0)

# --- Extract individual class masks ---
person_mask = (class_map == 15).astype(np.uint8) * 255
car_mask = (class_map == 7).astype(np.uint8) * 255

# --- Build visualization ---
# Row 1: Original | Segmentation Color Map
# Row 2: Blended Overlay | Class Legend
seg_display = colored_seg.copy()
cv2.putText(seg_display, 'Segmentation Map', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'Original Image', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(overlay, 'Blended Overlay (alpha=0.5)', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# Legend panel
legend = np.zeros((400, 600, 3), dtype=np.uint8)
legend[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(legend, 'Semantic Segmentation Pipeline:', (10, 30),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(legend, 'output = net.forward()  # (1, C, H, W)', (20, 60),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(legend, 'class_map = np.argmax(output[0], axis=0)  # (H, W)', (20, 85),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(legend, 'colored = color_palette[class_map]  # (H, W, 3)', (20, 110),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(legend, 'overlay = cv2.addWeighted(img, 0.5, colored, 0.5, 0)', (20, 135),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

# Draw color swatches
cv2.putText(legend, 'Class Legend:', (20, 175),
            font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
y_offset = 200
for class_id, label in sorted(SCENE_LABELS.items()):
    color = tuple(int(c) for c in VOC_COLORS[class_id])
    cv2.rectangle(legend, (30, y_offset - 12), (55, y_offset + 5), color, -1)
    cv2.rectangle(legend, (30, y_offset - 12), (55, y_offset + 5), (200, 200, 200), 1)
    pixel_count = np.sum(class_map == class_id)
    pct = pixel_count / class_map.size * 100
    cv2.putText(legend, f'{label} (class {class_id}): {pct:.1f}% of pixels',
                (65, y_offset), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    y_offset += 28

cv2.putText(legend, 'Resize tip: use cv2.INTER_NEAREST for class maps!', (20, 380),
            font, 0.4, (150, 150, 255), 1, cv2.LINE_AA)

top_row = np.hstack([img, seg_display])
bottom_row = np.hstack([overlay, legend])
result = np.vstack([top_row, bottom_row])

# Print class statistics
print('\nPixel class distribution:')
for class_id, label in sorted(SCENE_LABELS.items()):
    count = np.sum(class_map == class_id)
    pct = count / class_map.size * 100
    print(f'  {label:12s} (class {class_id:2d}): {count:6d} pixels ({pct:.1f}%)')

cv2.imshow('DNN Semantic Segmentation', result)
```
