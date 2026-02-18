---
slug: 85-dnn-loading-models
title: "DNN: Loading Pre-trained Models"
level: advanced
concepts: [cv2.dnn.readNet, blobFromImage, forward pass]
prerequisites: [07-image-resizing]
---

## What Problem Are We Solving?

Deep neural networks (DNNs) are the state of the art for image classification, object detection, and segmentation. But training a model from scratch requires massive datasets and GPU time. Instead, you can use **pre-trained models** trained by researchers on large datasets and load them directly into OpenCV's DNN module for inference. OpenCV supports models from TensorFlow, Caffe, ONNX, Darknet (YOLO), and PyTorch (via ONNX). This kata covers the fundamental pattern: **load model, preprocess image, run forward pass, interpret output**.

## OpenCV's DNN Module Overview

The `cv2.dnn` module provides a **framework-agnostic inference engine**. It does not train models — it loads pre-trained models and runs them for prediction. This means you get consistent API regardless of the original training framework.

Key advantages:
- No dependency on TensorFlow, PyTorch, or Caffe at runtime
- CPU-optimized (with optional OpenCL/CUDA backends)
- Single consistent API for all supported frameworks

## Loading Models with cv2.dnn.readNet

The universal loader `cv2.dnn.readNet()` auto-detects the framework from file extensions:

```python
# Universal loader (auto-detects framework)
net = cv2.dnn.readNet('model_file', 'config_file')
```

Framework-specific loaders give you more explicit control:

| Function | Framework | Model File | Config File |
|---|---|---|---|
| `cv2.dnn.readNetFromTensorflow()` | TensorFlow | `.pb` | `.pbtxt` |
| `cv2.dnn.readNetFromCaffe()` | Caffe | `.caffemodel` | `.prototxt` |
| `cv2.dnn.readNetFromONNX()` | ONNX | `.onnx` | — |
| `cv2.dnn.readNetFromDarknet()` | Darknet/YOLO | `.weights` | `.cfg` |
| `cv2.dnn.readNetFromTorch()` | Torch | `.t7` / `.net` | — |

```python
# TensorFlow model
# net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'config.pbtxt')

# Caffe model
# net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# ONNX model
# net = cv2.dnn.readNetFromONNX('model.onnx')

# Darknet/YOLO model
# net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
```

## Preprocessing with blobFromImage

Neural networks expect a specific input format. `cv2.dnn.blobFromImage()` converts an OpenCV image into a 4D blob (batch, channels, height, width):

```python
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0,       # Pixel value scaling (e.g., 1/255.0)
    size=(224, 224),        # Target spatial size
    mean=(104, 117, 123),   # Mean subtraction values (BGR)
    swapRB=True,            # Swap Blue and Red channels (BGR -> RGB)
    crop=False              # Whether to crop after resize
)
```

| Parameter | Purpose |
|---|---|
| `scalefactor` | Multiplied with each pixel value. Use `1/255.0` for models expecting [0,1] range |
| `size` | The input size the model expects (e.g., 224x224 for classification, 300x300 for SSD) |
| `mean` | Per-channel mean values to subtract. Depends on how the model was trained |
| `swapRB` | Set `True` if model expects RGB but OpenCV loads BGR |
| `crop` | If `True`, crops the center after resizing to maintain aspect ratio |

The output blob has shape `(1, C, H, W)` — a single batch of C channels, H height, W width.

```python
print(blob.shape)  # (1, 3, 224, 224) for a typical classification model
```

## Common Preprocessing Recipes

Different models expect different preprocessing:

```python
# ImageNet models (ResNet, VGG, etc.)
blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=True)

# MobileNet / models expecting [0, 1] range
blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), (0, 0, 0), swapRB=True)

# SSD detection models
blob = cv2.dnn.blobFromImage(img, 1/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True)

# YOLO models
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
```

## Running the Forward Pass

After setting the input, call `forward()` to get predictions:

```python
net.setInput(blob)
output = net.forward()  # Run through the entire network

# Or get output from specific layers:
output = net.forward('detection_out')

# Or multiple output layers:
outputs = net.forward(['layer1', 'layer2'])
```

## Getting Layer Names

To see what layers and output names are available:

```python
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()  # Final output layer names
print(f'Output layers: {output_layers}')
```

## Setting the Backend and Target

For performance, you can choose the computation backend:

```python
# CPU (default)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# OpenCL acceleration (if available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# CUDA acceleration (if built with CUDA support)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

## Tips & Common Mistakes

- The `mean` values must match exactly what was used during model training. Wrong means = garbage output.
- `swapRB=True` is needed for most models because they expect RGB, but OpenCV loads images as BGR.
- `scalefactor` is a **multiplier**, not a divisor. For 1/255 normalization, pass `1/255.0` not `255`.
- Always check the model's expected input size. Using the wrong size may crash or produce incorrect results.
- `blobFromImage` does **not** modify the original image. It creates a new blob.
- If `readNet` fails, the model file is likely corrupted, the wrong format, or uses unsupported operations.
- `forward()` with no arguments returns the output of the last layer. Pass layer names for specific outputs.
- DNN module is inference-only. You cannot train or fine-tune models with it.
- In a sandboxed playground, model files are not available on disk. You can still demonstrate `blobFromImage` and the preprocessing pipeline with synthetic data.

## Starter Code

```python
import cv2
import numpy as np

# --- Demonstrate the DNN preprocessing pipeline ---
# Create a test image with recognizable content
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (200, 180, 160)

# Draw some objects
cv2.rectangle(img, (50, 50), (250, 300), (0, 0, 200), -1)       # Red box
cv2.circle(img, (400, 180), 100, (0, 200, 0), -1)                # Green circle
cv2.putText(img, 'DNN Input', (180, 380),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2, cv2.LINE_AA)

# --- Step 1: blobFromImage preprocessing ---
# ImageNet-style preprocessing
blob_imagenet = cv2.dnn.blobFromImage(
    img, scalefactor=1.0, size=(224, 224),
    mean=(104, 117, 123), swapRB=True, crop=False
)

# MobileNet-style preprocessing (0-1 range)
blob_mobilenet = cv2.dnn.blobFromImage(
    img, scalefactor=1/255.0, size=(224, 224),
    mean=(0, 0, 0), swapRB=True, crop=False
)

# YOLO-style preprocessing
blob_yolo = cv2.dnn.blobFromImage(
    img, scalefactor=1/255.0, size=(416, 416),
    mean=(0, 0, 0), swapRB=True, crop=False
)

# SSD-style preprocessing
blob_ssd = cv2.dnn.blobFromImage(
    img, scalefactor=1/127.5, size=(300, 300),
    mean=(127.5, 127.5, 127.5), swapRB=True, crop=False
)

print('Blob shapes:')
print(f'  ImageNet: {blob_imagenet.shape}')
print(f'  MobileNet: {blob_mobilenet.shape}')
print(f'  YOLO:     {blob_yolo.shape}')
print(f'  SSD:      {blob_ssd.shape}')

# --- Visualize what blobFromImage does ---
# Extract the processed image from the blob for visualization
def blob_to_display(blob, title):
    """Convert a blob back to a displayable image."""
    b = blob[0]  # Remove batch dimension -> (C, H, W)
    # Transpose to (H, W, C)
    b = b.transpose(1, 2, 0)
    # Normalize to 0-255 for display
    b = b - b.min()
    if b.max() > 0:
        b = (b / b.max() * 255).astype(np.uint8)
    else:
        b = np.zeros_like(b, dtype=np.uint8)
    # Ensure 3 channels
    if len(b.shape) == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    return b

vis_imagenet = blob_to_display(blob_imagenet, 'ImageNet')
vis_mobilenet = blob_to_display(blob_mobilenet, 'MobileNet')
vis_yolo = blob_to_display(blob_yolo, 'YOLO')
vis_ssd = blob_to_display(blob_ssd, 'SSD')

# Resize all to same height for display
display_h = 200
def resize_for_display(img, h):
    aspect = img.shape[1] / img.shape[0]
    return cv2.resize(img, (int(h * aspect), h))

vis_imagenet = resize_for_display(vis_imagenet, display_h)
vis_mobilenet = resize_for_display(vis_mobilenet, display_h)
vis_yolo = resize_for_display(vis_yolo, display_h)
vis_ssd = resize_for_display(vis_ssd, display_h)
orig_resized = resize_for_display(img, display_h)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(orig_resized, 'Original', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(vis_imagenet, 'ImageNet Blob', (5, 20), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(vis_imagenet, '224x224', (5, 40), font, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
cv2.putText(vis_mobilenet, 'MobileNet Blob', (5, 20), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(vis_mobilenet, '224x224, /255', (5, 40), font, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
cv2.putText(vis_yolo, 'YOLO Blob', (5, 20), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(vis_yolo, '416x416, /255', (5, 40), font, 0.35, (0, 200, 200), 1, cv2.LINE_AA)
cv2.putText(vis_ssd, 'SSD Blob', (5, 20), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(vis_ssd, '300x300', (5, 40), font, 0.35, (0, 200, 200), 1, cv2.LINE_AA)

# Make all same width for stacking
target_w = 200
def pad_to_width(img, w):
    if img.shape[1] >= w:
        return cv2.resize(img, (w, display_h))
    pad = np.zeros((display_h, w - img.shape[1], 3), dtype=np.uint8)
    return np.hstack([img, pad])

orig_resized = pad_to_width(orig_resized, target_w + 100)
vis_imagenet = pad_to_width(vis_imagenet, target_w)
vis_mobilenet = pad_to_width(vis_mobilenet, target_w)
vis_yolo = pad_to_width(vis_yolo, target_w)
vis_ssd = pad_to_width(vis_ssd, target_w)

# Build grid
top_row = np.hstack([orig_resized, vis_imagenet, vis_mobilenet])
# Pad top row or bottom row to match widths
total_w = top_row.shape[1]
bottom_items = np.hstack([vis_yolo, vis_ssd])
if bottom_items.shape[1] < total_w:
    pad = np.zeros((display_h, total_w - bottom_items.shape[1], 3), dtype=np.uint8)
    bottom_items = np.hstack([bottom_items, pad])
else:
    bottom_items = bottom_items[:, :total_w]

result = np.vstack([top_row, bottom_items])

# --- Info panel ---
info = np.zeros((170, total_w, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
cv2.putText(info, 'DNN Model Loading Pattern:', (10, 25),
            font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'net = cv2.dnn.readNet("model.pb", "config.pbtxt")', (20, 50),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'blob = cv2.dnn.blobFromImage(img, scale, size, mean, swapRB)', (20, 70),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'net.setInput(blob)', (20, 90),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'output = net.forward()  # or net.forward(layer_names)', (20, 110),
            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'Supported: TensorFlow(.pb) Caffe(.caffemodel) ONNX(.onnx) Darknet(.weights)', (20, 140),
            font, 0.35, (150, 150, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'Blob shape: (batch, channels, height, width) = (1, 3, H, W)', (20, 160),
            font, 0.35, (150, 150, 255), 1, cv2.LINE_AA)

output = np.vstack([result, info])

print('\nDNN Loading Pattern:')
print('  net = cv2.dnn.readNet(model_file, config_file)')
print('  blob = cv2.dnn.blobFromImage(img, scalefactor, size, mean, swapRB)')
print('  net.setInput(blob)')
print('  output = net.forward()')

cv2.imshow('DNN Loading Pre-trained Models', output)
```
