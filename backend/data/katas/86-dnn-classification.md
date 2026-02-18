---
slug: 86-dnn-classification
title: "DNN: Image Classification"
level: advanced
concepts: [classification pipeline, softmax, class labels]
prerequisites: [85-dnn-loading-models]
---

## What Problem Are We Solving?

Image classification answers the question "What is in this image?" by assigning one or more labels from a predefined set of categories. Using OpenCV's DNN module with a pre-trained classification model (like ResNet, MobileNet, or GoogLeNet), you can classify images into one of 1000 ImageNet categories. This kata covers the complete classification pipeline: loading a model, preprocessing the image, running inference, and interpreting the output to get human-readable predictions.

## The Classification Pipeline

The end-to-end flow is:

```
Load Model -> Preprocess Image -> Forward Pass -> Softmax -> Top-K Labels
```

```python
# In production:
# 1. Load model
# net = cv2.dnn.readNetFromONNX('resnet50.onnx')

# 2. Preprocess
# blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), (0, 0, 0), swapRB=True)

# 3. Forward pass
# net.setInput(blob)
# output = net.forward()  # Shape: (1, 1000) for ImageNet

# 4. Interpret
# class_id = np.argmax(output)
# confidence = output[0][class_id]
```

## Understanding the Output

A classification model outputs a vector of **logits** (raw scores) or **probabilities** for each class. For ImageNet models, this is a 1D array of 1000 values:

```python
# output shape: (1, 1000) — one score per ImageNet class
output = net.forward()
print(output.shape)  # (1, 1000)
```

To convert logits to probabilities, apply **softmax**:

```python
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

probabilities = softmax(output[0])
```

Some models already include a softmax layer, in which case the output is already probabilities (all values between 0 and 1, summing to 1).

## Getting Top-K Predictions

Rather than just the single best prediction, top-K gives you the most likely classes:

```python
def get_top_k(output, k=5):
    """Get top-K class indices and their probabilities."""
    probs = softmax(output[0])
    top_indices = np.argsort(probs)[::-1][:k]
    return [(idx, probs[idx]) for idx in top_indices]

# Example usage:
# top5 = get_top_k(output, k=5)
# for class_id, confidence in top5:
#     print(f'{class_labels[class_id]}: {confidence:.4f}')
```

## Loading Class Labels

ImageNet models output class indices. You need a labels file to map indices to names:

```python
# Load class labels from a text file (one label per line)
# with open('imagenet_classes.txt', 'r') as f:
#     class_labels = [line.strip() for line in f.readlines()]

# Then map predictions:
# class_name = class_labels[class_id]
```

Common label files:
- `synset_words.txt` — ImageNet synset IDs and descriptions
- `imagenet_classes.txt` — Simple class names, one per line

## Common Classification Models

| Model | Input Size | Parameters | Speed | Accuracy |
|---|---|---|---|---|
| MobileNetV2 | 224x224 | 3.4M | Very Fast | Good |
| GoogLeNet | 224x224 | 6.8M | Fast | Good |
| ResNet-50 | 224x224 | 25M | Medium | Very Good |
| VGG-16 | 224x224 | 138M | Slow | Very Good |

Each model has different preprocessing requirements:

```python
# GoogLeNet (Caffe)
# blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104, 117, 123), swapRB=False)

# MobileNetV2 (TensorFlow)
# blob = cv2.dnn.blobFromImage(img, 1/127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)

# ResNet-50 (ONNX)
# blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), (0.485*255, 0.456*255, 0.406*255), swapRB=True)
```

## Measuring Inference Time

Use OpenCV's built-in timing for the DNN module:

```python
# net.setInput(blob)
# output = net.forward()
# t, _ = net.getPerfProfile()
# inference_time = t * 1000.0 / cv2.getTickFrequency()
# print(f'Inference time: {inference_time:.1f} ms')
```

## Tips & Common Mistakes

- Always match the preprocessing (mean, scale, size) to the specific model. Mismatched preprocessing is the most common cause of wrong predictions.
- Softmax converts logits to probabilities. If the model already outputs probabilities, applying softmax again will produce incorrect results. Check if the max output is >> 1.0 (logits) or all values are 0-1 (already softmaxed).
- Top-1 accuracy is unreliable for images that could belong to multiple categories. Always check top-5 predictions.
- ImageNet class indices are fixed. The same index always maps to the same class across all ImageNet models.
- Classification models output a **single label per image**. They do not tell you where an object is (use detection models for that).
- The label file must have exactly the same number of lines as the model's output dimension (1000 for ImageNet).
- In a sandboxed playground, you can simulate the full pipeline with synthetic model output to understand the data flow.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a test image ---
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (220, 210, 200)

# Draw a simple "cat-like" shape for demonstration
cv2.ellipse(img, (300, 200), (120, 100), 0, 0, 360, (150, 130, 100), -1)  # Body
cv2.ellipse(img, (300, 130), (60, 50), 0, 0, 360, (150, 130, 100), -1)    # Head
# Ears
pts_ear_l = np.array([[255, 90], [240, 130], [270, 120]], dtype=np.int32)
pts_ear_r = np.array([[345, 90], [330, 120], [360, 130]], dtype=np.int32)
cv2.fillPoly(img, [pts_ear_l, pts_ear_r], (150, 130, 100))
# Eyes
cv2.circle(img, (280, 125), 8, (50, 180, 50), -1)
cv2.circle(img, (320, 125), 8, (50, 180, 50), -1)
# Nose
cv2.circle(img, (300, 140), 4, (100, 80, 80), -1)
# Whiskers
cv2.line(img, (260, 140), (220, 135), (80, 60, 60), 1)
cv2.line(img, (260, 145), (220, 150), (80, 60, 60), 1)
cv2.line(img, (340, 140), (380, 135), (80, 60, 60), 1)
cv2.line(img, (340, 145), (380, 150), (80, 60, 60), 1)

# --- Step 1: Preprocess with blobFromImage ---
blob = cv2.dnn.blobFromImage(
    img, scalefactor=1/255.0, size=(224, 224),
    mean=(0, 0, 0), swapRB=True, crop=False
)
print(f'Input image shape: {img.shape}')
print(f'Blob shape: {blob.shape}')

# --- Step 2: In production, run forward pass ---
# net = cv2.dnn.readNetFromONNX('resnet50.onnx')
# net.setInput(blob)
# output = net.forward()  # shape: (1, 1000)

# --- Step 3: Simulate classification output ---
# Create a synthetic output vector (1000 classes, ImageNet-like)
np.random.seed(42)
simulated_logits = np.random.randn(1, 1000).astype(np.float32) * 0.5

# Make certain classes have high scores to simulate a real prediction
simulated_logits[0, 281] = 5.2   # 'tabby cat'
simulated_logits[0, 282] = 3.8   # 'tiger cat'
simulated_logits[0, 285] = 2.9   # 'Egyptian cat'
simulated_logits[0, 283] = 2.1   # 'Persian cat'
simulated_logits[0, 287] = 1.5   # 'lynx'

# --- Step 4: Apply softmax to get probabilities ---
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

probabilities = softmax(simulated_logits[0])

# --- Step 5: Get top-K predictions ---
k = 5
top_indices = np.argsort(probabilities)[::-1][:k]

# Simulated ImageNet labels for relevant classes
imagenet_labels = {
    281: 'tabby cat', 282: 'tiger cat', 283: 'Persian cat',
    285: 'Egyptian cat', 287: 'lynx', 288: 'leopard',
}

print(f'\nTop-{k} predictions (simulated):')
top_predictions = []
for idx in top_indices:
    label = imagenet_labels.get(idx, f'class_{idx}')
    conf = probabilities[idx]
    top_predictions.append((label, conf))
    print(f'  {label}: {conf:.4f} ({conf*100:.1f}%)')

# --- Visualize results ---
result = img.copy()
cv2.putText(result, 'Input Image', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2, cv2.LINE_AA)

# Build prediction display panel
pred_panel = np.zeros((400, 400, 3), dtype=np.uint8)
pred_panel[:] = (50, 50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(pred_panel, f'Top-{k} Classification Results', (10, 30),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(pred_panel, '(Simulated Output)', (10, 52),
            font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

bar_start_x = 180
bar_max_w = 200
for i, (label, conf) in enumerate(top_predictions):
    y_pos = 90 + i * 55
    # Label and confidence text
    cv2.putText(pred_panel, f'{label}', (10, y_pos),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(pred_panel, f'{conf*100:.1f}%', (10, y_pos + 20),
                font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    # Confidence bar
    bar_w = int(conf * bar_max_w / top_predictions[0][1])  # Relative to max
    color = (0, int(255 * conf / top_predictions[0][1]), 100)
    cv2.rectangle(pred_panel, (bar_start_x, y_pos - 12),
                  (bar_start_x + bar_w, y_pos + 5), color, -1)
    cv2.rectangle(pred_panel, (bar_start_x, y_pos - 12),
                  (bar_start_x + bar_max_w, y_pos + 5), (100, 100, 100), 1)

# Pipeline summary at bottom
cv2.putText(pred_panel, 'Pipeline: Load -> Blob -> Forward -> Softmax -> Top-K', (10, 370),
            font, 0.35, (150, 150, 255), 1, cv2.LINE_AA)
cv2.putText(pred_panel, f'Output shape: (1, 1000) -> {probabilities.sum():.4f} sum', (10, 390),
            font, 0.35, (150, 150, 255), 1, cv2.LINE_AA)

# Combine image and predictions side by side
# Resize to same height
pred_panel_resized = cv2.resize(pred_panel, (400, 400))
display = np.hstack([result, pred_panel_resized])

print(f'\nSoftmax properties:')
print(f'  Sum of all probabilities: {probabilities.sum():.6f}')
print(f'  Min probability: {probabilities.min():.8f}')
print(f'  Max probability: {probabilities.max():.4f}')

cv2.imshow('DNN Image Classification', display)
```
