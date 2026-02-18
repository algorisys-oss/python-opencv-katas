---
slug: 80-haar-cascade-face
title: "Haar Cascade: Face Detection"
level: advanced
concepts: [cv2.CascadeClassifier, detectMultiScale, scaleFactor]
prerequisites: [07-image-resizing, 02-color-spaces]
---

## What Problem Are We Solving?

Detecting faces in images is one of the most fundamental tasks in computer vision. Before deep learning dominated the field, **Haar cascade classifiers** were the go-to method for real-time face detection. Introduced by Viola and Jones in 2001, they use simple rectangular features (called Haar-like features) trained with AdaBoost to rapidly scan an image at multiple scales and find face-like patterns. OpenCV ships with pre-trained Haar cascade XML files for faces, eyes, smiles, and more.

## How Haar Cascades Work

A Haar cascade works by sliding a detection window across the image and evaluating a series of **stages**. Each stage contains a set of weak classifiers based on Haar-like features — simple patterns that measure intensity differences between rectangular regions:

```
Edge feature:    Line feature:    Four-rectangle:
[white][black]   [w][b][w]       [w][b]
                                 [b][w]
```

The "cascade" part means the classifier is organized as a chain of stages. Each stage quickly rejects non-face regions. Only regions that pass ALL stages are detected as faces. This makes it extremely fast — most image patches are rejected in the first few stages.

## Loading a Cascade Classifier

In OpenCV, you load a pre-trained cascade from an XML file using `cv2.CascadeClassifier`:

```python
# In production, load the pre-trained XML file:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Or use OpenCV's built-in data path:
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
```

The `cv2.data.haarcascades` attribute points to the directory where OpenCV stores its pre-trained cascade files. Common cascade files include:

| File | Detects |
|---|---|
| `haarcascade_frontalface_default.xml` | Frontal faces |
| `haarcascade_frontalface_alt2.xml` | Frontal faces (alternate, often more accurate) |
| `haarcascade_eye.xml` | Eyes |
| `haarcascade_smile.xml` | Smiles |
| `haarcascade_fullbody.xml` | Full body |

## The detectMultiScale Method

Once the cascade is loaded, detection happens via `detectMultiScale()`:

```python
faces = face_cascade.detectMultiScale(
    gray,            # Input image (must be grayscale)
    scaleFactor=1.1, # How much to shrink image at each scale
    minNeighbors=5,  # Minimum number of overlapping detections to keep
    minSize=(30, 30) # Minimum face size in pixels
)
```

| Parameter | Meaning |
|---|---|
| `scaleFactor` | Controls the image pyramid. 1.1 means 10% reduction each step. Smaller = more thorough but slower |
| `minNeighbors` | Higher = fewer false positives but may miss faces. Lower = more detections but more noise |
| `minSize` | Minimum object size to detect. Skips anything smaller |

The return value `faces` is a numpy array of shape `(N, 4)` where each row is `(x, y, w, h)` — the bounding box of a detected face.

## Tuning scaleFactor and minNeighbors

These two parameters control the trade-off between detection sensitivity and false positives:

```python
# Sensitive: catches more faces but more false positives
faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

# Conservative: fewer false positives but may miss faces
faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)
```

- **scaleFactor closer to 1.0** (e.g., 1.01): More scales are checked, so detection is more thorough but much slower.
- **scaleFactor further from 1.0** (e.g., 1.3): Fewer scales, faster but may miss faces between scales.
- **minNeighbors=0**: Returns all raw detections (many overlapping boxes).
- **minNeighbors=5-7**: Good default for production use.

## Drawing Detected Faces

Once you have detections, draw rectangles around them:

```python
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

## Tips & Common Mistakes

- The input to `detectMultiScale` **must be grayscale**. Always convert with `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` first.
- If `face_cascade.empty()` returns `True`, the XML file failed to load — check the file path.
- Haar cascades detect **frontal faces** only. Profile or tilted faces require different cascade files.
- Performance tip: resize large images before detection. Faces don't need to be at full resolution to be found.
- `scaleFactor` must be strictly greater than 1.0. Values like 1.05 to 1.3 are typical.
- Haar cascades are fast but less accurate than modern DNN-based detectors. They work best on well-lit frontal faces.
- The cascade files referenced by `cv2.data.haarcascades` may not be available in all environments. In a sandboxed playground, you can demonstrate the API pattern with synthetic data.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic face-like pattern for demonstration ---
canvas = np.zeros((500, 700, 3), dtype=np.uint8)
canvas[:] = (200, 200, 200)  # Light gray background

# Draw several "face-like" oval shapes with features
def draw_synthetic_face(img, cx, cy, size):
    """Draw a simple face-like pattern with eyes and mouth."""
    # Face oval
    cv2.ellipse(img, (cx, cy), (size, int(size * 1.2)), 0, 0, 360, (180, 150, 130), -1)
    cv2.ellipse(img, (cx, cy), (size, int(size * 1.2)), 0, 0, 360, (100, 80, 60), 2)
    # Eyes
    eye_offset_x = size // 3
    eye_offset_y = size // 3
    eye_radius = max(size // 8, 3)
    cv2.circle(img, (cx - eye_offset_x, cy - eye_offset_y), eye_radius, (50, 50, 50), -1)
    cv2.circle(img, (cx + eye_offset_x, cy - eye_offset_y), eye_radius, (50, 50, 50), -1)
    # Mouth
    mouth_y = cy + size // 2
    cv2.ellipse(img, (cx, mouth_y), (size // 3, size // 6), 0, 0, 180, (50, 50, 50), 2)

# Draw three faces at different positions and sizes
draw_synthetic_face(canvas, 150, 180, 70)
draw_synthetic_face(canvas, 400, 200, 90)
draw_synthetic_face(canvas, 580, 300, 50)

# --- Demonstrate the Haar Cascade API pattern ---
# In production, load the cascade XML:
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

# Convert to grayscale (required for detectMultiScale)
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

# In production, detect faces:
# faces = face_cascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30)
# )

# --- Simulate detections for demonstration ---
# These represent what detectMultiScale would return: (x, y, w, h)
simulated_faces = np.array([
    [80, 95, 140, 170],    # Face 1
    [310, 95, 180, 210],   # Face 2
    [530, 240, 100, 120],  # Face 3
])

# --- Draw detection results ---
result = canvas.copy()
for (x, y, w, h) in simulated_faces:
    # Green rectangle around detected face
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Label with confidence area
    label = f'Face ({w}x{h})'
    cv2.putText(result, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# --- Show parameter comparison ---
# Demonstrate how scaleFactor affects number of scales
info_panel = np.zeros((140, 700, 3), dtype=np.uint8)
info_panel[:] = (40, 40, 40)
cv2.putText(info_panel, 'detectMultiScale Parameters:', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info_panel, 'scaleFactor=1.05  -> 47 scales (slow, thorough)', (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info_panel, 'scaleFactor=1.1   -> 25 scales (balanced)', (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info_panel, 'scaleFactor=1.3   -> 10 scales (fast, may miss)', (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info_panel, f'Detections: {len(simulated_faces)} faces found', (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Add label to main image
cv2.putText(result, 'Haar Cascade Face Detection (Simulated)', (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

# Combine display
output = np.vstack([result, info_panel])

print(f'Detected {len(simulated_faces)} faces')
for i, (x, y, w, h) in enumerate(simulated_faces):
    print(f'  Face {i+1}: x={x}, y={y}, w={w}, h={h}')

print('\nAPI Pattern:')
print('  cascade = cv2.CascadeClassifier(xml_path)')
print('  faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)')

cv2.imshow('Haar Cascade Face Detection', output)
```
