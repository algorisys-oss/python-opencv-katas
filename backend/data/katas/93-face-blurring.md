---
slug: 93-face-blurring
title: Face Blurring Pipeline
level: advanced
concepts: [detection, ROI blur, privacy protection]
prerequisites: [80-haar-cascade-face, 21-gaussian-blur]
---

## What Problem Are We Solving?

Privacy regulations and ethical concerns often require **blurring or anonymizing faces** in images and video. Think of Google Street View blurring pedestrian faces, or a documentary obscuring bystanders. The pipeline is: detect faces in the image, extract each face region, apply a heavy blur to make the face unrecognizable, and paste the blurred region back.

In production, face detection usually relies on Haar cascades or deep learning models. For this kata, we simulate face-like regions with synthetic oval patterns and demonstrate the complete detection-to-blur pipeline.

## Step 1: Create Synthetic Face-Like Patterns

We draw simplified face-like shapes -- skin-colored ovals with eyes and mouth -- on a scene background. This lets us test the pipeline without needing real face images or cascade files:

```python
# Draw a face-like oval
cv2.ellipse(scene, (x, y), (40, 50), 0, 0, 360, (180, 200, 220), -1)
# Add eyes
cv2.circle(scene, (x - 15, y - 10), 5, (50, 50, 50), -1)
cv2.circle(scene, (x + 15, y - 10), 5, (50, 50, 50), -1)
```

## Step 2: Detecting Face Regions

In a real application, you would use `cv2.CascadeClassifier` to detect faces:

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
```

Each detection returns `(x, y, w, h)` -- a bounding rectangle around the face. For our synthetic scene, we define these rectangles directly based on where we drew the faces.

## Step 3: Extract the ROI and Apply Heavy Blur

For each detected face rectangle, we extract the region of interest (ROI), apply a heavy Gaussian blur, and place it back:

```python
for (x, y, w, h) in faces:
    roi = image[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
    image[y:y+h, x:x+w] = blurred_roi
```

The kernel size `(99, 99)` and sigma `30` are deliberately large to make the face unrecognizable. The kernel must be odd and large enough relative to the face size.

## Step 4: Alternative Blur Methods

Gaussian blur is the most common approach, but there are alternatives:

**Pixelation** -- shrink the ROI to a tiny size and enlarge it back, creating a blocky pixel effect:

```python
small = cv2.resize(roi, (6, 6), interpolation=cv2.INTER_LINEAR)
pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
```

**Median Blur** -- preserves edge structure less than Gaussian, producing a "painted" look:

```python
blurred_roi = cv2.medianBlur(roi, 99)
```

## The Complete Pipeline

1. **Input**: Image with faces (real or synthetic)
2. **Detect Faces**: Locate bounding rectangles around faces
3. **Extract ROI**: Crop each face region
4. **Apply Blur**: Heavy Gaussian blur, pixelation, or median blur
5. **Paste Back**: Replace the face region with the blurred version
6. **Output**: Image with anonymized faces

## Tips & Common Mistakes

- The blur kernel must be large enough to make features unrecognizable. A `(99, 99)` kernel is a good starting point for faces around 100-200 pixels wide.
- Always use odd kernel sizes for Gaussian and median blur. OpenCV will throw an error with even sizes.
- Add padding around the detected face rectangle to ensure the blur covers the full face and ears. Multiply `w` and `h` by 1.1-1.3.
- For video, apply face detection on every frame or every N-th frame (with tracking in between) for performance.
- Pixelation is often preferred over Gaussian blur because it is harder to reverse-engineer (deblurring algorithms can sometimes recover Gaussian-blurred content).
- Consider the ethical implications: in some contexts, blurring is legally required; in others, selective blurring can create misleading images.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create a synthetic scene with face-like patterns
# =============================================================
scene_h, scene_w = 400, 600
scene = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)

# Background: indoor room with wall and floor
scene[0:280] = (160, 150, 130)   # Wall
scene[280:] = (100, 90, 70)      # Floor
# Add some wall texture
for i in range(0, scene_w, 80):
    cv2.line(scene, (i, 0), (i, 280), (150, 140, 120), 1)

# Function to draw a synthetic face
def draw_face(img, cx, cy, scale=1.0):
    s = scale
    # Head (skin-colored ellipse)
    cv2.ellipse(img, (cx, cy), (int(35*s), int(45*s)), 0, 0, 360, (180, 210, 230), -1)
    cv2.ellipse(img, (cx, cy), (int(35*s), int(45*s)), 0, 0, 360, (150, 180, 200), 2)
    # Hair
    cv2.ellipse(img, (cx, cy - int(25*s)), (int(35*s), int(25*s)), 0, 180, 360, (50, 40, 30), -1)
    # Eyes
    cv2.circle(img, (cx - int(13*s), cy - int(8*s)), int(5*s), (255, 255, 255), -1)
    cv2.circle(img, (cx + int(13*s), cy - int(8*s)), int(5*s), (255, 255, 255), -1)
    cv2.circle(img, (cx - int(13*s), cy - int(8*s)), int(3*s), (50, 50, 50), -1)
    cv2.circle(img, (cx + int(13*s), cy - int(8*s)), int(3*s), (50, 50, 50), -1)
    # Nose
    cv2.line(img, (cx, cy - int(3*s)), (cx - int(3*s), cy + int(5*s)), (150, 170, 190), 1)
    # Mouth
    cv2.ellipse(img, (cx, cy + int(15*s)), (int(10*s), int(5*s)), 0, 0, 180, (100, 100, 150), 2)
    # Body (simple rectangle below head)
    cv2.rectangle(img, (cx - int(30*s), cy + int(45*s)),
                  (cx + int(30*s), cy + int(100*s)), (80, 80, 150), -1)

# Draw three "people" at different positions and scales
face_data = [
    (150, 150, 1.0),   # Person 1: left
    (350, 140, 1.2),   # Person 2: center (larger)
    (520, 160, 0.9),   # Person 3: right (smaller)
]

for (cx, cy, s) in face_data:
    draw_face(scene, cx, cy, s)

# =============================================================
# Step 2: Define face bounding boxes
# (In production, you'd use CascadeClassifier or DNN detector)
# =============================================================
# Bounding boxes: (x, y, w, h) around each face
face_rects = []
for (cx, cy, s) in face_data:
    x = cx - int(40 * s)
    y = cy - int(50 * s)
    w = int(80 * s)
    h = int(100 * s)
    face_rects.append((x, y, w, h))

print(f'Detected {len(face_rects)} faces')

# Keep an original copy for comparison
original = scene.copy()

# =============================================================
# Step 3: Apply Gaussian blur to each face
# =============================================================
gaussian_result = scene.copy()
for (x, y, w, h) in face_rects:
    # Clamp to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(scene_w, x + w)
    y2 = min(scene_h, y + h)

    roi = gaussian_result[y1:y2, x1:x2]
    # Heavy Gaussian blur to anonymize
    k_size = max(99, (min(w, h) // 2) * 2 + 1)  # Ensure odd, at least 99
    blurred = cv2.GaussianBlur(roi, (k_size, k_size), 30)
    gaussian_result[y1:y2, x1:x2] = blurred

# =============================================================
# Step 4: Apply pixelation to each face (alternative method)
# =============================================================
pixel_result = scene.copy()
for (x, y, w, h) in face_rects:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(scene_w, x + w)
    y2 = min(scene_h, y + h)

    roi = pixel_result[y1:y2, x1:x2]
    # Shrink to tiny size and scale back up
    small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    pixel_result[y1:y2, x1:x2] = pixelated

# =============================================================
# Step 5: Draw detection boxes on original for visualization
# =============================================================
detected_display = original.copy()
for (x, y, w, h) in face_rects:
    cv2.rectangle(detected_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(detected_display, 'face', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# =============================================================
# Step 6: Build comparison display
# =============================================================
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(detected_display, 'Detected Faces', (10, 25), font, 0.6, (0, 255, 0), 2)
cv2.putText(gaussian_result, 'Gaussian Blur', (10, 25), font, 0.6, (0, 255, 0), 2)
cv2.putText(pixel_result, 'Pixelated', (10, 25), font, 0.6, (0, 255, 0), 2)

# Arrange: top = detected, bottom = gaussian + pixelated
top_row = detected_display
# Resize bottom images to half width to fit side by side
half_w = scene_w // 2
gauss_small = cv2.resize(gaussian_result, (half_w, scene_h // 2))
pixel_small = cv2.resize(pixel_result, (half_w, scene_h // 2))

cv2.putText(gauss_small, 'Gaussian', (5, 18), font, 0.45, (0, 255, 0), 1)
cv2.putText(pixel_small, 'Pixelated', (5, 18), font, 0.45, (0, 255, 0), 1)

bottom_row = np.hstack([gauss_small, pixel_small])

# Match widths
if top_row.shape[1] != bottom_row.shape[1]:
    bottom_row = cv2.resize(bottom_row, (top_row.shape[1], bottom_row.shape[0]))

result = np.vstack([top_row, bottom_row])

print(f'Scene size: {scene_w}x{scene_h}')
print(f'Blur kernel used: {k_size}x{k_size}')
print(f'Face regions anonymized: {len(face_rects)}')

cv2.imshow('Face Blurring Pipeline', result)
```
