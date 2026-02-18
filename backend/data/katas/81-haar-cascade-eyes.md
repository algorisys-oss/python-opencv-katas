---
slug: 81-haar-cascade-eyes
title: "Haar Cascade: Eye & Smile Detection"
level: advanced
concepts: [nested cascades, ROI-based detection]
prerequisites: [80-haar-cascade-face]
---

## What Problem Are We Solving?

Once you can detect faces, the next step is to detect **features within faces** — eyes, smiles, noses, and more. Running a cascade detector on the full image is wasteful and error-prone (an eye detector will produce false positives on random textures). The solution is **nested detection**: first detect faces, then search for eyes and smiles only within each face region of interest (ROI). This is faster and far more accurate.

## The Nested Detection Pattern

The key insight is that eyes and smiles have known positions relative to a face. Instead of scanning the entire image, you:

1. Detect faces in the full image
2. For each face, extract the face ROI
3. Run the eye/smile cascade only on that ROI

```python
# In production, load cascade files:
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detect faces first
# faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# For each face, detect eyes within the face ROI
# for (x, y, w, h) in faces:
#     face_roi_gray = gray[y:y+h, x:x+w]
#     face_roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
```

## Optimizing ROI for Eye Detection

Eyes appear in the **upper half** of a face. You can narrow the ROI further to reduce false positives:

```python
# Eyes are in the upper 60% of the face
for (x, y, w, h) in faces:
    eye_region_h = int(h * 0.6)
    face_upper = gray[y:y+eye_region_h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_upper, 1.1, 5, minSize=(20, 20))
```

Similarly, smiles appear in the **lower half**:

```python
# Smiles are in the lower 50% of the face
for (x, y, w, h) in faces:
    smile_y = y + h // 2
    face_lower = gray[smile_y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(face_lower, 1.8, 20)
```

## Coordinate Mapping: ROI to Full Image

When you detect within a ROI, the coordinates are **relative to that ROI**, not the full image. To draw on the original image, you must offset them:

```python
for (x, y, w, h) in faces:
    face_roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
    for (ex, ey, ew, eh) in eyes:
        # ex, ey are relative to the ROI — add face offset
        cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
```

This offset addition is the most common mistake when working with nested cascades.

## Tuning Parameters for Eyes vs Smiles

Different features need different `detectMultiScale` parameters:

```python
# Eyes: moderate sensitivity
eyes = eye_cascade.detectMultiScale(
    face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
)

# Smiles: need higher scaleFactor and more minNeighbors
# because smile patterns are more variable
smiles = smile_cascade.detectMultiScale(
    face_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 15)
)
```

Smile detection typically requires a higher `minNeighbors` (15-25) because the smile cascade produces more false positives. The `scaleFactor` for smiles is often set higher (1.5-2.0) since smiles vary less in scale within a face ROI.

## Multiple Cascade Files

OpenCV provides several cascade variants:

| Cascade | Best For |
|---|---|
| `haarcascade_eye.xml` | General eye detection |
| `haarcascade_eye_tree_eyeglasses.xml` | Eyes with glasses |
| `haarcascade_smile.xml` | Smile/mouth detection |
| `haarcascade_frontalface_alt2.xml` | Faces (alternate, often better) |
| `haarcascade_profileface.xml` | Side-view faces |

You can load multiple cascades and run them in sequence:

```python
# Try glasses-aware eye detector first, fall back to standard
# eye_cascade_glasses = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
# eye_cascade_standard = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_eye.xml')
```

## Tips & Common Mistakes

- Always search for eyes/smiles within the face ROI, not the full image. Full-image eye detection produces many false positives.
- Remember to **offset coordinates** when drawing detections from a ROI onto the full image. This is the most common bug.
- Smile detection is notoriously unreliable with Haar cascades. High `minNeighbors` values (15-25) help reduce false positives.
- The eye cascade may detect eyebrows as eyes. Limiting the ROI to the middle band of the face (20%-60% from top) helps.
- If you detect 3+ eyes in a face, your `minNeighbors` is too low.
- Performance tip: converting the face ROI to grayscale is not needed if the full image is already grayscale — the ROI slice shares the same data.
- Cascade XML files may not be available in all environments. The API pattern works the same regardless of the specific cascade file used.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic image with face-like patterns ---
canvas = np.zeros((500, 700, 3), dtype=np.uint8)
canvas[:] = (210, 200, 190)

def draw_face_with_features(img, cx, cy, face_size):
    """Draw a synthetic face with eyes and mouth for demonstration."""
    fw = face_size
    fh = int(face_size * 1.3)
    # Face oval
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, (180, 155, 140), -1)
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, (120, 100, 80), 2)
    # Left eye
    eye_y = cy - fh // 4
    left_ex = cx - fw // 3
    right_ex = cx + fw // 3
    eye_w = max(fw // 5, 5)
    eye_h = max(fw // 7, 3)
    cv2.ellipse(img, (left_ex, eye_y), (eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (left_ex, eye_y), max(eye_h - 1, 2), (40, 30, 20), -1)
    # Right eye
    cv2.ellipse(img, (right_ex, eye_y), (eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (right_ex, eye_y), max(eye_h - 1, 2), (40, 30, 20), -1)
    # Smile
    smile_y = cy + fh // 3
    cv2.ellipse(img, (cx, smile_y), (fw // 3, fh // 6), 0, 10, 170, (60, 40, 40), 2)
    # Return bounding boxes for face, eyes, and smile
    face_box = (cx - fw, cy - fh, fw * 2, fh * 2)
    left_eye_box = (left_ex - eye_w, eye_y - eye_h, eye_w * 2, eye_h * 2)
    right_eye_box = (right_ex - eye_w, eye_y - eye_h, eye_w * 2, eye_h * 2)
    smile_box = (cx - fw // 3, smile_y - fh // 6, fw * 2 // 3, fh // 3)
    return face_box, [left_eye_box, right_eye_box], smile_box

# Draw two faces
face1, eyes1, smile1 = draw_face_with_features(canvas, 200, 230, 80)
face2, eyes2, smile2 = draw_face_with_features(canvas, 500, 250, 65)

# Simulated detections (what cascades would return)
sim_faces = [face1, face2]
sim_eyes = [eyes1, eyes2]
sim_smiles = [smile1, smile2]

# --- Demonstrate the nested detection pattern ---
result = canvas.copy()

# In production:
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 5)
# for (x, y, w, h) in faces:
#     roi_gray = gray[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
#     smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

for i, (fx, fy, fw, fh) in enumerate(sim_faces):
    # Draw face rectangle (green)
    cv2.rectangle(result, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    cv2.putText(result, f'Face {i+1}', (fx, fy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw eye rectangles (blue) - within face ROI
    for j, (ex, ey, ew, eh) in enumerate(sim_eyes[i]):
        cv2.rectangle(result, (ex, ey), (ex + ew, ey + eh), (255, 100, 0), 2)
        cv2.putText(result, 'Eye', (ex, ey - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1, cv2.LINE_AA)

    # Draw smile rectangle (magenta)
    sx, sy, sw, sh = sim_smiles[i]
    cv2.rectangle(result, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
    cv2.putText(result, 'Smile', (sx, sy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

# --- Info panel showing the pattern ---
info = np.zeros((160, 700, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'Nested Cascade Detection Pattern:', (10, 25),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, '1. Detect faces in full image', (20, 50),
            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '2. Extract face ROI: roi = gray[y:y+h, x:x+w]', (20, 73),
            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '3. Detect eyes in UPPER half of face ROI', (20, 96),
            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '4. Detect smiles in LOWER half of face ROI', (20, 119),
            font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, '5. Offset ROI coords to draw on full image', (20, 142),
            font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

# Legend
cv2.rectangle(info, (480, 10), (500, 25), (0, 255, 0), -1)
cv2.putText(info, 'Face', (505, 24), font, 0.4, (200, 200, 200), 1)
cv2.rectangle(info, (480, 33), (500, 48), (255, 100, 0), -1)
cv2.putText(info, 'Eye', (505, 47), font, 0.4, (200, 200, 200), 1)
cv2.rectangle(info, (480, 56), (500, 71), (255, 0, 255), -1)
cv2.putText(info, 'Smile', (505, 70), font, 0.4, (200, 200, 200), 1)

cv2.putText(result, 'Nested Cascade: Eye & Smile Detection', (10, 25),
            font, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

output = np.vstack([result, info])

print('Nested Cascade Detection Results:')
for i, (fx, fy, fw, fh) in enumerate(sim_faces):
    print(f'  Face {i+1}: ({fx},{fy}) {fw}x{fh}')
    for j, (ex, ey, ew, eh) in enumerate(sim_eyes[i]):
        print(f'    Eye {j+1}: ({ex},{ey}) {ew}x{eh}')
    sx, sy, sw, sh = sim_smiles[i]
    print(f'    Smile: ({sx},{sy}) {sw}x{sh}')

cv2.imshow('Haar Cascade Eye & Smile Detection', output)
```
