---
slug: 74-background-subtraction
title: Background Subtraction (MOG2)
level: advanced
concepts: [cv2.createBackgroundSubtractorMOG2, foreground mask]
prerequisites: [73-frame-differencing]
---

## What Problem Are We Solving?

Simple frame differencing only compares two consecutive frames, so it misses slow-moving objects and produces noisy results. **Background subtraction** builds and maintains a **statistical model of the background** over time, then classifies each pixel as foreground or background. The MOG2 (Mixture of Gaussians 2) algorithm models each pixel as a mixture of Gaussian distributions, adapting to gradual lighting changes and handling multimodal backgrounds (like waving trees).

## Creating a MOG2 Background Subtractor

```python
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `history` | 500 | Number of recent frames used to build the model |
| `varThreshold` | 16 | Threshold for Mahalanobis distance — higher means less sensitive |
| `detectShadows` | True | If True, shadows are detected and marked as gray (127) |

## Applying to Each Frame

Feed frames to the subtractor using `apply()`, which returns a foreground mask:

```python
fg_mask = bg_sub.apply(frame)
```

The mask contains:
- **255** (white) for foreground pixels
- **0** (black) for background pixels
- **127** (gray) for shadow pixels (if shadow detection is on)

## Processing the Foreground Mask

The raw mask is noisy. Clean it up with morphological operations:

```python
# Remove shadows by thresholding
_, clean_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

# Remove noise with morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

# Fill holes with closing
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
```

## Getting the Background Model

You can retrieve the computed background image:

```python
background = bg_sub.getBackgroundImage()
```

This is useful for visualization and debugging — it shows what the algorithm considers "normal."

## Tuning Parameters

- **Lower `history`** (100-200): Adapts faster to changes but may absorb slow foreground objects into the background.
- **Higher `history`** (500-1000): More stable background model but slower to adapt.
- **Lower `varThreshold`** (8-12): More sensitive — detects subtle changes but more false positives.
- **Higher `varThreshold`** (20-30): Less sensitive — only strong foreground objects detected.
- **`detectShadows=False`**: Faster processing, but shadows are classified as foreground.

## MOG2 vs KNN

OpenCV also provides `cv2.createBackgroundSubtractorKNN()` as an alternative:

```python
bg_sub_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
```

KNN is sometimes better for scenes with dynamic backgrounds (water, foliage), while MOG2 is generally faster and works well for most indoor scenes.

## Tips & Common Mistakes

- The background model needs time to initialize. The first 50-100 frames will produce noisy masks as the model is still learning.
- Always clean the foreground mask with morphological operations before finding contours.
- If shadows cause problems, either set `detectShadows=False` or threshold the mask at 200 to remove shadow pixels (gray=127).
- Very slow-moving objects can be absorbed into the background model if `history` is too short.
- A moving camera makes background subtraction useless — the entire frame changes every time. This technique requires a **stationary camera**.
- The learning rate can be controlled per-frame with `bg_sub.apply(frame, learningRate=0.01)`. A rate of 0 freezes the model; -1 uses the automatic rate.

## Starter Code

```python
import cv2
import numpy as np

# --- Simulate a scene with moving objects for background subtraction ---
num_frames = 12
frame_h, frame_w = 250, 350

# Create a static background with some texture
bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
bg[:] = (70, 60, 50)
# Floor
cv2.rectangle(bg, (0, 200), (frame_w, frame_h), (50, 80, 60), -1)
# Wall features
cv2.rectangle(bg, (30, 50), (80, 150), (90, 80, 70), -1)
cv2.rectangle(bg, (270, 40), (330, 160), (85, 75, 65), -1)

# Generate frames with two moving objects
frames = []
for i in range(num_frames):
    frame = bg.copy()
    # Object 1: circle moving right
    cx1 = int(40 + i * 25)
    cy1 = 120
    cv2.circle(frame, (cx1, cy1), 20, (200, 50, 50), -1)
    # Object 2: rectangle moving left
    rx = int(300 - i * 20)
    cv2.rectangle(frame, (rx, 80), (rx + 30, 130), (50, 50, 200), -1)
    frames.append(frame)

# --- Apply MOG2 background subtraction ---
bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=True
)

fg_masks = []
cleaned_masks = []
detection_frames = []
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for i, frame in enumerate(frames):
    # Apply background subtractor
    fg_mask = bg_sub.apply(frame)
    fg_masks.append(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))

    # Clean up the mask
    _, clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    cleaned_masks.append(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))

    # Find and draw contours
    annotated = frame.copy()
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 300:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detection_frames.append(annotated)

# Get the learned background
bg_model = bg_sub.getBackgroundImage()

# --- Build composite display ---
# Show frames 4-7 (after model has started learning) and frames 8-11
font = cv2.FONT_HERSHEY_SIMPLEX
start = 4
n = 4

# Select subset of frames
sel_fg = fg_masks[start:start + n]
sel_clean = cleaned_masks[start:start + n]
sel_det = detection_frames[start:start + n]

for j, idx in enumerate(range(start, start + n)):
    cv2.putText(sel_fg[j], f'Raw Mask {idx}', (5, 15), font, 0.4, (0, 200, 255), 1)
    cv2.putText(sel_clean[j], f'Cleaned {idx}', (5, 15), font, 0.4, (0, 200, 255), 1)
    cv2.putText(sel_det[j], f'Detected {idx}', (5, 15), font, 0.4, (0, 200, 255), 1)

row1 = np.hstack(sel_fg)
row2 = np.hstack(sel_clean)
row3 = np.hstack(sel_det)

def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

# Add background model display
bg_model_resized = cv2.resize(bg_model, (frame_w, frame_h))
bg_label = bg_model_resized.copy()
cv2.putText(bg_label, 'Learned Background', (5, 15), font, 0.4, (0, 255, 0), 1)

# Pad bg_label to match row width
pad_w = row1.shape[1] - frame_w
if pad_w > 0:
    info = np.zeros((frame_h, pad_w, 3), dtype=np.uint8)
    info[:] = (30, 30, 30)
    cv2.putText(info, 'MOG2 Parameters:', (10, 25), font, 0.5, (0, 255, 0), 1)
    cv2.putText(info, 'history=500', (10, 50), font, 0.45, (200, 200, 200), 1)
    cv2.putText(info, 'varThreshold=40', (10, 75), font, 0.45, (200, 200, 200), 1)
    cv2.putText(info, 'detectShadows=True', (10, 100), font, 0.45, (200, 200, 200), 1)
    cv2.putText(info, f'Frames processed: {num_frames}', (10, 130), font, 0.45, (200, 200, 200), 1)
    bg_row = np.hstack([bg_label, info])
else:
    bg_row = bg_label

result = np.vstack([
    make_label('MOG2 Raw Foreground Masks (gray=shadow)', row1.shape[1]),
    row1,
    make_label('Cleaned Binary Masks', row2.shape[1]),
    row2,
    make_label('Foreground Detection', row3.shape[1]),
    row3,
    make_label('Background Model', bg_row.shape[1]),
    bg_row
])

print(f'MOG2 background subtraction on {num_frames} synthetic frames')

cv2.imshow('Background Subtraction (MOG2)', result)
```
