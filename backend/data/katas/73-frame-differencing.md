---
slug: 73-frame-differencing
title: Frame Differencing
level: advanced
concepts: [frame delta, cv2.absdiff, motion detection]
prerequisites: [70-reading-video-files, 26-simple-thresholding]
---

## What Problem Are We Solving?

If the camera is stationary, the background stays mostly the same between frames. Anything that **moves** creates pixel differences between consecutive frames. **Frame differencing** computes these differences to detect motion — it's one of the simplest and fastest motion detection techniques, widely used in surveillance and activity monitoring.

## Computing the Frame Difference

The core operation is `cv2.absdiff()`, which computes the absolute difference between two images pixel by pixel:

```python
diff = cv2.absdiff(frame1, frame2)
```

For grayscale frames, each pixel in `diff` tells you how much that pixel changed between the two frames. A value of 0 means no change; higher values mean more change.

## The Basic Pipeline

1. **Convert to grayscale** — reduces noise and simplifies processing
2. **Blur slightly** — smooths out sensor noise
3. **Compute absolute difference** between consecutive frames
4. **Threshold** the difference to create a binary motion mask
5. **Optionally dilate** the mask to fill gaps

```python
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

diff = cv2.absdiff(gray1, gray2)
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
thresh = cv2.dilate(thresh, None, iterations=2)
```

## Two-Frame vs Three-Frame Differencing

**Two-frame differencing** compares the current frame to the previous one. It detects motion edges (leading and trailing edges of a moving object) but often shows a "ghost" double image.

**Three-frame differencing** uses the AND of two consecutive differences to keep only regions where motion is present in both:

```python
diff1 = cv2.absdiff(frame1, frame2)
diff2 = cv2.absdiff(frame2, frame3)
motion = cv2.bitwise_and(diff1, diff2)
```

This produces cleaner results because it eliminates the ghost artifacts.

## Finding Motion Regions

Once you have a binary motion mask, use `cv2.findContours()` to locate moving objects:

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 500:  # Ignore tiny noise
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

## Choosing the Threshold Value

The threshold value (e.g., 25) controls sensitivity:
- **Lower threshold** (10-20): Detects subtle motion but picks up more noise
- **Higher threshold** (30-50): Only detects significant motion, ignores small changes

## Tips & Common Mistakes

- Always convert to grayscale before differencing. Color channels add noise and make thresholding harder.
- Apply Gaussian blur before differencing to reduce sensor noise that causes false positives.
- The threshold value is scene-dependent. Indoor scenes with stable lighting can use lower values; outdoor scenes with wind and lighting changes need higher values.
- Dilate the binary mask to connect nearby motion regions into solid blobs.
- Filter contours by area to ignore tiny noise blobs. A minimum area of 500-1000 pixels works well for most cases.
- Frame differencing only detects **changes** — a moving object that stops becomes invisible after one frame.
- Lighting changes (clouds passing, lights turning on) will trigger false motion across the entire frame.

## Starter Code

```python
import cv2
import numpy as np

# --- Create synthetic frames with a moving object ---
num_frames = 6
frame_h, frame_w = 250, 350

# Create a static background
bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
bg[:] = (60, 50, 40)
# Add some static "furniture"
cv2.rectangle(bg, (20, 180), (100, 250), (80, 100, 70), -1)
cv2.rectangle(bg, (250, 160), (340, 250), (70, 80, 90), -1)

# Generate frames with a moving circle
frames = []
for i in range(num_frames):
    frame = bg.copy()
    # Moving circle (simulates a person/object)
    cx = int(60 + i * 50)
    cy = int(100 + 20 * np.sin(i * 0.8))
    cv2.circle(frame, (cx, cy), 30, (200, 180, 50), -1)
    cv2.putText(frame, f'F{i}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    frames.append(frame)

# --- Compute frame differences ---
diffs = []
thresh_imgs = []
motion_frames = []

for i in range(1, len(frames)):
    gray_prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)

    # Absolute difference
    diff = cv2.absdiff(gray_prev, gray_curr)
    diffs.append(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR))

    # Threshold to get binary motion mask
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh_imgs.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Draw bounding boxes on original frame
    annotated = frames[i].copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, 'Motion', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    motion_frames.append(annotated)

# --- Build composite display ---
# Show 5 consecutive pairs: diff, threshold, detection
# Row 1 labels
font = cv2.FONT_HERSHEY_SIMPLEX
for j, d in enumerate(diffs):
    cv2.putText(d, f'Diff {j}<->{j+1}', (10, 20), font, 0.4, (0, 200, 255), 1)
for j, t in enumerate(thresh_imgs):
    cv2.putText(t, f'Thresh {j}<->{j+1}', (10, 20), font, 0.4, (0, 200, 255), 1)
for j, m in enumerate(motion_frames):
    cv2.putText(m, f'Detected {j+1}', (10, 20), font, 0.4, (0, 200, 255), 1)

# Use first 4 pairs for display
n = min(4, len(diffs))
row1 = np.hstack(diffs[:n])
row2 = np.hstack(thresh_imgs[:n])
row3 = np.hstack(motion_frames[:n])

# Add row labels
def make_label(text, width):
    bar = np.zeros((25, width, 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 18), font, 0.5, (200, 200, 200), 1)
    return bar

result = np.vstack([
    make_label('Frame Differences (absdiff)', row1.shape[1]),
    row1,
    make_label('Thresholded Motion Masks', row2.shape[1]),
    row2,
    make_label('Motion Detection Results', row3.shape[1]),
    row3
])

print(f'Processed {len(diffs)} frame pairs')
print(f'Threshold: 25, Dilation iterations: 2')

cv2.imshow('Frame Differencing', result)
```
