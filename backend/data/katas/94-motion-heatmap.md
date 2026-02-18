---
slug: 94-motion-heatmap
title: Motion Heatmap
level: advanced
concepts: [frame accumulation, cv2.applyColorMap, heatmap overlay]
prerequisites: [73-frame-differencing]
---

## What Problem Are We Solving?

Security cameras record hours of footage, but most of the frame is static most of the time. A **motion heatmap** answers the question: "Where in the scene did the most activity happen?" Instead of watching all the footage, you generate a single image where hot colors (red/yellow) show high-activity areas and cool colors (blue/green) show low-activity areas.

The pipeline works by accumulating frame-to-frame differences over time, normalizing the accumulated values, applying a color map, and blending the result with a reference frame.

## Step 1: Generate Synthetic Video Frames

We simulate a sequence of frames with a moving object. Each frame has a static background, and a circle moves along a path, spending more time in some areas than others:

```python
frames = []
for i in range(num_frames):
    frame = background.copy()
    x = int(100 + 150 * np.sin(i * 0.1))
    y = int(200 + 80 * np.cos(i * 0.15))
    cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
    frames.append(frame)
```

## Step 2: Compute Frame Differences

For each consecutive pair of frames, we compute the absolute difference. This highlights pixels that changed between frames -- i.e., where motion occurred:

```python
diff = cv2.absdiff(frame_prev, frame_curr)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
```

The threshold step removes minor noise (sensor noise, compression artifacts) and keeps only significant changes.

## Step 3: Accumulate into a Heatmap

We use a **float32 accumulator** that sums up all the thresholded difference frames. Areas with frequent motion accumulate higher values:

```python
accumulator = np.zeros((h, w), dtype=np.float32)
for each frame difference:
    accumulator += thresh.astype(np.float32)
```

Using `float32` is critical -- `uint8` would overflow quickly since we're summing hundreds of frames.

## Step 4: Normalize to 0-255

After accumulation, we normalize to the 0-255 range so it can be displayed and color-mapped:

```python
if accumulator.max() > 0:
    heatmap = (accumulator / accumulator.max() * 255).astype(np.uint8)
```

## Step 5: Apply a Color Map

`cv2.applyColorMap` converts a single-channel grayscale image into a colorful visualization. `COLORMAP_JET` maps low values to blue and high values to red:

```python
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
```

| Colormap | Description |
|---|---|
| `cv2.COLORMAP_JET` | Blue (low) to red (high), classic heatmap |
| `cv2.COLORMAP_HOT` | Black to red to yellow to white |
| `cv2.COLORMAP_INFERNO` | Dark purple to yellow, perceptually uniform |
| `cv2.COLORMAP_TURBO` | Improved rainbow, better perceptual uniformity than JET |

## Step 6: Blend with the Original Frame

To see the heatmap in context (overlaid on the scene), use `cv2.addWeighted`:

```python
overlay = cv2.addWeighted(background, 0.6, heatmap_color, 0.4, 0)
```

The weights control transparency -- 0.6 for the background keeps the scene visible while 0.4 for the heatmap makes the hot zones clearly visible.

## The Complete Pipeline

1. **Input**: Sequence of video frames
2. **Frame Differencing**: Compute absolute difference between consecutive frames
3. **Threshold**: Remove noise, keep significant motion
4. **Accumulate**: Sum differences into a float32 buffer
5. **Normalize**: Scale accumulated values to 0-255
6. **Color Map**: Apply `cv2.applyColorMap` for visualization
7. **Overlay**: Blend heatmap with a reference frame

## Tips & Common Mistakes

- Use `float32` or `float64` for the accumulator. Using `uint8` causes overflow after just a few frames and produces incorrect results.
- The threshold value for frame differencing affects sensitivity. Too low and sensor noise creates false heat; too high and slow movements are missed.
- `cv2.applyColorMap` expects a single-channel `uint8` input. Passing a float or multi-channel image produces wrong results.
- For long videos, consider using `cv2.accumulateWeighted` instead of simple summation. This gives recent motion more weight than old motion (exponential decay).
- The heatmap resolution matches the frame resolution. Downsample frames for faster processing if you don't need pixel-level precision.
- Normalize AFTER all frames are accumulated, not per-frame, to get a true cumulative heatmap.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create synthetic video frames with a moving object
# =============================================================
frame_h, frame_w = 300, 500
num_frames = 120

# Static background: a room-like scene
background = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
background[:] = (50, 45, 40)
# Floor
cv2.rectangle(background, (0, 200), (frame_w, frame_h), (70, 65, 55), -1)
# Wall features
cv2.rectangle(background, (50, 50), (130, 150), (80, 70, 60), -1)   # Window
cv2.rectangle(background, (350, 80), (420, 180), (80, 70, 60), -1)  # Door

# Generate frames with a moving object (ball bouncing around)
frames = []
# Object follows a Lissajous-like path to create varied coverage
for i in range(num_frames):
    frame = background.copy()

    # Primary moving object: ball on a Lissajous path
    t = i * 0.08
    x = int(250 + 180 * np.sin(t * 1.3))
    y = int(150 + 100 * np.sin(t * 0.9))
    x = np.clip(x, 25, frame_w - 25)
    y = np.clip(y, 25, frame_h - 25)
    cv2.circle(frame, (x, y), 20, (0, 100, 255), -1)

    # Secondary object: smaller dot with different frequency
    x2 = int(150 + 80 * np.cos(t * 2.1))
    y2 = int(220 + 50 * np.sin(t * 1.7))
    x2 = np.clip(x2, 10, frame_w - 10)
    y2 = np.clip(y2, 10, frame_h - 10)
    cv2.circle(frame, (x2, y2), 10, (255, 100, 0), -1)

    frames.append(frame)

print(f'Generated {num_frames} synthetic frames ({frame_w}x{frame_h})')

# =============================================================
# Step 2: Accumulate frame differences
# =============================================================
accumulator = np.zeros((frame_h, frame_w), dtype=np.float32)

for i in range(1, num_frames):
    # Compute absolute difference between consecutive frames
    diff = cv2.absdiff(frames[i], frames[i - 1])
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to remove noise
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # Accumulate
    accumulator += thresh.astype(np.float32)

print(f'Accumulator range: [{accumulator.min():.0f}, {accumulator.max():.0f}]')

# =============================================================
# Step 3: Normalize to 0-255
# =============================================================
if accumulator.max() > 0:
    heatmap_gray = (accumulator / accumulator.max() * 255).astype(np.uint8)
else:
    heatmap_gray = np.zeros((frame_h, frame_w), dtype=np.uint8)

print(f'Heatmap range after normalization: [{heatmap_gray.min()}, {heatmap_gray.max()}]')

# =============================================================
# Step 4: Apply color maps
# =============================================================
heatmap_jet = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
heatmap_hot = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)
heatmap_inferno = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_INFERNO)

# =============================================================
# Step 5: Overlay heatmap on background
# =============================================================
overlay_jet = cv2.addWeighted(background, 0.6, heatmap_jet, 0.4, 0)
overlay_hot = cv2.addWeighted(background, 0.6, heatmap_hot, 0.4, 0)

# =============================================================
# Step 6: Build visualization
# =============================================================
font = cv2.FONT_HERSHEY_SIMPLEX

# Show a sample frame, the raw heatmap, and overlaid versions
sample_frame = frames[num_frames // 2].copy()
cv2.putText(sample_frame, 'Sample Frame', (10, 25), font, 0.5, (0, 255, 0), 1)

heatmap_gray_bgr = cv2.cvtColor(heatmap_gray, cv2.COLOR_GRAY2BGR)
cv2.putText(heatmap_gray_bgr, 'Raw Heatmap', (10, 25), font, 0.5, (0, 255, 0), 1)

cv2.putText(heatmap_jet, 'JET Colormap', (10, 25), font, 0.5, (255, 255, 255), 1)
cv2.putText(overlay_jet, 'JET Overlay', (10, 25), font, 0.5, (0, 255, 0), 1)

cv2.putText(heatmap_hot, 'HOT Colormap', (10, 25), font, 0.5, (255, 255, 255), 1)
cv2.putText(overlay_hot, 'HOT Overlay', (10, 25), font, 0.5, (0, 255, 0), 1)

# Grid: 3 rows x 2 cols
row1 = np.hstack([sample_frame, heatmap_gray_bgr])
row2 = np.hstack([heatmap_jet, overlay_jet])
row3 = np.hstack([heatmap_hot, overlay_hot])

result = np.vstack([row1, row2, row3])

# Count motion pixels
motion_pixels = cv2.countNonZero(heatmap_gray)
total_pixels = frame_h * frame_w
print(f'Pixels with any motion: {motion_pixels}/{total_pixels} '
      f'({100*motion_pixels/total_pixels:.1f}%)')

cv2.imshow('Motion Heatmap', result)
```
