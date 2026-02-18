---
slug: 72-writing-video
title: Writing Video Files
level: advanced
concepts: [cv2.VideoWriter, FourCC, codec selection]
prerequisites: [70-reading-video-files]
---

## What Problem Are We Solving?

After processing video frames — adding overlays, applying filters, or running detections — you need to **save the result as a video file**. OpenCV's `cv2.VideoWriter` lets you write frames one by one into a properly encoded video file. Understanding codecs, FourCC codes, and frame size requirements is essential for reliable video output.

## Creating a VideoWriter

The `VideoWriter` constructor takes four arguments:

```python
out = cv2.VideoWriter(filename, fourcc, fps, frameSize)
```

| Parameter | Meaning |
|---|---|
| `filename` | Output file path (e.g., `'output.avi'`) |
| `fourcc` | Four-character codec code |
| `fps` | Frames per second (e.g., `30.0`) |
| `frameSize` | Tuple of `(width, height)` — must match your frames exactly |

## Understanding FourCC Codes

FourCC (Four Character Code) identifies the video codec. You create one with:

```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
```

Common FourCC codes:

| Code | Codec | Container | Notes |
|---|---|---|---|
| `XVID` | MPEG-4 | `.avi` | Widely supported, good compression |
| `MJPG` | Motion JPEG | `.avi` | Large files, fast encoding |
| `mp4v` | MPEG-4 | `.mp4` | Good for MP4 containers |
| `H264` | H.264 | `.mp4` | Best compression, may need extra libs |
| `FFV1` | FFV1 | `.avi` | Lossless compression |

The `*'XVID'` syntax unpacks the string into four separate characters: `'X', 'V', 'I', 'D'`.

## Writing Frames

Write frames one at a time using `out.write()`:

```python
out.write(frame)
```

Each frame must be:
- A BGR color image (3 channels) unless you specified `isColor=False`
- The exact same `(width, height)` as declared in the constructor
- dtype `uint8`

## Complete Write Pattern

```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

for frame in my_frames:
    out.write(frame)

out.release()
```

## Writing Grayscale Video

To write grayscale frames, pass `isColor=False` as the fifth argument:

```python
out = cv2.VideoWriter('gray.avi', fourcc, 20.0, (640, 480), isColor=False)
```

Then write single-channel frames directly.

## Checking if VideoWriter is Ready

Just like `VideoCapture`, you can verify the writer opened successfully:

```python
if not out.isOpened():
    print('Error: Could not open VideoWriter')
```

## Tips & Common Mistakes

- The `frameSize` is `(width, height)`, **not** `(height, width)`. This is the opposite of NumPy's shape convention `(rows, cols)` = `(height, width)`.
- Every frame must match the declared size exactly. A frame with different dimensions will be silently dropped.
- Always call `out.release()` when done. Without it, the file may be corrupt or incomplete.
- Not all FourCC + container combinations work on all systems. If one codec fails, try `MJPG` with `.avi` — it works almost everywhere.
- `fps` is metadata — it tells the player how fast to play. Writing frames slower or faster doesn't change the actual frame rate in the file.
- On Linux, `XVID` requires the `libxvidcore` package. On macOS, `mp4v` with `.mp4` is often the safest choice.

## Starter Code

```python
import cv2
import numpy as np

# --- Demonstrate VideoWriter concepts with synthetic frames ---
# We create frames and show both the frames and the VideoWriter setup.

num_frames = 8
frame_h, frame_w = 200, 300
fps = 20.0

# FourCC code demonstration
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc_bytes = fourcc.to_bytes(4, 'little')
fourcc_str = fourcc_bytes.decode('ascii', errors='replace')

print(f'FourCC code: {fourcc} (0x{fourcc:08X})')
print(f'FourCC string: {fourcc_str}')
print(f'Frame size: {frame_w}x{frame_h}, FPS: {fps}')

# In a real scenario you would do:
#   out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_w, frame_h))
#   for frame in frames:
#       out.write(frame)
#   out.release()

# --- Generate frames that would be written to video ---
frames = []
for i in range(num_frames):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Gradient background that shifts over time
    for row in range(frame_h):
        blue = int(50 + 150 * row / frame_h)
        green = int(50 + 100 * np.sin(i * 0.5 + row * 0.02))
        frame[row, :] = (max(0, min(255, blue)),
                         max(0, min(255, green)),
                         40)

    # Draw a bouncing rectangle
    rect_x = int(30 + 200 * abs(np.sin(i * 0.5)))
    rect_y = int(50 + 80 * abs(np.cos(i * 0.4)))
    cv2.rectangle(frame, (rect_x, rect_y),
                  (rect_x + 60, rect_y + 40), (0, 255, 255), -1)

    # Add frame info
    cv2.putText(frame, f'Frame {i}/{num_frames}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'{frame_w}x{frame_h} @ {fps}fps', (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    frames.append(frame)

# --- Build an info panel ---
info_panel = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
info_panel[:] = (40, 30, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info_panel, 'VideoWriter Setup', (10, 30), font, 0.6, (0, 255, 0), 1)
cv2.putText(info_panel, f'Codec: XVID ({fourcc_str})', (10, 60), font, 0.45, (200, 200, 200), 1)
cv2.putText(info_panel, f'Container: .avi', (10, 85), font, 0.45, (200, 200, 200), 1)
cv2.putText(info_panel, f'Size: {frame_w}x{frame_h}', (10, 110), font, 0.45, (200, 200, 200), 1)
cv2.putText(info_panel, f'FPS: {fps}', (10, 135), font, 0.45, (200, 200, 200), 1)
cv2.putText(info_panel, f'Frames: {num_frames}', (10, 160), font, 0.45, (200, 200, 200), 1)

# --- Composite display ---
# Row 1: info + first 3 frames
# Row 2: remaining 5 frames (pad last slot if needed)
row1 = np.hstack([info_panel] + frames[:3])
# Pad to match row1 width
row2_frames = frames[3:7]
row2 = np.hstack(row2_frames)
# Pad row2 to match row1 width if needed
if row2.shape[1] < row1.shape[1]:
    pad = np.zeros((frame_h, row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
    row2 = np.hstack([row2, pad])

result = np.vstack([row1, row2])

print(f'Generated {num_frames} frames ready for VideoWriter')

cv2.imshow('Writing Video Files', result)
```
