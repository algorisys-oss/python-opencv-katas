---
slug: 70-reading-video-files
title: Reading Video Files
level: advanced
concepts: [cv2.VideoCapture, frame loop, release]
prerequisites: [01-image-loading]
---

## What Problem Are We Solving?

Images are single snapshots, but most real-world visual data comes as **video** — a sequence of frames played at a certain rate. To process video with OpenCV, you need to know how to **open a video file, read frames one at a time**, and properly release resources when done.

`cv2.VideoCapture` is OpenCV's universal interface for reading video — whether from a file, a webcam, or a network stream. Understanding how to loop through frames is the foundation for every video processing pipeline.

## Opening a Video File

You create a `VideoCapture` object by passing a file path:

```python
cap = cv2.VideoCapture('video.mp4')
```

Always check if the video was opened successfully:

```python
if not cap.isOpened():
    print('Error: Could not open video')
```

## Reading Frames in a Loop

Each call to `cap.read()` returns two values — a boolean indicating success and the frame itself:

```python
ret, frame = cap.read()
```

The typical pattern is a `while` loop that reads until no more frames are available:

```python
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process the frame here
```

When `ret` is `False`, it means the video has ended (or there was a read error).

## Retrieving Video Properties

You can query metadata from the capture object using `cap.get()`:

```python
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```

| Property | Meaning |
|---|---|
| `CAP_PROP_FRAME_WIDTH` | Width of each frame in pixels |
| `CAP_PROP_FRAME_HEIGHT` | Height of each frame in pixels |
| `CAP_PROP_FPS` | Frames per second |
| `CAP_PROP_FRAME_COUNT` | Total number of frames |
| `CAP_PROP_POS_FRAMES` | Current frame position (0-indexed) |

## Seeking to a Specific Frame

You can jump to a particular frame number using `cap.set()`:

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Jump to frame 100
ret, frame = cap.read()
```

This is useful for skipping ahead or implementing random access within a video.

## Releasing Resources

Always release the capture object when done — this frees the file handle and any internal buffers:

```python
cap.release()
```

Failing to release can cause resource leaks, especially when processing many files in sequence.

## Tips & Common Mistakes

- Always check `cap.isOpened()` after creating a `VideoCapture`. A typo in the file path silently returns an empty capture.
- Always check the `ret` value from `cap.read()`. Reading past the end of a video returns `(False, None)`.
- `CAP_PROP_FRAME_COUNT` may be inaccurate for some codecs. Don't rely on it for precise frame counting — use the read loop instead.
- Call `cap.release()` in a `finally` block or use a context manager pattern to ensure cleanup.
- Frame indices are 0-based. The first frame is frame 0.
- Seeking with `cap.set()` may not be frame-accurate for all video codecs. Seek to a keyframe and then read forward for best results.

## Starter Code

```python
import cv2
import numpy as np

# --- Simulate a video as a sequence of synthetic frames ---
# Since we don't have a real video file, we create frames programmatically
# to demonstrate how VideoCapture-style frame processing works.

num_frames = 8
frame_h, frame_w = 200, 300

frames = []
for i in range(num_frames):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    # Draw a circle that moves across the frame
    x = int(40 + i * (frame_w - 80) / (num_frames - 1))
    y = frame_h // 2
    color = (0, 255 - i * 25, i * 30)
    cv2.circle(frame, (x, y), 25, color, -1)

    # Add frame number label
    cv2.putText(frame, f'Frame {i}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    frames.append(frame)

# --- Simulate the read loop and display properties ---
# In a real scenario, you would do:
#   cap = cv2.VideoCapture('video.mp4')
#   while True:
#       ret, frame = cap.read()
#       if not ret: break
#       # process frame
#   cap.release()

print(f'Simulated video: {num_frames} frames, {frame_w}x{frame_h}')
print(f'Frame dtype: {frames[0].dtype}, shape: {frames[0].shape}')

# --- Build a composite showing all frames ---
# Arrange frames in 2 rows of 4
row1 = np.hstack(frames[:4])
row2 = np.hstack(frames[4:])
result = np.vstack([row1, row2])

# Add title
cv2.putText(result, 'Video Frame Sequence (simulated)', (10, result.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

print('Displaying all frames as a composite grid')

cv2.imshow('Reading Video Files', result)
```
