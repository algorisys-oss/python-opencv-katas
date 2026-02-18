---
slug: 117-live-mirror-kaleidoscope
title: "Live Mirror & Kaleidoscope"
level: live
concepts: [cv2.flip, array slicing, image tiling, mirror effects, kaleidoscope geometry]
prerequisites: [100-live-camera-fps, 10-image-flipping]
---

## What Problem Are We Solving?

Mirror and kaleidoscope effects are among the most visually striking real-time transformations you can apply to a camera feed -- and they are surprisingly simple to implement. A mirror effect duplicates a portion of the frame and reflects it, creating perfect symmetry. A kaleidoscope goes further by tiling reflected copies into repeating geometric patterns, producing mesmerizing visuals from ordinary scenes.

These effects are used in creative video tools, photo booth applications, VJ (video jockey) software, art installations, and social media filters. They are also excellent exercises for understanding NumPy array slicing, image coordinate systems, and how `cv2.flip` works under the hood. Every effect in this kata is built from just two primitives: array slicing (to extract a region) and flipping (to create its mirror image).

This kata builds four distinct modes -- horizontal mirror, vertical mirror, quad-mirror kaleidoscope, and a triangular kaleidoscope with rotational symmetry -- that you cycle through with keyboard controls.

## cv2.flip: The Foundation of All Mirror Effects

`cv2.flip` mirrors an image around one or both axes:

```python
flipped = cv2.flip(src, flipCode)
```

### flipCode Values

| flipCode | Axis | Effect | Visual Description |
|---|---|---|---|
| `0` | Horizontal axis (x-axis) | Top-bottom flip | Image appears upside down |
| `1` | Vertical axis (y-axis) | Left-right flip | Image appears as a mirror reflection |
| `-1` | Both axes | 180-degree rotation | Image is rotated 180 degrees |

```python
upside_down = cv2.flip(frame, 0)   # Top becomes bottom
mirror       = cv2.flip(frame, 1)   # Left becomes right
rotated_180  = cv2.flip(frame, -1)  # Both axes flipped
```

### Performance

`cv2.flip` is extremely fast because it does not compute new pixel values -- it only rearranges the order in which existing pixels are accessed. On a 640x480 frame, it takes less than 0.5 ms. This makes mirror effects essentially free in terms of processing budget.

## Array Slicing for Region Extraction

NumPy array slicing extracts rectangular regions from an image without copying data (it creates a view):

```python
h, w = frame.shape[:2]
mid_w = w // 2
mid_h = h // 2

left_half   = frame[:, :mid_w]         # All rows, columns 0 to mid_w-1
right_half  = frame[:, mid_w:]          # All rows, columns mid_w to end
top_half    = frame[:mid_h, :]          # Rows 0 to mid_h-1, all columns
bottom_half = frame[mid_h:, :]          # Rows mid_h to end, all columns
top_left    = frame[:mid_h, :mid_w]     # Top-left quadrant
```

### Slicing Syntax Reminder

| Slice | Meaning |
|---|---|
| `frame[a:b, c:d]` | Rows a through b-1, columns c through d-1 |
| `frame[:, :mid]` | All rows, first `mid` columns |
| `frame[:mid, :]` | First `mid` rows, all columns |
| `frame[:mid, :mid]` | Top-left quadrant |

The colon `:` alone means "all" along that dimension. Omitting the start means 0; omitting the end means the full extent.

## Effect 1: Horizontal Mirror (Left Half Reflected Right)

Take the left half of the frame, flip it left-to-right, and place it where the right half was:

```python
h, w = frame.shape[:2]
mid_w = w // 2

left_half = frame[:, :mid_w]
mirrored_right = cv2.flip(left_half, 1)  # Flip left-right
result = np.hstack([left_half, mirrored_right])
```

This creates a face with perfect bilateral symmetry. Since human faces are never perfectly symmetric, the result always looks subtly different from reality -- sometimes unsettling, sometimes amusing.

### Mirroring the Other Half

You can also mirror the right half to the left for a different symmetric appearance:

```python
right_half = frame[:, mid_w:]
mirrored_left = cv2.flip(right_half, 1)
result = np.hstack([mirrored_left, right_half])
```

Since left and right halves of a face have different character (expression asymmetry, dominant eye, etc.), mirroring each side produces noticeably different portraits.

## Effect 2: Vertical Mirror (Top Half Reflected Down)

The same concept applied along the vertical axis:

```python
h, w = frame.shape[:2]
mid_h = h // 2

top_half = frame[:mid_h, :]
mirrored_bottom = cv2.flip(top_half, 0)  # Flip top-bottom
result = np.vstack([top_half, mirrored_bottom])
```

This creates an eerie reflection effect, like looking at a calm lake reflecting the scene above it. The symmetry line runs horizontally across the middle of the frame.

## Effect 3: Quad Mirror (Top-Left Reflected to All Quadrants)

The quad-mirror kaleidoscope takes one quadrant and fills all four positions with reflected copies:

```python
h, w = frame.shape[:2]
mid_h, mid_w = h // 2, w // 2

# Source: top-left quadrant
quad = frame[:mid_h, :mid_w]

# Create three mirrored versions
quad_lr   = cv2.flip(quad, 1)    # Left-right mirror
quad_tb   = cv2.flip(quad, 0)    # Top-bottom mirror
quad_both = cv2.flip(quad, -1)   # Both axes (180-degree rotation)

# Assemble 2x2 grid
top_row = np.hstack([quad, quad_lr])
bot_row = np.hstack([quad_tb, quad_both])
result  = np.vstack([top_row, bot_row])
```

### Why It Looks Like a Kaleidoscope

The four quadrants share edges seamlessly because each adjacent pair is a mirror of the other. At every boundary, pixels match perfectly -- the left edge of `quad_lr` is identical to the right edge of `quad`, and the top edge of `quad_tb` is identical to the bottom edge of `quad`. The eye perceives this as a single continuous symmetric pattern rather than four separate tiles.

### How the Quadrants Are Arranged

```
+----+----+
| TL | TR |   TL = original quadrant
|    | LR |   TR = left-right flip of TL
+----+----+
| BL | BR |   BL = top-bottom flip of TL
| TB |BOTH|   BR = both-axes flip of TL
+----+----+
```

Each edge between adjacent quadrants is a mirror line:
- Vertical center line: TL and TR are mirrors
- Horizontal center line: TL and BL are mirrors
- BR mirrors both TL diagonals

### Choosing Different Source Quadrants

The visual pattern depends entirely on which quadrant you use as the source:

| Source Quadrant | Slice | Visual Character |
|---|---|---|
| Top-left | `frame[:mid_h, :mid_w]` | Whatever is in the upper-left camera view |
| Top-right | `frame[:mid_h, mid_w:]` | Upper-right camera content |
| Bottom-left | `frame[mid_h:, :mid_w]` | Lower-left camera content |
| Center crop | `frame[qh:qh+mid_h, qw:qw+mid_w]` | Centers the subject in the pattern |

## Effect 4: Kaleidoscope with Triangular Tiling

A real kaleidoscope uses triangular mirrors, not rectangular ones. This creates a more organic, rotational pattern. The approach: extract a triangular region, create its mirror, and tile them into a symmetric arrangement.

```python
h, w = frame.shape[:2]
size = min(h, w)

# Crop to square for clean geometry
square = frame[:size, :size]

# Create a triangular mask (upper-left triangle)
mask = np.zeros((size, size), dtype=np.uint8)
pts = np.array([[0, 0], [size - 1, 0], [0, size - 1]], dtype=np.int32)
cv2.fillPoly(mask, [pts], 255)

# Extract the triangle
triangle = cv2.bitwise_and(square, square, mask=mask)

# Create the mirror by transposing (swaps x and y axes)
transposed = cv2.transpose(triangle)

# Combine: original triangle fills upper-left, transpose fills lower-right
combined = cv2.add(triangle, transposed)
```

### Why cv2.transpose Creates the Mirror

`cv2.transpose` swaps rows and columns -- pixel at position `(r, c)` moves to position `(c, r)`. For our upper-left triangle (above the diagonal), transposing reflects it to the lower-right triangle (below the diagonal). The result fills the entire square with a symmetric pattern around the diagonal.

### cv2.transpose vs cv2.flip

| Operation | What It Does | Geometric Effect |
|---|---|---|
| `cv2.flip(img, 1)` | Reverses column order | Mirror around vertical axis |
| `cv2.flip(img, 0)` | Reverses row order | Mirror around horizontal axis |
| `cv2.transpose(img)` | Swaps rows and columns | Mirror around the main diagonal |

`cv2.transpose` is unique because it produces a reflection that `cv2.flip` alone cannot achieve -- the diagonal mirror that physical kaleidoscopes create.

### Resizing Back to Original Frame Size

The kaleidoscope produces a square output. To display it in the original frame dimensions, resize it:

```python
result = cv2.resize(combined, (w, h))
```

## Handling Odd Frame Dimensions

When combining mirrored halves, the dimensions must match exactly. For frames with odd width or height, the two halves differ by one pixel:

```python
h, w = frame.shape[:2]
mid_w = w // 2

left_half = frame[:, :mid_w]           # mid_w columns
right_half = frame[:, mid_w:mid_w*2]   # Also mid_w columns (discard extra pixel)
```

The simplest approach is to crop the frame to even dimensions at the start:

```python
# Ensure even dimensions by trimming one pixel if needed
frame_even = frame[:mid_h * 2, :mid_w * 2]
```

This avoids `np.hstack` and `np.vstack` dimension mismatch errors.

## np.hstack and np.vstack for Assembly

These NumPy functions concatenate arrays along the horizontal and vertical axes:

```python
# Horizontal concatenation: images side by side (same height required)
combined = np.hstack([left_image, right_image])

# Vertical concatenation: images stacked top to bottom (same width required)
combined = np.vstack([top_image, bottom_image])
```

| Function | Axis | Requirement |
|---|---|---|
| `np.hstack` | Horizontal (columns) | All images must have the same height (number of rows) |
| `np.vstack` | Vertical (rows) | All images must have the same width (number of columns) |

If the requirement is not met, NumPy raises a `ValueError` with a shape mismatch message. This is the most common error when building mirror effects with odd-dimension frames.

## Performance Considerations

| Operation | Typical Time (640x480) |
|---|---|
| `cv2.flip` | < 0.5 ms |
| NumPy slicing (extract half/quadrant) | < 0.1 ms (creates a view, no copy) |
| `np.hstack` / `np.vstack` | < 1 ms |
| `cv2.transpose` | < 0.5 ms |
| `cv2.fillPoly` (triangle mask, once) | < 0.1 ms |
| `cv2.bitwise_and` (triangle extraction) | < 0.5 ms |

All mirror and kaleidoscope effects are extremely fast -- well under 2 ms total per frame. These are among the cheapest real-time effects possible, making them ideal for combining with other heavier processing (blur, edge detection, etc.) in the same pipeline.

## Tips & Common Mistakes

- **`np.hstack` requires identical height** for all images; `np.vstack` requires identical width. If dimensions mismatch by even 1 pixel (common with odd frame sizes), you get a `ValueError`. Always ensure even dimensions before splitting.
- **`cv2.flip` returns a new array** -- it does not modify the input in place. Always capture the return value: `flipped = cv2.flip(src, 1)`.
- For the quad kaleidoscope, use `[:mid_h, :mid_w]` (exclusive upper bound). Using `[:mid_h+1, :mid_w+1]` creates a quadrant that is too large by one pixel, causing stacking errors.
- The triangular kaleidoscope requires a **square** input. Either crop or resize the frame to a square before processing.
- When stacking arrays, all parts must have the **same number of channels**. Mixing BGR (3-channel) and grayscale (1-channel) causes a shape mismatch error.
- **Add the FPS overlay after building the effect**, not before. If you add text before mirroring, the text gets mirrored too, making it unreadable.
- `cv2.transpose` swaps rows and columns, which also swaps width and height. A 640x480 image transposed becomes 480x640. Keep this in mind when assembling tiles.
- For the most visually interesting kaleidoscope patterns, position your face or an object near one edge of the frame rather than the center. The asymmetry of the source creates more complex symmetric patterns.
- All these effects work on the raw pixel data with no color space conversion needed -- they operate directly on BGR frames.

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_times = deque(maxlen=30)

modes = ["Normal", "H-Mirror (Left)", "V-Mirror (Top)", "Quad Kaleidoscope", "Triangle Kaleidoscope"]
mode = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]
        mid_w = w // 2
        mid_h = h // 2

        # Ensure even dimensions for clean splits
        frame_even = frame[:mid_h * 2, :mid_w * 2]
        h, w = frame_even.shape[:2]
        mid_w = w // 2
        mid_h = h // 2

        if mode == 0:
            # --- Normal: no effect ---
            display = frame_even.copy()

        elif mode == 1:
            # --- Horizontal mirror: left half reflected to right ---
            left = frame_even[:, :mid_w]
            display = np.hstack([left, cv2.flip(left, 1)])

        elif mode == 2:
            # --- Vertical mirror: top half reflected to bottom ---
            top = frame_even[:mid_h, :]
            display = np.vstack([top, cv2.flip(top, 0)])

        elif mode == 3:
            # --- Quad kaleidoscope: top-left quadrant mirrored to all four ---
            quad = frame_even[:mid_h, :mid_w]
            quad_lr = cv2.flip(quad, 1)      # Left-right mirror
            quad_tb = cv2.flip(quad, 0)      # Top-bottom mirror
            quad_both = cv2.flip(quad, -1)   # Both axes

            top_row = np.hstack([quad, quad_lr])
            bot_row = np.hstack([quad_tb, quad_both])
            display = np.vstack([top_row, bot_row])

        elif mode == 4:
            # --- Triangle kaleidoscope: diagonal mirror with rotational symmetry ---
            size = min(h, w)
            square = frame_even[:size, :size]

            # Create triangular mask (upper-left triangle)
            mask = np.zeros((size, size), dtype=np.uint8)
            pts = np.array([[0, 0], [size - 1, 0], [0, size - 1]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # Extract triangle and create its diagonal mirror
            triangle = cv2.bitwise_and(square, square, mask=mask)
            transposed = cv2.transpose(triangle)

            # Combine original triangle + transposed mirror
            combined = cv2.add(triangle, transposed)

            # Resize back to original frame dimensions
            display = cv2.resize(combined, (w, h))

        # --- HUD overlay (after effect so text is not mirrored) ---
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Mode: {modes[mode]}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(display, "'n'=next mode  'p'=prev mode  'q'=quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Mirror & Kaleidoscope', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            mode = (mode + 1) % len(modes)
        elif key == ord('p'):
            mode = (mode - 1) % len(modes)

finally:
    cap.release()
    cv2.destroyAllWindows()
```
