---
slug: 115-live-background-removal
title: "Live Background Removal"
level: live
concepts: [cv2.createBackgroundSubtractorMOG2, foreground masking, background replacement, cv2.bitwise_and]
prerequisites: [100-live-camera-fps, 74-background-subtraction, 14-bitwise-operations]
---

## What Problem Are We Solving?

Video conferencing apps, broadcast studios, and content creation tools all offer "virtual background" features -- the ability to replace everything behind you with a different image, a solid color, or a blurred version of the scene. This effect was historically achieved with a physical green screen and chroma keying, but modern approaches use computational background subtraction to separate foreground from background without any special backdrop.

OpenCV provides the MOG2 (Mixture of Gaussians 2) background subtractor, a statistical algorithm that models each pixel as a mixture of Gaussian distributions. Over time, it learns what the "background" looks like and classifies new pixels as either foreground or background. By combining the resulting foreground mask with bitwise operations, you can replace the background with anything: a solid color (green screen effect), a gradient, or a completely different scene.

This kata teaches you to run MOG2 on a live camera feed, clean up the noisy foreground mask using morphological operations, and composite the foreground onto a replacement background. You will toggle between multiple visualization modes -- original feed, raw mask, green screen, and custom gradient background -- to understand each stage of the pipeline.

## How MOG2 Background Subtraction Works

MOG2 models each pixel's color history as a mixture of K Gaussian distributions (typically K=5). For each new frame, each pixel is compared against its mixture model:

- If the pixel value is close to one of the Gaussians in the mixture, it is classified as **background** and that Gaussian's parameters are updated.
- If the pixel value does not match any Gaussian well enough, it is classified as **foreground** and a new Gaussian may replace the least probable one in the mixture.

This approach naturally adapts to gradual lighting changes (time of day, clouds passing), swaying curtains, and other slow background variations. Fast-moving objects (like a person) do not get absorbed into the model quickly, so they remain classified as foreground.

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)
```

### createBackgroundSubtractorMOG2 Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `history` | int | 500 | Number of recent frames used to build the background model. Higher values make the model slower to adapt but more stable |
| `varThreshold` | float | 16 | Threshold on the squared Mahalanobis distance between a pixel and the model. Higher values classify more pixels as background (less sensitive to change) |
| `detectShadows` | bool | True | When True, the algorithm detects shadows and marks them as gray (127) in the mask instead of white (255). Useful for cleaner masks but adds ~10% computation |

### How `history` Affects the Model

| History Value | Behavior | Use Case |
|---|---|---|
| 50-100 | Adapts very quickly -- moving objects are absorbed into the background within a few seconds | Scenes with frequent changes |
| 300-500 | Moderate adaptation -- good default for webcam use | Video conferencing, general use |
| 1000-2000 | Very slow adaptation -- background is essentially fixed after initial learning | Static camera surveillance |
| 5000+ | Near-permanent model -- only the very first frames matter | One-time background capture |

### How `varThreshold` Affects Sensitivity

| varThreshold | Behavior |
|---|---|
| 10-20 | Very sensitive -- even small color fluctuations are classified as foreground. More noisy masks |
| 30-50 | Balanced sensitivity. Good starting point for indoor webcam scenes |
| 80-150 | Low sensitivity -- only very different pixels are classified as foreground. Cleaner mask but may miss subtle foreground edges |
| 200+ | Very low sensitivity -- only dramatic changes trigger foreground classification |

## Applying the Background Subtractor

Each frame, call `.apply()` to get the foreground mask:

```python
fg_mask = bg_subtractor.apply(frame)
```

The returned `fg_mask` is a single-channel `uint8` image where:
- `255` = definite foreground
- `127` = shadow (if `detectShadows=True`)
- `0` = background

### Handling Shadows

Shadows appear as gray (127) in the mask. For clean foreground extraction, you typically want to either include shadows as foreground or exclude them entirely:

```python
# Option 1: Treat shadows as foreground (threshold at 100)
_, fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

# Option 2: Exclude shadows -- only keep definite foreground (threshold at 200)
_, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
```

### The learningRate Parameter

The `.apply()` method accepts an optional `learningRate` that overrides the default rate computed from `history`:

```python
# During initialization (first 60 frames): learn fast
fg_mask = bg_subtractor.apply(frame, learningRate=0.05)

# During normal operation: learn slowly
fg_mask = bg_subtractor.apply(frame, learningRate=0.002)

# Freeze the model: stop learning entirely
fg_mask = bg_subtractor.apply(frame, learningRate=0)
```

| Learning Rate | Effect |
|---|---|
| `-1` (default) | Automatically computed from `history` parameter |
| `0.0` | No learning -- background model is frozen |
| `0.001-0.005` | Very slow learning, stable background |
| `0.01-0.05` | Moderate learning, adapts to gradual changes |
| `1.0` | Instant learning -- current frame becomes the entire background model |

## Morphological Cleanup of the Mask

The raw MOG2 mask is noisy -- it contains small holes inside the foreground (where your shirt has a flat region that matches the background model) and scattered false-positive pixels in the background (sensor noise, light flicker). Morphological operations clean this up systematically:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Step 1: Remove small noise specks (opening = erosion followed by dilation)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 2: Fill small holes in the foreground (closing = dilation followed by erosion)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

# Step 3 (optional): Dilate slightly to capture edge pixels
fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
```

### Morphological Operations Summary

| Operation | What It Does | Effect on Mask | When to Use |
|---|---|---|---|
| `MORPH_OPEN` | Erode then dilate | Removes small white noise specks in the background | Always -- first cleanup step |
| `MORPH_CLOSE` | Dilate then erode | Fills small black holes inside the foreground region | Always -- second cleanup step |
| `MORPH_DILATE` | Expand white regions outward | Grows the foreground boundary | When edges are being clipped too tightly |
| `MORPH_ERODE` | Shrink white regions inward | Trims the foreground boundary | When there is a halo around the foreground |

### Order Matters

Apply `MORPH_OPEN` before `MORPH_CLOSE`. Opening first removes noise, then closing fills holes. If you reverse the order, closing first preserves noise specks by expanding them, making subsequent opening less effective.

### Kernel Shape and Size

| Kernel | Shape | Best For |
|---|---|---|
| `cv2.MORPH_ELLIPSE` | Circular | Smoother edges, better for organic shapes like people |
| `cv2.MORPH_RECT` | Square | General-purpose, fastest to compute |
| `cv2.MORPH_CROSS` | Cross/plus | Directional cleanup, preserves thin features |

Larger kernels (7x7, 9x9) produce more aggressive cleanup but can erode fine details like fingers and hair. A 5x5 elliptical kernel with 2-3 iterations is a good starting point.

## Background Replacement with Bitwise Operations

The core technique uses `cv2.bitwise_and` to extract the foreground from the camera frame and combine it with a replacement background:

```python
# Extract foreground pixels from the original frame
fg = cv2.bitwise_and(frame, frame, mask=fg_mask)

# Create the inverse mask for the background
bg_mask = cv2.bitwise_not(fg_mask)

# Extract background pixels from the replacement image
bg = cv2.bitwise_and(replacement_bg, replacement_bg, mask=bg_mask)

# Combine: foreground + background
result = cv2.add(fg, bg)
```

### cv2.bitwise_and with Mask

| Parameter | Type | Description |
|---|---|---|
| `src1` | ndarray | First input image |
| `src2` | ndarray | Second input image (same as src1 for masking) |
| `mask` | ndarray | Single-channel `uint8` mask. Only pixels where mask is non-zero are kept; all others become 0 |

Where the mask is 255 (foreground), the original pixel passes through unchanged. Where the mask is 0 (background), the pixel becomes black (0, 0, 0). By applying complementary masks to the camera frame and the replacement background, every output pixel comes from exactly one source.

## Creating Replacement Backgrounds

### Solid Color (Green Screen Effect)

```python
green_bg = np.zeros_like(frame)
green_bg[:] = (0, 200, 0)  # BGR green
```

### Vertical Gradient

```python
def create_gradient(height, width, color_top, color_bottom):
    """Create a vertical gradient from color_top to color_bottom."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for ch in range(3):
        gradient[:, :, ch] = np.linspace(
            color_top[ch], color_bottom[ch], height, dtype=np.uint8
        ).reshape(-1, 1)
    return gradient

gradient_bg = create_gradient(480, 640, (180, 50, 20), (50, 20, 120))
```

### Blurred Original (Bokeh / Portrait Mode)

```python
blurred_bg = cv2.GaussianBlur(frame, (55, 55), 0)
```

This creates the "portrait mode" effect where the background is blurred while the foreground person remains sharp.

## Soft-Edge Compositing with Alpha Blending

The bitwise approach produces hard edges at the foreground boundary, creating an obvious "paper cutout" look. For higher quality, blur the mask slightly and use it as a continuous alpha channel:

```python
soft_mask = cv2.GaussianBlur(fg_mask, (9, 9), 0).astype(float) / 255.0
alpha3 = np.stack([soft_mask] * 3, axis=-1)

result = (frame.astype(float) * alpha3 +
          replacement_bg.astype(float) * (1.0 - alpha3)).astype(np.uint8)
```

The blurred mask creates a gradual transition at the edges rather than an abrupt cutoff, producing much smoother and more natural-looking composites.

## The Learning Period

MOG2 needs several frames to build an accurate background model. During the first 30-100 frames, the mask will be very noisy because the model has not yet stabilized. Strategies to handle this:

1. **Let it learn naturally.** Show the empty background (without you in frame) for a few seconds before stepping into the camera view.
2. **Higher initial learning rate.** Use `learningRate=0.05` for the first 60 frames, then switch to the default `-1`.
3. **Display a learning indicator.** Show a progress bar or message during the learning phase so the user knows to wait.

## Tips & Common Mistakes

- **Keep the camera still during the learning phase.** MOG2 models the static background -- if you move the camera, the entire scene registers as foreground.
- **Always threshold the mask** to remove shadow values (127). Shadows cause semi-transparent artifacts in the composite.
- **Morphological cleanup order matters.** Apply `MORPH_OPEN` first (remove noise), then `MORPH_CLOSE` (fill holes). Reversing the order preserves noise.
- **Match replacement background size to the frame.** If your gradient or image is a different resolution than the camera frame, `cv2.resize` it to match `frame.shape[:2]`. Mismatched sizes cause crashes.
- **Blur the mask edges** for soft compositing. Hard binary masks create an obvious "paper cutout" effect.
- Avoid wearing clothing that matches your background color. MOG2 models color, so if your shirt is the same color as your wall, parts of you will be classified as background.
- If you move your chair or rearrange objects, the model treats the old positions as foreground until it learns the new arrangement. Use a reset key to reinitialize the model.
- `detectShadows=True` adds approximately 10% processing time. Disable it if performance is critical and you handle shadows via thresholding anyway.
- The `varThreshold` parameter has the most impact on mask quality. Experiment with values between 20 and 80 to find the best setting for your specific lighting environment.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Wait for the "Learning background" progress indicator to reach 100% before stepping into frame — keep the camera still during this phase
- Press **m** to cycle through modes: Original, Mask View, Green Screen, and Gradient BG — verify each mode displays correctly
- Press **r** to reset the background model (useful if you rearranged objects or moved the camera)

## Starter Code

```python
import cv2
import numpy as np
import time
from collections import deque

# --- Create background subtractor ---
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Create replacement backgrounds ---
# Solid green screen
green_bg = np.zeros((height, width, 3), dtype=np.uint8)
green_bg[:] = (0, 200, 0)

# Gradient background (dark blue to purple)
gradient_bg = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    ratio = i / height
    gradient_bg[i, :] = (
        int(180 * (1 - ratio) + 40 * ratio),   # B: 180 -> 40
        int(30 * (1 - ratio) + 10 * ratio),     # G: 30 -> 10
        int(20 * (1 - ratio) + 120 * ratio)     # R: 20 -> 120
    )

# --- Morphological kernel ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- State ---
mode = 0  # 0=original, 1=mask, 2=green screen, 3=gradient
mode_names = ["Original", "Mask View", "Green Screen", "Gradient BG"]
frame_times = deque(maxlen=30)
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        # --- Apply background subtraction ---
        # Higher learning rate for first 60 frames to build model faster
        lr = 0.05 if frame_count < 60 else -1
        fg_mask = bg_subtractor.apply(frame, learningRate=lr)

        # --- Threshold to remove shadows (keep only definite foreground) ---
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # --- Morphological cleanup ---
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # --- Choose output based on mode ---
        if mode == 0:
            output = frame.copy()

        elif mode == 1:
            # Show the cleaned foreground mask as a 3-channel image
            output = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        elif mode == 2:
            # Green screen: hard mask with bitwise operations
            fg = cv2.bitwise_and(frame, frame, mask=fg_mask)
            bg_mask = cv2.bitwise_not(fg_mask)
            bg = cv2.bitwise_and(green_bg, green_bg, mask=bg_mask)
            output = cv2.add(fg, bg)

        elif mode == 3:
            # Gradient background: soft-edge alpha compositing
            soft_mask = cv2.GaussianBlur(fg_mask, (9, 9), 0).astype(float) / 255.0
            alpha3 = np.stack([soft_mask] * 3, axis=-1)
            output = (frame.astype(float) * alpha3 +
                      gradient_bg.astype(float) * (1.0 - alpha3)).astype(np.uint8)

        # --- HUD overlay ---
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"Mode: {mode_names[mode]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        if frame_count < 60:
            progress = int((frame_count / 60) * 100)
            cv2.putText(output, f"Learning background... {progress}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(output, "'m'=next mode  'r'=reset model  'q'=quit",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live Background Removal', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = (mode + 1) % len(mode_names)
        elif key == ord('r'):
            # Reset the background model completely
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
            frame_count = 0
            print("Background model reset")

finally:
    cap.release()
    cv2.destroyAllWindows()
```
