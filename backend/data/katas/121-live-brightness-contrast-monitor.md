---
slug: 121-live-brightness-contrast-monitor
title: Live Brightness & Contrast Monitor
level: live
concepts: [histogram analysis, mean brightness, standard deviation, exposure warnings, real-time metrics]
prerequisites: [100-live-camera-fps, 18-understanding-histograms, 103-live-histogram-display]
---

## What Problem Are We Solving?

Photographers and videographers constantly monitor exposure to avoid losing detail in shadows (underexposure) or highlights (overexposure). Professional cinema cameras have built-in scopes and meters that display real-time statistics. Security camera operators need to know when a camera's view has degraded due to lighting changes. Machine vision systems need to verify that illumination is within acceptable bounds before running analysis.

A live brightness and contrast monitor provides a **dashboard of image quality metrics** computed from every frame: mean brightness, standard deviation (contrast), clipping percentages for shadows and highlights, a brightness gauge, and a live histogram. Together, these metrics give a complete picture of exposure quality, and color-coded warnings (green for good, yellow for warning, red for critical) make problems immediately visible.

This kata builds on the histogram concepts from kata 18 and the live histogram display from kata 103, adding quantitative metrics, thresholds, visual gauges, and a warning system that turns raw histogram data into actionable feedback.

## Brightness: Mean Pixel Intensity

The simplest measure of exposure is the **mean intensity** of the grayscale image:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mean_brightness = np.mean(gray)
```

| Mean Value | Interpretation |
|---|---|
| 0-50 | Very dark -- likely underexposed, detail lost in shadows |
| 50-80 | Dark -- acceptable for some scenes, but may need correction |
| 80-180 | Good exposure -- most detail preserved |
| 180-210 | Bright -- acceptable but approaching overexposure |
| 210-255 | Very bright -- likely overexposed, highlights clipped |

The "ideal" range depends on the scene. A nighttime shot naturally has a low mean; a snow scene naturally has a high mean. But for general-purpose monitoring, 80-180 is a useful default "good" range.

### Computing Mean from Histogram vs. from Pixels

You can compute the mean directly with `np.mean(gray)`, or derive it from the histogram:

```python
# Method 1: Direct (faster for the mean alone)
mean_val = np.mean(gray)

# Method 2: From histogram (useful when you already computed the histogram)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
bin_values = np.arange(256)
mean_from_hist = np.sum(bin_values * hist.flatten()) / np.sum(hist)
```

Both give the same result. If you are already computing the histogram for display, method 2 avoids redundant computation.

## Contrast: Standard Deviation

**Standard deviation** of pixel intensities is a practical measure of contrast:

```python
std_dev = np.std(gray)
```

| Std Dev | Interpretation |
|---|---|
| 0-20 | Very low contrast -- image looks flat, washed out, or foggy |
| 20-40 | Low contrast -- muted, may benefit from equalization |
| 40-80 | Good contrast -- clear separation between light and dark areas |
| 80+ | High contrast -- strong blacks and whites, possibly harsh |

A low standard deviation means most pixels are close to the mean (narrow histogram). A high standard deviation means pixels are spread across the intensity range (wide histogram).

## Clipping Detection: Overexposed and Underexposed Pixels

**Clipping** occurs when pixel values are crushed to 0 (pure black) or saturated to 255 (pure white). Clipped pixels have lost all detail and cannot be recovered.

```python
total_pixels = gray.size
underexposed = np.sum(gray <= 5) / total_pixels * 100    # % near black
overexposed = np.sum(gray >= 250) / total_pixels * 100   # % near white
```

Using a small margin (5 and 250 instead of exactly 0 and 255) catches near-clipping pixels that are effectively clipped for practical purposes.

| Clipping % | Severity | Action |
|---|---|---|
| 0-1% | Normal | No action needed |
| 1-5% | Mild | Monitor -- some detail loss in extremes |
| 5-15% | Warning | Adjust exposure -- noticeable detail loss |
| 15%+ | Critical | Significant data loss -- immediate correction needed |

## Dynamic Range

The **dynamic range** measures how much of the 0-255 intensity range the image actually uses:

```python
min_val = int(np.min(gray))
max_val = int(np.max(gray))
dynamic_range = max_val - min_val
```

| Dynamic Range | Interpretation |
|---|---|
| 0-50 | Very narrow -- image uses only a small part of the tonal range |
| 50-150 | Moderate -- common in controlled lighting |
| 150-255 | Wide -- image uses most of the available range |

A low dynamic range combined with a reasonable mean brightness indicates low contrast (e.g., overcast day, fog). A full dynamic range (close to 255) with high clipping indicates harsh lighting with strong shadows and highlights.

## Drawing a Brightness Gauge

A visual gauge provides an intuitive "speedometer" for brightness:

```python
def draw_brightness_gauge(brightness, width=200, height=25):
    """Draw a horizontal gauge bar for brightness 0-255."""
    gauge = np.zeros((height, width, 3), dtype=np.uint8)
    gauge[:] = (40, 40, 40)

    # Draw gradient background
    for x in range(width):
        intensity = int(255 * x / width)
        gray_val = intensity // 2
        gauge[:, x] = (gray_val, gray_val, gray_val)

    # Draw "good zone" markers
    good_start = int(width * 80 / 255)
    good_end = int(width * 180 / 255)
    cv2.rectangle(gauge, (good_start, 0), (good_end, 3), (0, 255, 0), -1)
    cv2.rectangle(gauge, (good_start, height - 3), (good_end, height), (0, 255, 0), -1)

    # Draw current brightness indicator
    pos = int(width * brightness / 255)
    pos = max(2, min(pos, width - 3))
    cv2.rectangle(gauge, (pos - 2, 0), (pos + 2, height), (0, 255, 255), -1)
    cv2.rectangle(gauge, (pos - 2, 0), (pos + 2, height), (255, 255, 255), 1)

    return gauge
```

## Color-Coded Status System

Combine all metrics into a single status indicator with traffic-light coloring:

```python
def get_status(mean_brightness, std_dev, underexposed_pct, overexposed_pct):
    """Return (status_text, color) based on image quality metrics."""
    issues = []

    # Brightness check
    if mean_brightness < 50:
        issues.append("VERY DARK")
    elif mean_brightness < 80:
        issues.append("Dark")
    elif mean_brightness > 210:
        issues.append("VERY BRIGHT")
    elif mean_brightness > 180:
        issues.append("Bright")

    # Contrast check
    if std_dev < 20:
        issues.append("LOW CONTRAST")

    # Clipping check
    if overexposed_pct > 15:
        issues.append("CLIPPING (highlights)")
    elif overexposed_pct > 5:
        issues.append("Highlight warning")

    if underexposed_pct > 15:
        issues.append("CLIPPING (shadows)")
    elif underexposed_pct > 5:
        issues.append("Shadow warning")

    if not issues:
        return "GOOD EXPOSURE", (0, 255, 0)  # Green
    elif any(word in ' '.join(issues) for word in ["VERY", "CLIPPING"]):
        return " | ".join(issues), (0, 0, 255)  # Red
    else:
        return " | ".join(issues), (0, 200, 255)  # Yellow
```

## Computing Histogram for Visualization

Reuse the histogram computation for both statistics and visualization:

```python
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Normalize for display
hist_display = hist.copy()
cv2.normalize(hist_display, hist_display, 0, hist_height - 10, cv2.NORM_MINMAX)

# Draw as bar chart
hist_canvas = np.zeros((hist_height, 256, 3), dtype=np.uint8)
for x in range(256):
    bar_h = int(hist_display[x])
    if bar_h > 0:
        # Color bars based on zone
        if x < 50:
            color = (200, 100, 50)    # Shadow zone -- blue-ish
        elif x > 210:
            color = (50, 100, 200)    # Highlight zone -- red-ish
        else:
            color = (200, 200, 200)   # Midtone zone -- white
        cv2.line(hist_canvas, (x, hist_height), (x, hist_height - bar_h), color, 1)
```

Color-coding the histogram bars makes it immediately obvious where the pixel mass is concentrated: blue/cool tones in the shadows, red/warm tones in the highlights, and neutral white in the midtones.

## Performance Considerations

Computing all these metrics every frame adds processing time. Benchmark the individual operations:

| Operation | Typical Time (640x480) |
|---|---|
| `cv2.cvtColor` (BGR to gray) | ~0.3 ms |
| `np.mean(gray)` | ~0.1 ms |
| `np.std(gray)` | ~0.2 ms |
| `cv2.calcHist` (1 channel, 256 bins) | ~0.5 ms |
| Clipping calculation (`np.sum`) | ~0.1 ms |
| Total | ~1.2 ms |

This overhead is negligible for a 30 FPS pipeline. No downsampling or skip-frame optimization is needed for these operations.

## Tips & Common Mistakes

- `np.mean(gray)` returns a float64. For display, cast to int or format with one decimal place.
- `cv2.calcHist` requires the image in a list: `[gray]`, not just `gray`. Missing the list wrapper is the most common bug.
- The grayscale histogram range `[0, 256]` has an exclusive upper bound -- it covers values 0 through 255.
- Clipping thresholds of 5 and 250 (instead of 0 and 255) provide more practical results. Exact 0/255 pixels include sensor noise at the extremes.
- Standard deviation is affected by the scene content, not just lighting. A checkerboard pattern has high std dev regardless of exposure. Use it as one signal among several, not the sole contrast metric.
- Auto-exposure cameras constantly adjust brightness, causing the mean to fluctuate. Disable auto-exposure (`cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)`) for consistent readings.
- Normalize the histogram display every frame. The distribution shape changes with the scene, so the normalization bounds must update.
- The "good" brightness range (80-180) is a general guideline. High-key photography (intentionally bright) and low-key photography (intentionally dark) are valid artistic choices that will trigger warnings. Context matters.
- When combining the histogram display with the camera feed, resize the histogram canvas to match the frame height for clean stacking.
- For per-channel analysis (B, G, R separately), compute three histograms. But for brightness and contrast monitoring, a single grayscale histogram is both simpler and more meaningful.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Cover the lens or point at a dark area — the dashboard on the right should show "VERY DARK" or "Dark" in red/yellow, with low brightness and high underexposed percentage
- Point at a bright light or window — the status should shift to "VERY BRIGHT" or "Bright" with the highlight clipping percentage rising
- Watch the histogram panel update in real-time as you pan the camera, and verify the brightness gauge needle moves within the green "good zone" markers under normal lighting

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

# --- Histogram display parameters ---
HIST_W = 256
HIST_H = 150

# --- Thresholds ---
SHADOW_THRESH = 5       # Pixels <= this are considered underexposed
HIGHLIGHT_THRESH = 250  # Pixels >= this are considered overexposed
GOOD_BRIGHTNESS_LOW = 80
GOOD_BRIGHTNESS_HIGH = 180


def draw_histogram(gray_img, hist_h=HIST_H):
    """Draw a color-zoned grayscale histogram."""
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_h - 10, cv2.NORM_MINMAX)

    canvas = np.zeros((hist_h, 256, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # Draw zone backgrounds (subtle)
    cv2.rectangle(canvas, (0, 0), (SHADOW_THRESH, hist_h), (30, 20, 15), -1)
    cv2.rectangle(canvas, (HIGHLIGHT_THRESH, 0), (255, hist_h), (15, 20, 30), -1)

    for x in range(256):
        bar_h = int(hist[x][0])
        if bar_h > 0:
            if x <= SHADOW_THRESH:
                color = (200, 120, 60)       # Shadow zone -- blue tone
            elif x >= HIGHLIGHT_THRESH:
                color = (60, 100, 220)       # Highlight zone -- red tone
            else:
                color = (200, 200, 200)      # Midtone -- neutral
            cv2.line(canvas, (x, hist_h), (x, hist_h - bar_h), color, 1)

    # Zone labels
    cv2.putText(canvas, "Shadows", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 100, 60), 1)
    cv2.putText(canvas, "Highlights", (195, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60, 80, 180), 1)

    return canvas


def draw_gauge(value, min_val=0, max_val=255, width=200, height=20,
               good_lo=GOOD_BRIGHTNESS_LOW, good_hi=GOOD_BRIGHTNESS_HIGH):
    """Draw a horizontal gauge with a good-zone indicator."""
    gauge = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background
    for x in range(width):
        v = int(255 * x / width) // 3
        gauge[:, x] = (v, v, v)

    # Good zone markers (green lines at top and bottom)
    lo_x = int(width * good_lo / max_val)
    hi_x = int(width * good_hi / max_val)
    cv2.rectangle(gauge, (lo_x, 0), (hi_x, 2), (0, 200, 0), -1)
    cv2.rectangle(gauge, (lo_x, height - 2), (hi_x, height), (0, 200, 0), -1)

    # Current value needle
    pos = int(width * np.clip(value, min_val, max_val) / max_val)
    pos = max(2, min(pos, width - 3))

    # Needle color based on zone
    if good_lo <= value <= good_hi:
        needle_color = (0, 255, 0)
    elif value < good_lo * 0.6 or value > good_hi + (255 - good_hi) * 0.6:
        needle_color = (0, 0, 255)
    else:
        needle_color = (0, 200, 255)

    cv2.rectangle(gauge, (pos - 2, 0), (pos + 2, height), needle_color, -1)
    cv2.rectangle(gauge, (pos - 2, 0), (pos + 2, height), (255, 255, 255), 1)

    return gauge


def get_exposure_status(mean_b, std_d, under_pct, over_pct):
    """Return (status_text, bgr_color) based on metrics."""
    issues = []

    if mean_b < 50:
        issues.append("VERY DARK")
    elif mean_b < GOOD_BRIGHTNESS_LOW:
        issues.append("Dark")
    elif mean_b > 210:
        issues.append("VERY BRIGHT")
    elif mean_b > GOOD_BRIGHTNESS_HIGH:
        issues.append("Bright")

    if std_d < 20:
        issues.append("LOW CONTRAST")

    if over_pct > 15:
        issues.append("HIGHLIGHT CLIP")
    elif over_pct > 5:
        issues.append("Highlight warn")

    if under_pct > 15:
        issues.append("SHADOW CLIP")
    elif under_pct > 5:
        issues.append("Shadow warn")

    if not issues:
        return "GOOD EXPOSURE", (0, 255, 0)
    elif any(w in ' '.join(issues).upper() for w in ["VERY", "CLIP"]):
        return " | ".join(issues), (0, 0, 255)
    else:
        return " | ".join(issues), (0, 200, 255)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]

        # --- Compute metrics ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))
        dynamic_range = max_val - min_val

        total_pixels = gray.size
        underexposed_pct = np.sum(gray <= SHADOW_THRESH) / total_pixels * 100
        overexposed_pct = np.sum(gray >= HIGHLIGHT_THRESH) / total_pixels * 100

        # --- Get status ---
        status_text, status_color = get_exposure_status(
            mean_brightness, std_dev, underexposed_pct, overexposed_pct
        )

        # --- Build dashboard panel ---
        panel_w = 280
        panel_h = h
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 25

        # Title
        cv2.putText(panel, "Exposure Dashboard", (10, y_offset),
                    font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y_offset += 30

        # Status bar
        cv2.rectangle(panel, (5, y_offset - 15), (panel_w - 5, y_offset + 8), status_color, -1)
        cv2.putText(panel, status_text[:30], (10, y_offset + 2),
                    font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 28

        # Brightness
        cv2.putText(panel, f"Brightness: {mean_brightness:.1f} / 255", (10, y_offset),
                    font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        y_offset += 18
        gauge = draw_gauge(mean_brightness, width=panel_w - 20)
        gauge_h = gauge.shape[0]
        panel[y_offset:y_offset + gauge_h, 10:10 + gauge.shape[1]] = gauge
        y_offset += gauge_h + 15

        # Contrast (std dev)
        contrast_color = (0, 255, 0) if std_dev >= 40 else (0, 200, 255) if std_dev >= 20 else (0, 0, 255)
        cv2.putText(panel, f"Contrast (StdDev): {std_dev:.1f}", (10, y_offset),
                    font, 0.4, contrast_color, 1, cv2.LINE_AA)
        y_offset += 18

        # Contrast bar
        contrast_bar_w = panel_w - 20
        contrast_bar = np.zeros((12, contrast_bar_w, 3), dtype=np.uint8)
        contrast_bar[:] = (50, 50, 50)
        filled = int(np.clip(std_dev / 100, 0, 1) * contrast_bar_w)
        cv2.rectangle(contrast_bar, (0, 0), (filled, 12), contrast_color, -1)
        panel[y_offset:y_offset + 12, 10:10 + contrast_bar_w] = contrast_bar
        y_offset += 25

        # Dynamic range
        cv2.putText(panel, f"Range: [{min_val}, {max_val}]  DR: {dynamic_range}",
                    (10, y_offset), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y_offset += 25

        # Clipping percentages
        shadow_color = (0, 0, 255) if underexposed_pct > 5 else (0, 200, 255) if underexposed_pct > 1 else (0, 255, 0)
        highlight_color = (0, 0, 255) if overexposed_pct > 5 else (0, 200, 255) if overexposed_pct > 1 else (0, 255, 0)

        cv2.putText(panel, f"Underexposed: {underexposed_pct:.1f}%", (10, y_offset),
                    font, 0.4, shadow_color, 1, cv2.LINE_AA)
        y_offset += 18
        cv2.putText(panel, f"Overexposed:  {overexposed_pct:.1f}%", (10, y_offset),
                    font, 0.4, highlight_color, 1, cv2.LINE_AA)
        y_offset += 28

        # Histogram
        cv2.putText(panel, "Histogram:", (10, y_offset),
                    font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += 8
        hist_canvas = draw_histogram(gray, hist_h=HIST_H)
        # Center histogram in panel
        hx = (panel_w - 256) // 2
        if hx < 0:
            # Resize histogram to fit
            hist_canvas = cv2.resize(hist_canvas, (panel_w - 10, HIST_H))
            hx = 5
        if y_offset + HIST_H < panel_h:
            panel[y_offset:y_offset + HIST_H, hx:hx + hist_canvas.shape[1]] = hist_canvas
        y_offset += HIST_H + 15

        # Mean marker on histogram
        mean_x = int(mean_brightness) + hx
        if y_offset - HIST_H - 15 + HIST_H < panel_h:
            marker_y = y_offset - 15
            cv2.putText(panel, f"mean={mean_brightness:.0f}", (hx + 5, marker_y),
                        font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Draw FPS on camera frame ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 15),
                    font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # --- Combine frame and dashboard ---
        display = np.hstack([frame, panel])

        cv2.imshow('Live Brightness & Contrast Monitor', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
```
