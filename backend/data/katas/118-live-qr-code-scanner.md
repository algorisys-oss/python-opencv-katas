---
slug: 118-live-qr-code-scanner
title: Live QR Code Scanner
level: live
concepts: [cv2.QRCodeDetector, detectAndDecode, bounding polygon, QR code scanning]
prerequisites: [100-live-camera-fps, 83-qrcode-detection]
---

## What Problem Are We Solving?

QR codes are embedded in every corner of modern life -- restaurant menus, boarding passes, payment terminals, Wi-Fi setup cards, product packaging. A live QR code scanner that reads codes in real-time from a webcam turns your computer into a versatile scanning station. Instead of pulling out your phone, you simply hold up a QR code to the camera and instantly see the decoded content.

Building this live scanner also teaches a pattern that generalizes to many real-time detection tasks: detect an object in each frame, extract information from it, draw visual feedback showing what was found, and maintain a history of results. The QR code detector is a clean, self-contained example because OpenCV provides a built-in detector that requires no external models or libraries.

The challenge in real-time scanning is not just detecting a single QR code once -- it is handling the continuous stream of frames where codes appear, disappear, move, and sometimes fail to decode due to motion blur or partial occlusion. A robust scanner highlights detected codes with a bounding polygon, displays the decoded text, and keeps a scrolling history of all codes scanned during the session.

## The QRCodeDetector API

OpenCV provides `cv2.QRCodeDetector`, a class with three main methods that separate or combine the detection and decoding steps:

```python
detector = cv2.QRCodeDetector()

# 1. Detect only -- find the QR code location, don't decode
found, points = detector.detect(img)

# 2. Decode only -- given known corner points, decode the content
data, points, straight_qr = detector.decode(img, points)

# 3. Detect + Decode in one call -- the most common approach
data, points, straight_qr = detector.detectAndDecode(img)
```

### Method Comparison

| Method | What It Does | Returns | When to Use |
|---|---|---|---|
| `detect(img)` | Locates the QR code boundary | `(found_bool, points)` | When you only need the position, not the content |
| `decode(img, points)` | Decodes content from known location | `(data_str, points, straight_qr)` | When you already know where the QR code is |
| `detectAndDecode(img)` | Finds and decodes in one step | `(data_str, points, straight_qr)` | Most common -- use this by default |

### Understanding the Return Values

```python
data, points, straight_qr = detector.detectAndDecode(frame)
```

| Return Value | Type | Shape | Meaning |
|---|---|---|---|
| `data` | `str` | -- | Decoded text content; empty string if detected but not decoded |
| `points` | `ndarray` or `None` | `(1, 4, 2)` | Four corner coordinates of the QR bounding quadrilateral |
| `straight_qr` | `ndarray` or `None` | Variable | The rectified (flattened) QR code image |

The `points` array contains the four corners in order: top-left, top-right, bottom-right, bottom-left. Access them with `points[0]` to get a `(4, 2)` array of (x, y) coordinates.

## Detect vs Decode: When They Differ

Detection and decoding are separate operations that can succeed or fail independently:

- **Detected but not decoded**: The QR code's three finder patterns (the large squares in three corners) are visible, so OpenCV finds the boundary. But the data modules are too blurry, partially occluded, or at too steep an angle to read. In this case, `data` is an empty string but `points` is not `None`.

- **Neither detected nor decoded**: The QR code is too small, too far away, or not in frame. Both `data` is empty and `points` is `None`.

```python
data, points, straight_qr = detector.detectAndDecode(frame)

if points is not None and data:
    # Fully detected and decoded -- show green boundary + text
    status = "DECODED"
elif points is not None:
    # Detected but decode failed -- show yellow boundary
    status = "DETECTED (cannot decode)"
else:
    # Nothing found
    status = "No QR code"
```

This distinction is useful for user feedback: showing a yellow boundary when the code is found but unreadable tells the user to hold the code steadier or move closer.

## Handling Multiple QR Codes

OpenCV 4.3+ provides `detectMulti()` and `decodeMulti()` for frames containing more than one QR code:

```python
detector = cv2.QRCodeDetector()

# Detect all QR codes in the frame
found, points = detector.detectMulti(frame)

if found:
    # points shape: (N, 4, 2) -- N detected codes, each with 4 corners
    decoded_list, points, straight_codes = detector.decodeMulti(frame, points)

    for i, data in enumerate(decoded_list):
        if data:
            print(f"QR {i+1}: {data}")
        pts = points[i].astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
```

| Method | Return Shape for Points | Notes |
|---|---|---|
| `detect` | `(1, 4, 2)` | Single QR code |
| `detectMulti` | `(N, 4, 2)` | N QR codes |
| `decodeMulti` | List of N strings | One string per detected code |

> **Version check:** `detectMulti` and `decodeMulti` require OpenCV 4.3 or later. Check with `cv2.__version__`.

## Drawing the Bounding Polygon

The detected QR code boundary is a quadrilateral (not necessarily axis-aligned). Draw it with `cv2.polylines`:

```python
if points is not None:
    pts = points[0].astype(int)

    # Draw closed polygon
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    # Optionally draw corner circles
    for i, pt in enumerate(pts):
        cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
```

For a more polished look, draw a semi-transparent filled polygon behind the code:

```python
overlay = frame.copy()
cv2.fillPoly(overlay, [pts], (0, 255, 0))
cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
```

## Maintaining a Decoded History

In a live scanner, you want to track all QR codes seen during the session and display them in a scrollable list. A deque with deduplication prevents repeated entries from the same code being held up for many frames:

```python
from collections import deque

history = deque(maxlen=10)  # Last 10 unique codes
seen_set = set()            # For deduplication

# When a new code is decoded:
if data and data not in seen_set:
    seen_set.add(data)
    history.appendleft(data)
```

Display the history in a corner of the frame:

```python
for i, entry in enumerate(history):
    y_pos = 60 + i * 25
    # Truncate long URLs for display
    display_text = entry[:40] + "..." if len(entry) > 40 else entry
    cv2.putText(frame, display_text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
```

## Preprocessing for Difficult QR Codes

When QR codes are hard to read (low contrast, uneven lighting, blurry), preprocessing can help:

```python
# Convert to grayscale and sharpen
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(gray, -1, kernel)

# Try detection on the sharpened grayscale
data, points, straight_qr = detector.detectAndDecode(sharpened)
```

The detector works on both grayscale and color images. Grayscale is slightly faster and sharpening can improve decode success rates for distant or blurry codes.

## Tips & Common Mistakes

- `detectAndDecode` works on both grayscale and color images. Grayscale may be slightly faster, but color works fine for most webcam resolutions.
- The `points` array has shape `(1, 4, 2)` for a single detection. Always access the corners with `points[0]` to get the `(4, 2)` array.
- If `data` is an empty string but `points` is not `None`, the QR code was found but could not be decoded. Show a "detected but unreadable" indicator to guide the user.
- Very small QR codes (under ~50 pixels wide) often fail to detect. If scanning small codes, reduce the camera-to-code distance or increase camera resolution.
- QR codes at extreme perspective angles (>45 degrees) may not be detected. The built-in detector handles moderate perspective distortion but not extreme skew.
- The `straight_qr` output is the rectified QR code image -- useful for debugging decode failures.
- Create the `QRCodeDetector()` once outside the loop, not inside it. The constructor is lightweight but there is no reason to recreate it every frame.
- For multi-code scanning, remember that `decodeMulti` may return empty strings for codes it detected but could not decode. Always check each entry.
- Deduplication by content prevents the history list from filling up when the same code stays in frame for many seconds.
- Motion blur is the biggest enemy of QR decoding. If the camera or the code is moving fast, the decode step often fails even when detection succeeds.

## How to Test This Kata

> **This is a live camera kata.** Click **"Run on Desktop"** in the Code tab — an OpenCV window will open on your desktop using your real webcam. Press **q** in the OpenCV window to quit.

- Hold a QR code (from your phone or a printout) in front of the camera — a green polygon and decoded text should appear when successfully scanned
- If the code is detected but too blurry to decode, you should see a yellow outline with a "DETECTED (move closer)" message
- Check the scrolling history panel on the left — each unique QR code is listed with a number prefix
- Press **r** to clear the scan history and start fresh

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

# --- QR Code detector (create once, reuse every frame) ---
detector = cv2.QRCodeDetector()

# --- FPS tracking ---
frame_times = deque(maxlen=30)

# --- Decoded history ---
history = deque(maxlen=10)
seen_set = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_times.append(time.time())
        fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        h, w = frame.shape[:2]

        # --- Detect and decode QR codes ---
        data, points, straight_qr = detector.detectAndDecode(frame)

        status = "Scanning..."
        status_color = (150, 150, 150)

        if points is not None:
            pts = points[0].astype(int)

            if data:
                # Fully decoded -- green highlight
                # Semi-transparent fill
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

                # Solid border
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

                # Corner dots
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 5, (0, 200, 0), -1)

                # Display decoded text above the QR code
                text_x = int(pts[:, 0].min())
                text_y = int(pts[:, 1].min()) - 15
                display_data = data[:50] + "..." if len(data) > 50 else data
                cv2.putText(frame, display_data, (max(5, text_x), max(20, text_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                status = "QR DECODED"
                status_color = (0, 255, 0)

                # Add to history (deduplicate)
                if data not in seen_set:
                    seen_set.add(data)
                    history.appendleft(data)
            else:
                # Detected but not decoded -- yellow highlight
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 4, (0, 200, 255), -1)

                status = "DETECTED (move closer)"
                status_color = (0, 255, 255)

        # --- Draw decoded history panel ---
        panel_x = 10
        panel_y_start = 80
        if history:
            cv2.putText(frame, "History:", (panel_x, panel_y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            for i, entry in enumerate(history):
                y_pos = panel_y_start + 18 + i * 22
                if y_pos > h - 40:
                    break
                truncated = entry[:45] + "..." if len(entry) > 45 else entry
                # Number prefix
                prefix = f"{i+1}."
                cv2.putText(frame, prefix, (panel_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, truncated, (panel_x + 20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Draw crosshair guide in center ---
        cx, cy = w // 2, h // 2
        cross_size = 20
        cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (100, 100, 100), 1)
        cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (100, 100, 100), 1)

        # --- Overlays ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Codes found: {len(history)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset history", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Live QR Code Scanner', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            history.clear()
            seen_set.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")
    if history:
        print(f"\nScanned {len(history)} QR code(s):")
        for i, entry in enumerate(history):
            print(f"  {i+1}. {entry}")
```
