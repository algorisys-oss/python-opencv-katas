---
slug: 83-qrcode-detection
title: QR Code & Barcode Detection
level: advanced
concepts: [cv2.QRCodeDetector, decode, detect]
prerequisites: [38-finding-contours]
---

## What Problem Are We Solving?

QR codes are everywhere — product labels, tickets, URLs, payment systems. Being able to **detect and decode** QR codes in images is a practical computer vision skill. OpenCV provides a built-in `cv2.QRCodeDetector` that can locate QR codes in an image and decode their content, all without external libraries or model files.

## The QRCodeDetector API

OpenCV's QR code detector provides three main operations:

```python
detector = cv2.QRCodeDetector()

# Detect only (find bounding polygon, don't decode)
found, points = detector.detect(img)

# Decode only (given known location)
data, points, straight_qr = detector.decode(img, points)

# Detect + Decode in one step
data, points, straight_qr = detector.detectAndDecode(img)
```

## Detect: Finding the QR Code Location

The `detect()` method finds the QR code's position and returns its corner points:

```python
detector = cv2.QRCodeDetector()
found, points = detector.detect(gray_or_color_img)
# found: bool — whether a QR code was found
# points: numpy array of shape (1, 4, 2) — four corner coordinates
```

The four points define the **bounding quadrilateral** (not necessarily axis-aligned) of the QR code, in order: top-left, top-right, bottom-right, bottom-left.

## Decode: Reading the Data

The `decode()` method reads the actual content from a detected QR code:

```python
data, points, straight_qr = detector.decode(img, points)
# data: str — the decoded text content
# points: corner points (same as input or refined)
# straight_qr: the rectified QR code image
```

## DetectAndDecode: The All-in-One Method

Most commonly, you want both operations at once:

```python
detector = cv2.QRCodeDetector()
data, points, straight_qr = detector.detectAndDecode(img)

if data:
    print(f'QR Code content: {data}')
    # points shape: (1, 4, 2) — four corners
    pts = points[0].astype(int)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i+1) % 4]), (0, 255, 0), 3)
```

## Drawing the Bounding Polygon

The detected QR code boundary is a quadrilateral. Draw it with `cv2.polylines()`:

```python
if points is not None:
    pts = points[0].astype(int)
    # Draw as a closed polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    # Or draw individual corners
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
```

## Detecting Multiple QR Codes

OpenCV 4.x also provides `detectMulti()` and `decodeMulti()` for handling images with multiple QR codes:

```python
detector = cv2.QRCodeDetector()

# Detect multiple QR codes
found, points = detector.detectMulti(img)

if found:
    # Decode all detected codes
    decoded, points, straight_codes = detector.decodeMulti(img, points)
    for i, data in enumerate(decoded):
        if data:
            print(f'QR {i+1}: {data}')
```

## Preprocessing for Better Detection

QR code detection works best on sharp, high-contrast images:

```python
# Preprocessing pipeline for difficult QR codes:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sharpen
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp = cv2.filter2D(gray, -1, kernel)

# Adaptive threshold for uneven lighting
binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 10)
```

## Tips & Common Mistakes

- `detectAndDecode()` works on both grayscale and color images. Grayscale may be slightly faster.
- The `points` array has shape `(1, 4, 2)`. Access the corners with `points[0]` to get shape `(4, 2)`.
- If `data` is an empty string, the QR code was detected but could not be decoded (too blurry, damaged, etc.).
- For very small QR codes, resize the image up before detection.
- QR codes at extreme angles may not be detected. The detector handles moderate perspective distortion.
- The `straight_qr` output is the rectified (flattened) QR code image — useful for debugging.
- `detectMulti` / `decodeMulti` require OpenCV 4.3+. Check your version with `cv2.__version__`.
- In a sandboxed environment, you can create a synthetic QR-like pattern to demonstrate the API. Real QR codes require specific encoding patterns.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a synthetic QR-code-like pattern for demonstration ---
canvas = np.ones((500, 700, 3), dtype=np.uint8) * 240

def draw_qr_like_pattern(img, x, y, size, label):
    """Draw a simplified QR-code-like pattern with finder patterns."""
    # White background for the QR area
    cv2.rectangle(img, (x, y), (x + size, y + size), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 2)

    cell = size // 25  # Module size
    if cell < 1:
        cell = 1

    # Draw finder patterns (the three large squares in QR corners)
    def draw_finder(fx, fy):
        s = cell * 7
        cv2.rectangle(img, (fx, fy), (fx + s, fy + s), (0, 0, 0), -1)
        cv2.rectangle(img, (fx + cell, fy + cell),
                      (fx + s - cell, fy + s - cell), (255, 255, 255), -1)
        cv2.rectangle(img, (fx + cell * 2, fy + cell * 2),
                      (fx + s - cell * 2, fy + s - cell * 2), (0, 0, 0), -1)

    # Top-left, top-right, bottom-left finder patterns
    draw_finder(x + cell, y + cell)
    draw_finder(x + size - cell * 8, y + cell)
    draw_finder(x + cell, y + size - cell * 8)

    # Add some random data modules in the center area
    np.random.seed(42)
    data_start = cell * 9
    data_end = size - cell * 9
    for row in range(data_start, data_end, cell):
        for col in range(data_start, data_end, cell):
            if np.random.random() > 0.5:
                cv2.rectangle(img, (x + col, y + row),
                              (x + col + cell, y + row + cell), (0, 0, 0), -1)

    # Timing patterns (alternating black/white between finders)
    for i in range(cell * 8, size - cell * 8, cell * 2):
        cv2.rectangle(img, (x + i, y + cell * 6),
                      (x + i + cell, y + cell * 7), (0, 0, 0), -1)
        cv2.rectangle(img, (x + cell * 6, y + i),
                      (x + cell * 7, y + i + cell), (0, 0, 0), -1)

    cv2.putText(img, label, (x, y + size + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    return np.array([[x, y], [x + size, y], [x + size, y + size], [x, y + size]])

# Draw two QR-like patterns
corners1 = draw_qr_like_pattern(canvas, 50, 80, 200, 'QR Code 1')
corners2 = draw_qr_like_pattern(canvas, 350, 120, 150, 'QR Code 2')

# --- Demonstrate the QRCodeDetector API ---
detector = cv2.QRCodeDetector()

# Try detection on our synthetic image
# Note: synthetic patterns may not decode since they don't follow QR encoding rules
data, points, straight_qr = detector.detectAndDecode(canvas)

result = canvas.copy()

if data:
    print(f'Decoded QR content: {data}')
    pts = points[0].astype(int)
    cv2.polylines(result, [pts], True, (0, 255, 0), 3)
else:
    print('No valid QR code decoded (expected with synthetic patterns)')
    print('Demonstrating with simulated detection results...')

    # Simulate what detection results look like
    for i, (corners, label) in enumerate([(corners1, 'QR 1'), (corners2, 'QR 2')]):
        pts = corners.astype(int)
        # Draw bounding polygon
        cv2.polylines(result, [pts], True, (0, 255, 0), 3)
        # Draw corner points
        for j, pt in enumerate(pts):
            color = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)][j]
            cv2.circle(result, tuple(pt), 6, color, -1)
        # Label
        cx = pts[:, 0].mean()
        cy = pts[:, 1].min() - 10
        cv2.putText(result, f'{label}: "https://example.com"',
                    (int(cx) - 60, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1, cv2.LINE_AA)

# --- Info panel ---
info = np.zeros((170, 700, 3), dtype=np.uint8)
info[:] = (40, 40, 40)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(info, 'QR Code Detection API:', (10, 25),
            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'detector = cv2.QRCodeDetector()', (20, 50),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'data, points, qr = detector.detectAndDecode(img)', (20, 73),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'found, points = detector.detect(img)       # detect only', (20, 96),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
cv2.putText(info, 'found, pts = detector.detectMulti(img)     # multiple QR codes', (20, 119),
            font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

# Corner legend
cv2.circle(info, (500, 20), 5, (0, 0, 255), -1)
cv2.putText(info, 'Top-Left', (510, 25), font, 0.35, (200, 200, 200), 1)
cv2.circle(info, (500, 38), 5, (0, 165, 255), -1)
cv2.putText(info, 'Top-Right', (510, 43), font, 0.35, (200, 200, 200), 1)
cv2.circle(info, (500, 56), 5, (0, 255, 255), -1)
cv2.putText(info, 'Bottom-Right', (510, 61), font, 0.35, (200, 200, 200), 1)
cv2.circle(info, (500, 74), 5, (0, 255, 0), -1)
cv2.putText(info, 'Bottom-Left', (510, 79), font, 0.35, (200, 200, 200), 1)

cv2.putText(info, 'Points shape: (1, 4, 2) -> 4 corner coordinates', (20, 145),
            font, 0.42, (150, 150, 255), 1, cv2.LINE_AA)
cv2.putText(info, 'Returns empty string if detected but cannot decode', (20, 165),
            font, 0.42, (150, 150, 255), 1, cv2.LINE_AA)

cv2.putText(result, 'QR Code Detection', (10, 30),
            font, 0.7, (0, 100, 200), 2, cv2.LINE_AA)

output = np.vstack([result, info])

print('\nQRCodeDetector API:')
print('  detector = cv2.QRCodeDetector()')
print('  data, points, straight_qr = detector.detectAndDecode(img)')
print('  points shape: (1, 4, 2) - four corner coordinates')

cv2.imshow('QR Code Detection', output)
```
