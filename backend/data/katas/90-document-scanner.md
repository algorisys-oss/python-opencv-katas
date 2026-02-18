---
slug: 90-document-scanner
title: Document Scanner
level: advanced
concepts: [edge detection, contour detection, perspective transform, thresholding]
prerequisites: [32-canny-edge-detection, 38-finding-contours, 58-perspective-transform]
---

## What Problem Are We Solving?

You have a photo of a document taken at an angle -- perhaps a receipt on a table, a whiteboard from across a room, or a form held by hand. The document appears skewed and distorted by perspective. A **document scanner pipeline** takes that messy photo and produces a clean, top-down, rectangular view of the document, much like a flatbed scanner would produce.

This pipeline chains together multiple OpenCV techniques: edge detection to find boundaries, contour detection to locate the document's four corners, a perspective transform to "unwarp" the document, and adaptive thresholding to produce a clean black-and-white result.

## Step 1: Create a Synthetic Scene

To test our scanner without needing a real photo, we create a synthetic image -- a white "document" with text-like content drawn onto a colored background, rotated at an angle to simulate a photo taken from a perspective:

```python
# Create a blank document with some content
doc = np.ones((400, 300, 3), dtype=np.uint8) * 255
cv2.putText(doc, 'INVOICE', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
cv2.line(doc, (20, 70), (280, 70), (0, 0, 0), 2)
```

We then define four source corners (the document corners) and four destination corners (where they'll land on the scene) and use `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to place the document at an angle on the background.

## Step 2: Grayscale, Blur, and Canny Edge Detection

The first processing step converts the scene to grayscale and applies a Gaussian blur to reduce noise, followed by Canny edge detection to find all edges:

```python
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
```

The blur is critical -- without it, texture and noise produce too many edges, making it hard to find the document boundary.

## Step 3: Find the Largest 4-Point Contour

With edges detected, we find contours and look for the largest one that can be approximated as a quadrilateral. `cv2.approxPolyDP` simplifies a contour to fewer points based on an epsilon tolerance:

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc_contour = approx
        break
```

The epsilon `0.02 * perimeter` is a common heuristic -- it allows enough simplification to collapse nearly-straight segments while preserving actual corners.

## Step 4: Perspective Warp (The "Scan")

Once we have four corners, we need to order them consistently (top-left, top-right, bottom-right, bottom-left) and compute a perspective transform to map them to a rectangle:

```python
pts = doc_contour.reshape(4, 2).astype(np.float32)
# Order points: top-left, top-right, bottom-right, bottom-left
s = pts.sum(axis=1)
ordered = np.zeros((4, 2), dtype=np.float32)
ordered[0] = pts[np.argmin(s)]   # top-left has smallest x+y
ordered[2] = pts[np.argmax(s)]   # bottom-right has largest x+y
d = np.diff(pts, axis=1)
ordered[1] = pts[np.argmin(d)]   # top-right has smallest y-x
ordered[3] = pts[np.argmax(d)]   # bottom-left has largest y-x
```

With ordered points, `cv2.getPerspectiveTransform` maps them to a destination rectangle, and `cv2.warpPerspective` applies the transform.

## Step 5: Adaptive Threshold for Clean Output

Finally, we convert the warped document to grayscale and apply adaptive thresholding to get a clean binary result:

```python
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
scanned = cv2.adaptiveThreshold(warped_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
```

Adaptive thresholding works better than global thresholding here because lighting across a document is rarely uniform.

## The Complete Pipeline

1. **Input**: Scene image containing a document at an angle
2. **Grayscale + Blur**: Remove color info and reduce noise
3. **Canny**: Detect edges in the scene
4. **Find Contours**: Locate all edge boundaries
5. **Approximate to Quad**: Find the largest 4-sided contour (the document)
6. **Order Corners**: Sort the 4 corners consistently
7. **Perspective Warp**: Transform the skewed quad into a flat rectangle
8. **Adaptive Threshold**: Produce a clean, scanner-like output

## Tips & Common Mistakes

- The `0.02 * perimeter` epsilon for `cv2.approxPolyDP` is a starting point. If the document has rounded corners or the edges are noisy, you may need to increase it slightly.
- Always sort contours by area (largest first) -- the document should be the largest quadrilateral in the scene.
- Corner ordering matters. If you get the corners wrong, the warp will produce a flipped or rotated result. The sum/difference method (sum of coordinates for TL/BR, difference for TR/BL) is robust.
- Gaussian blur kernel size affects edge detection quality. Too small and noise survives; too large and thin document borders vanish.
- If your document has a dark border on a dark background, Canny may not find edges. Ensure contrast between the document and the background.
- Adaptive thresholding block size must be odd. Start with 11 and adjust based on text size.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create a synthetic document on an angled background
# =============================================================
# Create a "document" with text-like content
doc_h, doc_w = 400, 300
document = np.ones((doc_h, doc_w, 3), dtype=np.uint8) * 245

# Add fake document content
cv2.putText(document, 'INVOICE #1042', (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
cv2.line(document, (20, 60), (280, 60), (0, 0, 0), 2)
for i in range(8):
    y = 90 + i * 35
    cv2.line(document, (20, y), (270, y), (180, 180, 180), 1)
    cv2.putText(document, f'Item {i+1} ......... ${ (i+1)*12.50:.2f}',
                (25, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
cv2.putText(document, 'TOTAL: $450.00', (20, 380),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

# Create a scene (colored background) and place the document at an angle
scene_h, scene_w = 500, 600
scene = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)
scene[:] = (60, 40, 30)  # Dark wood-like background

# Define where document corners land on the scene (simulating perspective)
src_pts = np.array([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]], dtype=np.float32)
dst_pts = np.array([[120, 50], [430, 80], [460, 430], [80, 390]], dtype=np.float32)

M_place = cv2.getPerspectiveTransform(src_pts, dst_pts)
scene = cv2.warpPerspective(document, M_place, (scene_w, scene_h),
                            dst=scene, borderMode=cv2.BORDER_TRANSPARENT)

# =============================================================
# Step 2: Grayscale, blur, and Canny edge detection
# =============================================================
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Dilate edges to close small gaps
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

# =============================================================
# Step 3: Find the largest 4-point contour
# =============================================================
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc_contour = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        doc_contour = approx
        break

# Draw detected contour on a copy of the scene
scene_contour = scene.copy()
if doc_contour is not None:
    cv2.drawContours(scene_contour, [doc_contour], -1, (0, 255, 0), 3)
    for pt in doc_contour:
        cv2.circle(scene_contour, tuple(pt[0]), 8, (0, 0, 255), -1)
    print(f'Document contour found with 4 corners')
    print(f'Corners: {doc_contour.reshape(4,2).tolist()}')
else:
    print('No 4-point contour found!')

# =============================================================
# Step 4: Order corners and apply perspective warp
# =============================================================
if doc_contour is not None:
    pts = doc_contour.reshape(4, 2).astype(np.float32)

    # Order: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    # Compute output dimensions
    w1 = np.linalg.norm(ordered[1] - ordered[0])
    w2 = np.linalg.norm(ordered[2] - ordered[3])
    out_w = int(max(w1, w2))

    h1 = np.linalg.norm(ordered[3] - ordered[0])
    h2 = np.linalg.norm(ordered[2] - ordered[1])
    out_h = int(max(h1, h2))

    target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M_warp = cv2.getPerspectiveTransform(ordered, target)
    warped = cv2.warpPerspective(scene, M_warp, (out_w, out_h))

    # =========================================================
    # Step 5: Adaptive threshold for clean scanned look
    # =========================================================
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(warped_gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 8)

    # Convert scanned to BGR for display
    scanned_bgr = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)

    # Resize all images to same height for side-by-side display
    display_h = 400
    def resize_to_height(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    scene_disp = resize_to_height(scene_contour, display_h)
    warped_disp = resize_to_height(warped, display_h)
    scanned_disp = resize_to_height(scanned_bgr, display_h)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(scene_disp, 'Detected', (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(warped_disp, 'Warped', (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(scanned_disp, 'Scanned', (10, 25), font, 0.6, (0, 0, 0), 2)

    result = np.hstack([scene_disp, warped_disp, scanned_disp])
    print(f'Output document size: {out_w}x{out_h}')
else:
    result = scene
    print('Pipeline failed: could not detect document contour')

cv2.imshow('Document Scanner', result)
```
