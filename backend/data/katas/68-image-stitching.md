---
slug: 68-image-stitching
title: Image Stitching Basics
level: advanced
concepts: [feature matching, homography, panorama blending]
prerequisites: [67-homography-warping]
---

## What Problem Are We Solving?

When you take multiple photos of a scene by rotating the camera, each photo captures a different portion of the view. **Image stitching** combines these overlapping images into a single seamless **panorama**. This involves detecting features in each image, matching them across the overlap region, computing a homography to align the images, warping one image into the other's coordinate system, and blending the result to hide the seam.

## The Stitching Pipeline

The complete stitching process follows these steps:

```
1. Detect features    → Find keypoints + descriptors in both images
2. Match features     → Find corresponding features in the overlap region
3. Compute homography → Estimate the geometric transform (with RANSAC)
4. Warp image         → Transform one image into the other's coordinate system
5. Blend              → Combine the warped and reference images seamlessly
```

Each step builds on the previous one, and errors compound — so robust feature matching and RANSAC are critical.

## Step 1: Detect Features

Use a feature detector that produces descriptors for matching. SIFT gives the best accuracy; ORB is faster:

```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```

## Step 2: Match Features

Match descriptors between the two images and filter using the ratio test:

```python
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
```

## Step 3: Compute Homography

Extract the matched point coordinates and compute a homography with RANSAC:

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

This homography H maps points from image 1 into image 2's coordinate system.

## Step 4: Warp Image

To stitch the images, warp image 1 into a larger canvas that can hold both images, then place image 2 in its position:

```python
# Determine the output canvas size
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Find where img1's corners land after warping
corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
warped_corners = cv2.perspectiveTransform(corners_img1, H)

# Combine with img2's corners to find total canvas bounds
all_corners = np.concatenate([warped_corners, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)])
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

# Translation to shift everything into positive coordinates
translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

# Warp img1 onto the canvas
canvas_size = (x_max - x_min, y_max - y_min)
warped1 = cv2.warpPerspective(img1, translation @ H, canvas_size)
```

## Step 5: Blend

The simplest blending approach is to place image 2 directly onto the canvas where it belongs:

```python
# Place img2 at its position on the canvas
warped1[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2
```

For better results, use **alpha blending** in the overlap region:

```python
# Create masks for each image on the canvas
mask1 = np.zeros(warped1.shape[:2], dtype=np.float32)
mask1[warped1.sum(axis=2) > 0] = 1.0

canvas2 = np.zeros_like(warped1)
canvas2[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2
mask2 = np.zeros(canvas2.shape[:2], dtype=np.float32)
mask2[canvas2.sum(axis=2) > 0] = 1.0

# Weighted blend in overlap region
overlap = (mask1 > 0) & (mask2 > 0)
total = mask1 + mask2
total[total == 0] = 1  # avoid division by zero
blend = (warped1 * (mask1 / total)[..., None] + canvas2 * (mask2 / total)[..., None]).astype(np.uint8)
```

## Tips & Common Mistakes

- The homography assumes a **planar scene** or pure camera rotation. Stitching images with significant parallax (close objects at different depths) will produce ghosting artifacts.
- You need **sufficient overlap** between images — typically at least 30-50% overlap for reliable matching.
- Always use RANSAC when computing the homography from feature matches. Without it, a few bad matches can completely ruin the alignment.
- The canvas size calculation is critical — if you make it too small, the warped image gets clipped. Use `cv2.perspectiveTransform` on the image corners to find the required bounds.
- Simple overwrite blending creates a visible seam. Alpha blending or feathering in the overlap region produces much better results.
- The homography direction matters: `H` maps points from image 1 to image 2. Make sure you are warping the correct image with the correct matrix.
- For more than two images, you need to chain homographies. Warp all images into a common reference frame (usually the center image).
- OpenCV provides `cv2.Stitcher` as a high-level API that handles all these steps automatically, but understanding the manual pipeline is essential for custom stitching needs.

## Starter Code

```python
import cv2
import numpy as np

# Create a wide scene, then split into two overlapping images
scene = np.zeros((300, 700, 3), dtype=np.uint8)
scene[:] = (60, 60, 60)

# Draw features across the scene
cv2.rectangle(scene, (20, 30), (120, 130), (255, 255, 255), 2)
cv2.circle(scene, (200, 80), 40, (200, 200, 255), 2)
cv2.putText(scene, 'LEFT', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
cv2.rectangle(scene, (250, 50), (380, 170), (100, 200, 100), -1)
cv2.polylines(scene, [np.array([[300, 60], [370, 60], [370, 160], [300, 160]])], True, (200, 255, 200), 2)
cv2.circle(scene, (450, 150), 50, (255, 180, 100), 2)
cv2.putText(scene, 'RIGHT', (420, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
cv2.rectangle(scene, (530, 30), (670, 150), (200, 150, 255), 2)
cv2.line(scene, (530, 30), (670, 150), (200, 150, 255), 1)
cv2.line(scene, (670, 30), (530, 150), (200, 150, 255), 1)

# Add random texture for feature detection
rng = np.random.RandomState(42)
for _ in range(60):
    x, y = rng.randint(0, 700), rng.randint(0, 300)
    color = tuple(int(c) for c in rng.randint(80, 255, 3))
    cv2.circle(scene, (x, y), rng.randint(2, 6), color, -1)

# Draw lines across overlap region for visual reference
for y in range(250, 290, 10):
    cv2.line(scene, (0, y), (700, y), (80, 80, 80), 1)

# Split into two overlapping images (150px overlap)
img1 = scene[:, :420].copy()   # Left portion
img2 = scene[:, 270:].copy()   # Right portion (overlap: 270-420)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- Step 1 & 2: Detect and match features ---
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

# --- Step 3: Compute homography ---
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
else:
    H = None
    inliers = 0

# --- Step 4: Warp and stitch ---
if H is not None:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find canvas bounds
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners1, H)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([warped_corners, corners2])

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Warp img1 onto canvas
    panorama = cv2.warpPerspective(img1, translation @ H, (canvas_w, canvas_h))

    # --- Step 5: Blend ---
    # Create mask for warped img1
    mask_warp = (panorama.sum(axis=2) > 0).astype(np.float32)

    # Place img2 on canvas
    canvas2 = np.zeros_like(panorama)
    y_off = -y_min
    x_off = -x_min
    canvas2[y_off:y_off + h2, x_off:x_off + w2] = img2
    mask2 = (canvas2.sum(axis=2) > 0).astype(np.float32)

    # Alpha blend in overlap
    total_mask = mask_warp + mask2
    total_mask[total_mask == 0] = 1.0
    panorama = (panorama * (mask_warp / total_mask)[..., None] +
                canvas2 * (mask2 / total_mask)[..., None]).astype(np.uint8)
else:
    panorama = np.hstack([img1, img2])

# --- Draw feature matches ---
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good[:30], None,
                              matchColor=(0, 255, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# --- Build final display ---
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1, 'Image 1', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img2, 'Image 2', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Resize panorama to fit display
pano_display = cv2.resize(panorama, (700, 300))
cv2.putText(pano_display, f'Panorama ({inliers} inliers)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Stack: source images on top, matches in middle, panorama at bottom
sources = np.hstack([img1, np.zeros((300, 700 - 420 - 430, 3), dtype=np.uint8) if 420 + 430 < 700 else np.empty((300, 0, 3), dtype=np.uint8), img2])
if sources.shape[1] < 700:
    pad = np.zeros((300, 700 - sources.shape[1], 3), dtype=np.uint8)
    sources = np.hstack([sources, pad])
elif sources.shape[1] > 700:
    sources = cv2.resize(sources, (700, 300))

matches_resized = cv2.resize(img_matches, (700, 200))
result = np.vstack([sources, matches_resized, pano_display])

print(f'Features: img1={len(kp1)}, img2={len(kp2)}')
print(f'Good matches: {len(good)}')
print(f'RANSAC inliers: {inliers}')
print(f'Panorama size: {panorama.shape[1]}x{panorama.shape[0]}')

cv2.imshow('Image Stitching Basics', result)
```
