---
slug: 67-homography-warping
title: Homography & Warping
level: advanced
concepts: [cv2.findHomography, RANSAC, perspective correction]
prerequisites: [65-brute-force-matching, 58-perspective-transform]
---

## What Problem Are We Solving?

When you match features between two images of the same scene taken from different viewpoints, the matched points define a geometric relationship between the images. A **homography** is a 3x3 transformation matrix that maps points from one image plane to another, capturing the perspective change between two views of a planar surface. This enables **perspective correction** (straightening a tilted document), **image alignment**, and is a key building block for panoramic stitching.

## The Homography Matrix

A homography H is a 3x3 matrix that transforms points from one image to another:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]
```

The actual transformed coordinates are `(x'/w', y'/w')`. This projective transformation can represent any combination of rotation, translation, scaling, shearing, and perspective distortion — as long as the scene is approximately **planar** (or the camera only rotates).

## Computing Homography from Matched Points

Given matched feature points from two images, `cv2.findHomography()` computes the best-fit homography:

```python
H, mask = cv2.findHomography(src_pts, dst_pts, method, ransacReprojThreshold)
```

| Parameter | Meaning |
|---|---|
| `src_pts` | Source points, shape `(N, 1, 2)` or `(N, 2)`, float32 |
| `dst_pts` | Destination points, same shape |
| `method` | `0` (least squares), `cv2.RANSAC`, `cv2.LMEDS`, or `cv2.RHO` |
| `ransacReprojThreshold` | Max reprojection error (pixels) for a point to be considered inlier (used with RANSAC) |

Returns:
- `H` — the 3x3 homography matrix
- `mask` — inlier mask (which points are consistent with the homography)

## RANSAC for Outlier Rejection

Feature matching inevitably produces some wrong matches (outliers). **RANSAC** (Random Sample Consensus) handles this by:

1. Randomly selecting 4 point correspondences
2. Computing a homography from those 4 points
3. Counting how many other matches agree with this homography (inliers)
4. Repeating many times and keeping the best homography

```python
# RANSAC: robust to outliers — the recommended method
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# mask is a (N, 1) array: 1 = inlier, 0 = outlier
inlier_count = mask.sum()
print(f'Inliers: {inlier_count}/{len(mask)}')
```

The `ransacReprojThreshold` (5.0 in the example) controls how close a projected point must be to its match to be considered an inlier. Lower values are stricter.

## Warping One Image to Another's Perspective

Once you have the homography, use `cv2.warpPerspective()` to transform one image into the other's coordinate system:

```python
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Warp img1 to align with img2's perspective
height, width = img2.shape[:2]
warped = cv2.warpPerspective(img1, H, (width, height))
```

For perspective correction (e.g., straightening a tilted document), you define the destination points manually:

```python
# Source: the 4 corners of the tilted document in the image
src = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Destination: where we want those corners to end up (a rectangle)
dst = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])

H = cv2.getPerspectiveTransform(src, dst)
corrected = cv2.warpPerspective(img, H, (300, 400))
```

## From Feature Matches to Homography

The typical pipeline to compute a homography from feature matches:

```python
# Detect and match features
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Extract matched point coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# Compute homography with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

## Tips & Common Mistakes

- You need **at least 4 point correspondences** to compute a homography. In practice, you want many more (20+) so RANSAC can reliably identify inliers.
- Always use `cv2.RANSAC` (not plain least squares) when working with feature matches, because there will always be some wrong matches.
- The `ransacReprojThreshold` controls outlier sensitivity. Values of 3.0-5.0 pixels work well for most cases. Too low = too strict, too high = accepts bad matches.
- Homography assumes a **planar scene** or pure camera rotation. For 3D scenes with significant depth variation, the homography model breaks down and you need a fundamental matrix instead.
- Point arrays must be `float32` and shaped as `(N, 1, 2)`. Use `.reshape(-1, 1, 2)` to ensure the correct shape.
- Check that `H` is not `None` before using it — `findHomography` returns `None` if it cannot compute a valid homography (e.g., too few points or all points are collinear).
- The output of `warpPerspective` may have black borders where the source image does not cover the destination. This is normal and expected.

## Starter Code

```python
import cv2
import numpy as np

# Create an image representing a "document" with features
doc = np.zeros((300, 400, 3), dtype=np.uint8)
doc[:] = (240, 235, 220)

# Add content to the document
cv2.putText(doc, 'DOCUMENT', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
cv2.line(doc, (50, 80), (350, 80), (30, 30, 30), 2)
for i in range(5):
    y = 110 + i * 30
    cv2.line(doc, (50, y), (350, y), (150, 150, 150), 1)
cv2.rectangle(doc, (50, 90), (350, 270), (100, 100, 100), 1)
cv2.circle(doc, (320, 240), 20, (0, 0, 180), -1)
cv2.putText(doc, 'OpenCV', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)

# Create a perspective-distorted version (as if photographed at an angle)
h, w = doc.shape[:2]
src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
# Simulate a tilted view
dst_corners = np.float32([[60, 40], [w - 30, 20], [w + 20, h + 30], [-40, h - 10]])

H_distort = cv2.getPerspectiveTransform(src_corners, dst_corners)
canvas_size = (w + 100, h + 100)
distorted = cv2.warpPerspective(doc, H_distort, canvas_size, borderValue=(50, 50, 50))

# --- Method 1: Manual perspective correction (known corners) ---
# We know where the document corners ended up in the distorted image
corrected_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
H_correct = cv2.getPerspectiveTransform(dst_corners, corrected_dst)
corrected = cv2.warpPerspective(distorted, H_correct, (w, h))

# --- Method 2: Feature-based homography with RANSAC ---
gray_doc = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
gray_dist = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray_doc, None)
kp2, des2 = orb.detectAndCompute(gray_dist, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches_knn = bf.knnMatch(des1, des2, k=2)

good = []
for pair in matches_knn:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts_feat = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_feat, mask = cv2.findHomography(dst_pts_feat, src_pts, cv2.RANSAC, 5.0)
    inliers = mask.sum() if mask is not None else 0
    feat_corrected = cv2.warpPerspective(distorted, H_feat, (w, h))
else:
    feat_corrected = np.zeros_like(doc)
    inliers = 0

# --- Draw matches ---
img_matches = cv2.drawMatches(doc, kp1, distorted, kp2, good[:30], None,
                              matchColor=(0, 255, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# --- Build comparison ---
# Resize for display
distorted_resized = cv2.resize(distorted, (w, h))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(doc, 'Original', (5, 20), font, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
cv2.putText(distorted_resized, 'Distorted', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(corrected, 'Manual Correction', (5, 20), font, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
cv2.putText(feat_corrected, f'RANSAC ({inliers} inliers)', (5, 20), font, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

top_row = np.hstack([doc, distorted_resized])
bottom_row = np.hstack([corrected, feat_corrected])
comparison = np.vstack([top_row, bottom_row])

# Final layout: comparison on top, matches on bottom
matches_resized = cv2.resize(img_matches, (comparison.shape[1], 250))
result = np.vstack([comparison, matches_resized])

print(f'Good feature matches: {len(good)}')
print(f'RANSAC inliers: {inliers}')
if len(good) >= 4:
    print(f'Homography matrix:\n{H_feat}')

cv2.imshow('Homography & Warping', result)
```
