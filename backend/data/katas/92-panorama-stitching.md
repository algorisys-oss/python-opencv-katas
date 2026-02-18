---
slug: 92-panorama-stitching
title: Panorama Stitching Pipeline
level: advanced
concepts: [feature detection, matching, homography, blending]
prerequisites: [68-image-stitching]
---

## What Problem Are We Solving?

You take two photos of a scene, panning the camera slightly between shots, so the images overlap. You want to combine them into a single wide **panorama** that seamlessly covers the full field of view. This requires finding corresponding points between the images, computing a geometric transformation to align them, warping one image to match the other's perspective, and blending the overlap region to avoid visible seams.

This is the full panorama stitching pipeline: feature detection, feature matching, homography estimation, perspective warping, and blending.

## Step 1: Create Two Overlapping Synthetic Scenes

We create two images that share an overlapping region. The left image shows shapes on the left side plus a shared middle area, and the right image shows the same middle area plus shapes on the right:

```python
# Full scene
full = np.zeros((300, 800, 3), dtype=np.uint8)
# Draw various shapes across the full scene...

# Left and right "photos" with overlap
left = full[:, 0:500].copy()
right = full[:, 300:800].copy()
# Overlap region: columns 300-500 of the full scene
```

## Step 2: Feature Detection with ORB

ORB (Oriented FAST and Rotated BRIEF) detects keypoints and computes binary descriptors. It is fast and free to use (unlike SIFT/SURF which were patented):

```python
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
```

Each keypoint has a position, scale, and orientation. The descriptor is a 32-byte binary string that encodes the local image patch around the keypoint.

## Step 3: Feature Matching with BFMatcher

`cv2.BFMatcher` (Brute-Force Matcher) compares every descriptor in image 1 against every descriptor in image 2. For binary descriptors (ORB), we use the Hamming distance:

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
```

We use `knnMatch` with `k=2` to get the two best matches for each descriptor, enabling **Lowe's ratio test**:

```python
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
```

The ratio test discards ambiguous matches where the best and second-best matches are too similar, significantly reducing false matches.

## Step 4: Compute Homography with RANSAC

A **homography** is a 3x3 matrix that maps points from one image plane to another. Given at least 4 good point correspondences, we use `cv2.findHomography` with RANSAC to robustly estimate it:

```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

RANSAC (Random Sample Consensus) handles outliers by randomly sampling minimal subsets, computing a homography from each, and keeping the one that has the most inliers within the 5.0 pixel threshold.

## Step 5: Warp and Blend

With the homography `H`, we warp the first image into the coordinate space of the second, then blend the overlapping region:

```python
warped = cv2.warpPerspective(left, H, (output_w, output_h))
# Place the right image into the panorama
warped[0:h2, 0:w2] = right
```

For a cleaner result, we blend the overlap region by averaging or using distance-weighted masks.

## The Complete Pipeline

1. **Input**: Two overlapping images
2. **Feature Detection**: ORB keypoints and descriptors
3. **Feature Matching**: BFMatcher with ratio test
4. **Homography Estimation**: RANSAC-based robust estimation
5. **Warp**: Transform one image to align with the other
6. **Blend**: Combine the images with overlap handling
7. **Output**: Single panoramic image

## Tips & Common Mistakes

- ORB's `nfeatures` parameter controls how many keypoints to detect. Too few and you may not find enough matches; too many and matching becomes slow.
- The ratio test threshold (0.75) is a balance between keeping good matches and rejecting bad ones. Lower values are stricter.
- `cv2.findHomography` needs at least 4 point pairs, but works better with many more. If you have fewer than 10 good matches, the result may be unreliable.
- The RANSAC threshold (5.0 pixels) determines how close a point must be to the model to be considered an inlier. Increase it for lower-resolution images.
- When warping, compute the output canvas size carefully to avoid clipping. Use the homography to transform corner points and determine the bounding box.
- Simple overwriting of pixels in the overlap region creates hard seams. Linear blending or feathering produces much smoother results.
- Homography assumes a planar scene or pure camera rotation. For scenes with significant depth variation, the stitching may show ghosting.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create a full synthetic scene and split into two
#         overlapping "photos"
# =============================================================
scene_h, scene_w = 300, 800
full_scene = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)
full_scene[:] = (40, 35, 30)  # Dark background

# Draw a ground plane
cv2.rectangle(full_scene, (0, 200), (800, 300), (60, 80, 60), -1)

# Draw various distinct shapes across the scene
cv2.rectangle(full_scene, (50, 80), (150, 190), (0, 0, 200), -1)      # Red rect
cv2.circle(full_scene, (250, 140), 50, (200, 200, 0), -1)              # Cyan circle
cv2.rectangle(full_scene, (320, 60), (420, 190), (200, 100, 0), -1)    # Blue rect
cv2.circle(full_scene, (500, 130), 40, (0, 200, 200), -1)              # Yellow circle
cv2.rectangle(full_scene, (570, 90), (650, 180), (0, 180, 0), -1)      # Green rect
cv2.circle(full_scene, (720, 150), 45, (180, 0, 180), -1)              # Purple circle

# Add textured elements for better feature detection
for x in range(0, 800, 40):
    cv2.line(full_scene, (x, 200), (x + 20, 195), (100, 120, 80), 1)
for y in range(0, 200, 25):
    for x in range(0, 800, 30):
        cv2.circle(full_scene, (x + (y % 2) * 15, y), 2, (80, 70, 60), -1)

# Add text landmarks
cv2.putText(full_scene, 'A', (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(full_scene, 'B', (360, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.putText(full_scene, 'C', (600, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

# Split into overlapping left and right images
# Left: columns 0-499, Right: columns 300-799 => overlap = 300-499 (200px)
left_img = full_scene[:, 0:500].copy()
right_img = full_scene[:, 300:800].copy()

print(f'Left image: {left_img.shape}, Right image: {right_img.shape}')
print(f'Overlap region: 200 pixels wide')

# =============================================================
# Step 2: Detect ORB features in both images
# =============================================================
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray_left, None)
kp2, des2 = orb.detectAndCompute(gray_right, None)

print(f'Keypoints - Left: {len(kp1)}, Right: {len(kp2)}')

# =============================================================
# Step 3: Match features using BFMatcher + ratio test
# =============================================================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f'Good matches after ratio test: {len(good_matches)}')

# =============================================================
# Step 4: Compute homography with RANSAC
# =============================================================
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f'Homography inliers: {inliers}/{len(good_matches)}')
    print(f'Homography matrix:\n{H}')

    # ==========================================================
    # Step 5: Warp left image and combine with right image
    # ==========================================================
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]

    # Determine output canvas size by transforming corners of left image
    corners_left = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners_warped = cv2.perspectiveTransform(corners_left, H)

    # Combine with corners of right image
    corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_warped, corners_right], axis=0)

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    # Translation to keep everything in positive coordinates
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]], dtype=np.float64)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # Warp left image into the panorama canvas
    panorama = cv2.warpPerspective(left_img, translation @ H, (out_w, out_h))

    # Place right image (with blending in overlap)
    tx, ty = -x_min, -y_min
    # Create a mask for where the warped left image has content
    left_mask = (panorama > 0).any(axis=2)

    # Place right image onto panorama
    right_region = panorama[ty:ty + h2, tx:tx + w2]
    right_mask = (right_region > 0).any(axis=2)

    # Blend overlap: simple averaging
    overlap = right_mask[:right_img.shape[0], :right_img.shape[1]]
    blended = right_img.copy()
    if overlap.any():
        blend_area = overlap
        blended[blend_area] = cv2.addWeighted(
            right_region[:right_img.shape[0], :right_img.shape[1]][blend_area].reshape(-1, 3),
            0.5,
            right_img[blend_area].reshape(-1, 3),
            0.5, 0
        ).reshape(-1, 3)

    panorama[ty:ty + h2, tx:tx + w2] = blended

    # ==========================================================
    # Step 6: Build visualization
    # ==========================================================
    # Draw matches for display
    match_img = cv2.drawMatches(left_img, kp1, right_img, kp2,
                                good_matches[:30], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Resize panorama to fit display
    disp_w = match_img.shape[1]
    scale = disp_w / panorama.shape[1]
    pano_resized = cv2.resize(panorama, (disp_w, int(panorama.shape[0] * scale)))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(match_img, 'Feature Matches (top 30)', (10, 25),
                font, 0.6, (0, 255, 0), 2)
    cv2.putText(pano_resized, 'Stitched Panorama', (10, 25),
                font, 0.6, (0, 255, 0), 2)

    result = np.vstack([match_img, pano_resized])
    print(f'Panorama size: {panorama.shape[1]}x{panorama.shape[0]}')
else:
    print(f'Not enough matches ({len(good_matches)}). Need at least 4.')
    result = np.hstack([left_img, right_img])

cv2.imshow('Panorama Stitching Pipeline', result)
```
