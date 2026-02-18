---
slug: 97-ar-overlay
title: Augmented Reality Overlay
level: advanced
concepts: [marker detection, homography, image overlay]
prerequisites: [67-homography-warping, 14-bitwise-operations]
---

## What Problem Are We Solving?

Augmented reality (AR) overlays virtual content onto a real-world scene. The most common approach uses a **known marker** or pattern: the system detects the marker in the camera frame, computes a homography that maps from the overlay image to the marker's position, warps the overlay to match the marker's perspective, and blends it into the scene.

Think of it like projecting a movie onto a screen that can be at any angle -- the overlay must be warped to match the marker's orientation and position in the scene.

## Step 1: Define a Known Marker Pattern

The marker is a distinctive pattern that is easy to detect. In practice, ArUco markers or QR codes are used. For our synthetic demo, we use a simple geometric pattern -- a bordered square with an asymmetric internal design:

```python
marker = np.ones((100, 100, 3), dtype=np.uint8) * 255
cv2.rectangle(marker, (0, 0), (99, 99), (0, 0, 0), 5)
cv2.rectangle(marker, (15, 15), (50, 50), (0, 0, 0), -1)
cv2.circle(marker, (70, 70), 15, (0, 0, 0), -1)
```

The asymmetry ensures we can determine the marker's orientation (not just its position).

## Step 2: Place the Marker in a Scene

We warp the marker onto a synthetic scene at an angle, simulating how a marker would appear in a camera frame:

```python
src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
dst_pts = np.array([[120, 80], [320, 100], [310, 280], [100, 260]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
cv2.warpPerspective(marker, M, (scene_w, scene_h), dst=scene, borderMode=cv2.BORDER_TRANSPARENT)
```

## Step 3: Detect the Marker in the Scene

In a real AR system, you would detect the marker using feature matching or a dedicated library (ArUco). Here we simulate detection by using ORB features to find the marker's corners in the scene:

```python
orb = cv2.ORB_create(500)
kp1, des1 = orb.detectAndCompute(marker_gray, None)
kp2, des2 = orb.detectAndCompute(scene_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)
```

After matching and filtering, we compute the homography from the marker's canonical coordinates to its position in the scene.

## Step 4: Compute the Homography

The homography `H` maps points from the marker (flat, frontal view) to their positions in the scene (warped, angled). Given good matches:

```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

This `H` tells us exactly how to warp any overlay content so it aligns with the marker in the scene.

## Step 5: Warp the Overlay Image

The overlay image (what we want to "project" onto the marker) is warped using the same homography:

```python
overlay_warped = cv2.warpPerspective(overlay, H, (scene_w, scene_h))
```

## Step 6: Blend Using Masks

To composite the warped overlay onto the scene cleanly, we create a mask that covers only the overlay region, then use bitwise operations to blend:

```python
# Create a white mask the same size as the overlay
mask = np.ones_like(overlay) * 255
mask_warped = cv2.warpPerspective(mask, H, (scene_w, scene_h))
mask_gray = cv2.cvtColor(mask_warped, cv2.COLOR_BGR2GRAY)
_, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

# Invert mask for the background
mask_inv = cv2.bitwise_not(mask_bin)

# Combine: scene background + warped overlay foreground
bg = cv2.bitwise_and(scene, scene, mask=mask_inv)
fg = cv2.bitwise_and(overlay_warped, overlay_warped, mask=mask_bin)
result = cv2.add(bg, fg)
```

## The Complete Pipeline

1. **Input**: Scene image with a known marker + overlay content
2. **Detect Marker**: Find the marker using feature matching
3. **Compute Homography**: Map from marker space to scene space
4. **Warp Overlay**: Transform the overlay to match the marker's perspective
5. **Create Mask**: Define the overlay region
6. **Composite**: Blend overlay onto scene using bitwise operations
7. **Output**: Scene with virtual content overlaid on the marker

## Tips & Common Mistakes

- The marker must have enough texture/features for ORB to detect. Plain colored squares won't work. Add internal patterns with corners and edges.
- Homography requires at least 4 point correspondences. Having 10+ good matches makes RANSAC much more robust.
- The mask must be perfectly aligned with the warped overlay. Using the same homography for both ensures this.
- Use `cv2.BORDER_TRANSPARENT` in `warpPerspective` when warping onto an existing image to preserve the background.
- For smooth AR, apply temporal filtering (average homographies across frames) to prevent jittering.
- Anti-aliasing at the edges of the overlay improves visual quality. Apply a slight Gaussian blur to the mask boundary.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create the marker pattern (known reference)
# =============================================================
marker_size = 120
marker = np.ones((marker_size, marker_size, 3), dtype=np.uint8) * 255

# Draw distinctive pattern inside marker
cv2.rectangle(marker, (0, 0), (marker_size - 1, marker_size - 1), (0, 0, 0), 8)
# Quadrant-based pattern for orientation detection
cv2.rectangle(marker, (15, 15), (50, 50), (0, 0, 0), -1)      # Top-left: filled square
cv2.circle(marker, (85, 35), 18, (0, 0, 0), -1)                 # Top-right: circle
cv2.rectangle(marker, (15, 70), (50, 105), (0, 0, 0), 3)        # Bottom-left: hollow square
# Bottom-right: triangle
tri = np.array([[70, 105], [105, 105], [87, 70]], np.int32)
cv2.fillPoly(marker, [tri], (0, 0, 0))

# =============================================================
# Step 2: Create the overlay image (virtual content)
# =============================================================
overlay = np.zeros((marker_size, marker_size, 3), dtype=np.uint8)
overlay[:] = (30, 120, 30)  # Green background

# Draw virtual content on the overlay
cv2.putText(overlay, 'AR', (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(overlay, 'DEMO', (15, 95), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (200, 255, 200), 2, cv2.LINE_AA)
# Add a border to the overlay
cv2.rectangle(overlay, (3, 3), (marker_size - 4, marker_size - 4), (0, 255, 0), 2)

# =============================================================
# Step 3: Create a scene and place the marker at an angle
# =============================================================
scene_h, scene_w = 400, 600
scene = np.zeros((scene_h, scene_w, 3), dtype=np.uint8)
scene[:] = (50, 45, 40)

# Add scene background elements (table, wall)
cv2.rectangle(scene, (0, 0), (scene_w, 220), (80, 75, 65), -1)   # Wall
cv2.rectangle(scene, (0, 220), (scene_w, scene_h), (60, 55, 45), -1)  # Table
# Wall decoration
cv2.rectangle(scene, (400, 30), (560, 180), (90, 85, 70), -1)
cv2.rectangle(scene, (410, 40), (550, 170), (100, 95, 80), -1)

# Define where the marker appears in the scene (simulating 3D perspective)
marker_corners_scene = np.array([
    [130, 120],   # top-left
    [330, 100],   # top-right
    [340, 320],   # bottom-right
    [110, 310]    # bottom-left
], dtype=np.float32)

# Warp marker onto scene
marker_corners_flat = np.array([
    [0, 0], [marker_size, 0],
    [marker_size, marker_size], [0, marker_size]
], dtype=np.float32)

M_place = cv2.getPerspectiveTransform(marker_corners_flat, marker_corners_scene)
scene_with_marker = scene.copy()
cv2.warpPerspective(marker, M_place, (scene_w, scene_h),
                     dst=scene_with_marker, borderMode=cv2.BORDER_TRANSPARENT)

# =============================================================
# Step 4: Detect the marker using ORB feature matching
# =============================================================
marker_gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
scene_gray = cv2.cvtColor(scene_with_marker, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(marker_gray, None)
kp2, des2 = orb.detectAndCompute(scene_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f'ORB keypoints - Marker: {len(kp1)}, Scene: {len(kp2)}')
    print(f'Good matches: {len(good_matches)}')
else:
    good_matches = []
    print('Not enough descriptors for matching')

# =============================================================
# Step 5: Compute homography and warp overlay
# =============================================================
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f'Homography inliers: {inliers}')

    if H is not None:
        # Warp the overlay using the detected homography
        overlay_warped = cv2.warpPerspective(overlay, H, (scene_w, scene_h))

        # Create a mask for the overlay region
        overlay_mask = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
        mask_warped = cv2.warpPerspective(overlay_mask, H, (scene_w, scene_h))
        _, mask_bin = cv2.threshold(mask_warped, 1, 255, cv2.THRESH_BINARY)

        # Slightly erode mask to avoid edge artifacts
        mask_bin = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)

        # Composite: background where mask is 0, overlay where mask is 255
        mask_inv = cv2.bitwise_not(mask_bin)
        bg_part = cv2.bitwise_and(scene_with_marker, scene_with_marker, mask=mask_inv)
        fg_part = cv2.bitwise_and(overlay_warped, overlay_warped, mask=mask_bin)
        ar_result = cv2.add(bg_part, fg_part)

        print('AR overlay applied successfully')
    else:
        ar_result = scene_with_marker.copy()
        print('Homography computation failed')
else:
    # Fallback: use the known placement homography directly
    print(f'Not enough matches ({len(good_matches)}). Using known marker position.')
    H = M_place

    overlay_warped = cv2.warpPerspective(overlay, H, (scene_w, scene_h))
    overlay_mask = np.ones((marker_size, marker_size), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(overlay_mask, H, (scene_w, scene_h))
    _, mask_bin = cv2.threshold(mask_warped, 1, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.erode(mask_bin, np.ones((3, 3), np.uint8), iterations=1)

    mask_inv = cv2.bitwise_not(mask_bin)
    bg_part = cv2.bitwise_and(scene_with_marker, scene_with_marker, mask=mask_inv)
    fg_part = cv2.bitwise_and(overlay_warped, overlay_warped, mask=mask_bin)
    ar_result = cv2.add(bg_part, fg_part)

# =============================================================
# Step 6: Build visualization
# =============================================================
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw detected marker boundary on a copy of the scene
detection_display = scene_with_marker.copy()
corners = marker_corners_flat.reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(corners, H)
if projected is not None:
    pts = projected.reshape(-1, 2).astype(np.int32)
    cv2.polylines(detection_display, [pts], True, (0, 255, 0), 3)
    for pt in pts:
        cv2.circle(detection_display, tuple(pt), 6, (0, 0, 255), -1)

# Labels
cv2.putText(detection_display, 'Marker Detected', (10, 25), font, 0.6, (0, 255, 0), 2)
cv2.putText(ar_result, 'AR Overlay', (10, 25), font, 0.6, (0, 255, 0), 2)

# Side panel: show marker and overlay at small size
panel_h = scene_h
panel_w = 160
side_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
side_panel[:] = (30, 30, 30)

# Place marker and overlay previews
marker_small = cv2.resize(marker, (100, 100))
overlay_small = cv2.resize(overlay, (100, 100))
side_panel[20:120, 30:130] = marker_small
side_panel[160:260, 30:130] = overlay_small
cv2.putText(side_panel, 'Marker', (40, 140), font, 0.45, (200, 200, 200), 1)
cv2.putText(side_panel, 'Overlay', (38, 280), font, 0.45, (200, 200, 200), 1)
cv2.putText(side_panel, 'Inputs', (45, 15), font, 0.45, (0, 255, 0), 1)

# Combine: side panel + detection + AR result
top_row = np.hstack([side_panel, detection_display])
# Pad AR result to match width
ar_padded = np.zeros_like(top_row)
x_off = (top_row.shape[1] - ar_result.shape[1]) // 2
ar_padded[:ar_result.shape[0], x_off:x_off + ar_result.shape[1]] = ar_result
cv2.putText(ar_padded, 'Final AR Result', (x_off + 10, 25), font, 0.6, (0, 255, 0), 2)

result = np.vstack([top_row, ar_padded])

cv2.imshow('Augmented Reality Overlay', result)
```
