---
slug: 64-sift-descriptors
title: SIFT Descriptors
level: advanced
concepts: [cv2.SIFT_create, scale-invariant features, 128-dim descriptor]
prerequisites: [63-orb-descriptors]
---

## What Problem Are We Solving?

When images are taken at different scales (zoomed in vs. zoomed out), simple corner detectors like Harris or FAST fail because a corner at one scale may look like an edge or flat region at another scale. **SIFT** (Scale-Invariant Feature Transform) solves this by detecting features that are **stable across scales** and computing highly distinctive **128-dimensional descriptors** that remain consistent under scale changes, rotation, and moderate viewpoint changes. SIFT is considered the gold standard for feature description accuracy.

## SIFT's Scale Invariance

SIFT achieves scale invariance by searching for features across a **scale space** — a series of progressively blurred versions of the image organized into **octaves**:

1. **Gaussian blurring** at multiple sigma values within each octave
2. **Difference of Gaussians (DoG)**: subtracting adjacent blur levels to detect edges/blobs
3. **Extrema detection**: finding points that are local maxima or minima in both spatial and scale dimensions
4. **Keypoint refinement**: sub-pixel and sub-scale localization using Taylor expansion

```
Octave 0: original resolution
  Scale 0: sigma = 1.6
  Scale 1: sigma = 1.6 * k
  Scale 2: sigma = 1.6 * k^2
  ...
Octave 1: half resolution
  Scale 0: sigma = 2 * 1.6
  ...
```

Each keypoint records the scale at which it was found, making the detector truly scale-invariant.

## The 128-Dimensional Descriptor

Once a keypoint is detected, SIFT computes its descriptor:

1. A 16x16 pixel neighborhood around the keypoint is taken (at the keypoint's detected scale)
2. The neighborhood is rotated to align with the keypoint's dominant orientation (rotation invariance)
3. The patch is divided into 4x4 sub-regions
4. In each sub-region, an 8-bin gradient orientation histogram is computed
5. 4x4 sub-regions x 8 bins = **128-dimensional float vector**

```python
# SIFT descriptor properties
# - 128 float32 values per keypoint
# - L2-normalized, then clipped at 0.2, then re-normalized
# - Very distinctive — low false match rate
```

## Using cv2.SIFT_create()

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
```

| Return Value | Meaning |
|---|---|
| `keypoints` | List of `cv2.KeyPoint` with `.pt`, `.size` (scale), `.angle` (orientation), `.octave` |
| `descriptors` | NumPy array of shape `(N, 128)` with dtype `float32` |

Key creation parameters:

```python
sift = cv2.SIFT_create(
    nfeatures=0,           # Max features (0 = no limit)
    nOctaveLayers=3,       # Layers per octave (default 3)
    contrastThreshold=0.04, # Filter low-contrast features
    edgeThreshold=10,      # Filter edge-like features
    sigma=1.6,             # Initial Gaussian blur sigma
)
```

## Octaves and Scales

The `nOctaveLayers` parameter controls how many blur levels are computed per octave:

```python
# Default: 3 layers per octave — good balance
sift_default = cv2.SIFT_create(nOctaveLayers=3)

# More layers: finer scale sampling, more keypoints, slower
sift_fine = cv2.SIFT_create(nOctaveLayers=5)
```

The number of octaves is determined automatically from the image size (each octave halves the resolution until the image is too small).

## Contrast and Edge Thresholds

SIFT has two filtering thresholds that control keypoint quality:

```python
# contrastThreshold: reject low-contrast keypoints
# Lower value = more keypoints (including weaker ones)
sift_sensitive = cv2.SIFT_create(contrastThreshold=0.02)

# edgeThreshold: reject edge-like keypoints (not corners)
# Higher value = more permissive (keeps more edge-like features)
sift_permissive = cv2.SIFT_create(edgeThreshold=20)
```

## Comparison with ORB

| Aspect | SIFT | ORB |
|---|---|---|
| Descriptor type | Float (128 x float32) | Binary (32 x uint8) |
| Descriptor size | 512 bytes | 32 bytes |
| Distance metric | L2 (Euclidean) | Hamming |
| Detection speed | Slow | Very fast (~10x) |
| Matching accuracy | Excellent | Good |
| Scale invariance | Full (scale space) | Partial (pyramid) |
| Best for | Accuracy-critical matching | Real-time applications |

```python
# SIFT: use L2 distance for matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# ORB: use Hamming distance for matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

## Tips & Common Mistakes

- SIFT requires **grayscale** input. Convert color images with `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.
- SIFT descriptors are `float32`, not binary. You **must** use L2 distance (not Hamming) for matching. Using the wrong distance metric is a very common bug.
- SIFT is significantly slower than ORB — expect 5-10x longer computation times. For real-time applications, ORB is usually the better choice.
- The `nfeatures` parameter limits the total number of keypoints. Set to 0 (default) for no limit, but be aware that complex images can produce thousands of keypoints.
- SIFT's patent expired in 2020, so it is now freely available in the main OpenCV package (`cv2.SIFT_create()`). In older OpenCV versions, it was in `cv2.xfeatures2d`.
- The 128-dimensional descriptor is very distinctive, which means SIFT has a lower false match rate than ORB, but at the cost of more memory and slower matching.
- If `detectAndCompute` returns `([], None)`, the image likely has insufficient texture or contrast. Try lowering `contrastThreshold`.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with rich features at various scales
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (50, 50, 50)

# Draw shapes at different sizes to demonstrate scale detection
cv2.rectangle(img, (20, 20), (100, 100), (255, 255, 255), 2)
cv2.rectangle(img, (25, 25), (95, 95), (200, 200, 200), 1)
cv2.rectangle(img, (150, 20), (280, 150), (255, 255, 255), 2)
cv2.rectangle(img, (160, 30), (270, 140), (200, 200, 200), 1)
cv2.circle(img, (400, 80), 60, (255, 200, 150), 2)
cv2.circle(img, (400, 80), 30, (200, 150, 255), 2)
cv2.circle(img, (400, 80), 10, (150, 255, 200), 2)
cv2.putText(img, 'SIFT', (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)
cv2.putText(img, 'Scale', (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)

# Add texture
pts = np.array([[350, 200], [550, 220], [570, 380], [380, 360]], dtype=np.int32)
cv2.fillPoly(img, [pts], (80, 120, 80))
rng = np.random.RandomState(42)
for _ in range(30):
    x, y = rng.randint(360, 560), rng.randint(210, 370)
    cv2.circle(img, (x, y), rng.randint(2, 8), (150, 200, 150), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- SIFT with default parameters ---
sift_default = cv2.SIFT_create()
kp_default, des_default = sift_default.detectAndCompute(gray, None)

# --- SIFT with more features ---
sift_many = cv2.SIFT_create(nfeatures=1000)
kp_many, des_many = sift_many.detectAndCompute(gray, None)

# --- SIFT with lower contrast threshold (more sensitive) ---
sift_sensitive = cv2.SIFT_create(contrastThreshold=0.02)
kp_sensitive, des_sensitive = sift_sensitive.detectAndCompute(gray, None)

# --- ORB for comparison ---
orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(gray, None)

# Draw keypoints with size and orientation
img_default = cv2.drawKeypoints(img, kp_default, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_many = cv2.drawKeypoints(img, kp_many, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sensitive = cv2.drawKeypoints(img, kp_sensitive, None, color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 0, 255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_default, f'SIFT default ({len(kp_default)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_many, f'SIFT nfeatures=1000 ({len(kp_many)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_sensitive, f'SIFT contrast=0.02 ({len(kp_sensitive)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_orb, f'ORB comparison ({len(kp_orb)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Build comparison grid
top_row = np.hstack([img_default, img_many])
bottom_row = np.hstack([img_sensitive, img_orb])
result = np.vstack([top_row, bottom_row])

# Print descriptor comparison
print(f'SIFT keypoints (default): {len(kp_default)}')
print(f'SIFT keypoints (nfeatures=1000): {len(kp_many)}')
print(f'SIFT keypoints (contrast=0.02): {len(kp_sensitive)}')
print(f'ORB keypoints: {len(kp_orb)}')
if des_default is not None:
    print(f'SIFT descriptor shape: {des_default.shape} (128 floats)')
    print(f'SIFT descriptor dtype: {des_default.dtype}')
if des_orb is not None:
    print(f'ORB descriptor shape:  {des_orb.shape} (32 bytes)')
    print(f'ORB descriptor dtype:  {des_orb.dtype}')

cv2.imshow('SIFT Descriptors', result)
```
