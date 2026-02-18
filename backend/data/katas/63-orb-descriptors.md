---
slug: 63-orb-descriptors
title: ORB Descriptors
level: advanced
concepts: [cv2.ORB_create, keypoints, descriptors, oriented BRIEF]
prerequisites: [62-fast-keypoints]
---

## What Problem Are We Solving?

Detecting keypoints (like corners) tells you **where** interesting features are, but to actually **match** features between images, you need a **descriptor** — a compact numerical summary of the local patch around each keypoint. ORB (Oriented FAST and Rotated BRIEF) is OpenCV's recommended feature detector+descriptor because it is **fast, effective, and completely free** to use, unlike patented alternatives like SIFT and SURF.

## ORB as a Free Alternative to SIFT/SURF

SIFT and SURF were patent-encumbered for years (SIFT's patent expired in 2020). ORB was created by Ethan Rublee et al. at Willow Garage in 2011 as a fast, open-source alternative with competitive performance:

| Feature | ORB | SIFT | SURF |
|---|---|---|---|
| Speed | Very fast | Slow | Medium |
| Descriptor size | 32 bytes | 128 floats (512 bytes) | 64 floats (256 bytes) |
| Rotation invariant | Yes | Yes | Yes |
| Scale invariant | Partial (pyramid) | Yes (full) | Yes (full) |
| License | Free | Free (patent expired) | Patented |

ORB is typically **10x faster** than SIFT while achieving comparable matching accuracy for many tasks.

## How ORB Works

ORB combines two techniques:

1. **Oriented FAST** for keypoint detection — it uses FAST to find corners, then adds a scale pyramid for partial scale invariance and computes the orientation of each keypoint using the intensity centroid method.

2. **Rotated BRIEF** for description — BRIEF generates a binary descriptor by comparing pairs of pixel intensities in a patch, and ORB rotates these comparison pairs according to the keypoint orientation, making the descriptor rotation-invariant.

## Detecting Keypoints and Computing Descriptors

```python
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
```

| Return Value | Meaning |
|---|---|
| `keypoints` | List of `cv2.KeyPoint` objects with `.pt`, `.size`, `.angle`, `.response` |
| `descriptors` | NumPy array of shape `(N, 32)` with dtype `uint8` — each row is a 256-bit binary descriptor |

You can also separate detection and description:

```python
keypoints = orb.detect(gray, None)
keypoints, descriptors = orb.compute(gray, keypoints)
```

## The nfeatures Parameter

`nfeatures` controls how many keypoints ORB will retain (sorted by Harris corner score):

```python
# Default: 500 keypoints
orb_default = cv2.ORB_create(nfeatures=500)

# More keypoints for complex scenes
orb_many = cv2.ORB_create(nfeatures=2000)

# Fewer keypoints for speed
orb_few = cv2.ORB_create(nfeatures=100)
```

## Other Important Parameters

```python
orb = cv2.ORB_create(
    nfeatures=500,      # Max keypoints to retain
    scaleFactor=1.2,    # Pyramid scale factor (>1.0)
    nlevels=8,          # Number of pyramid levels
    edgeThreshold=31,   # Border where features are not detected
    patchSize=31,       # Size of the patch used by BRIEF descriptor
)
```

| Parameter | Meaning |
|---|---|
| `scaleFactor` | Scale ratio between pyramid levels. 1.2 means each level is 1.2x smaller |
| `nlevels` | Number of pyramid levels. More levels = more scale invariance |
| `edgeThreshold` | Size of border where no features are detected. Should be >= `patchSize` |
| `patchSize` | Size of the patch from which the BRIEF descriptor is computed |

## Drawing Keypoints

Use `cv2.drawKeypoints()` with the `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` flag to show keypoint size and orientation:

```python
img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

The rich flag draws circles proportional to the keypoint size and lines showing the orientation — this visualizes what ORB computed for each feature.

## Understanding the Binary Descriptor

ORB descriptors are **binary strings** stored as `uint8` arrays. Each descriptor is 32 bytes = 256 bits, where each bit is the result of one intensity comparison:

```python
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)

print(des.shape)    # (N, 32) — N keypoints, 32 bytes each
print(des.dtype)    # uint8
print(des[0])       # First descriptor: array of 32 uint8 values
```

Because descriptors are binary, matching uses **Hamming distance** (counting different bits) rather than Euclidean distance. This makes matching very fast.

## Tips & Common Mistakes

- Always convert to **grayscale** before calling `detectAndCompute`. ORB does not work on color images.
- The `descriptors` array can be `None` if no keypoints are detected. Always check before using it.
- ORB descriptors are **binary** (uint8), so you must use Hamming distance for matching — not L2 (Euclidean). Using L2 distance with ORB descriptors will give wrong results.
- Increasing `nfeatures` beyond what the image can support does not help — ORB will simply return fewer keypoints.
- The `scaleFactor` and `nlevels` together determine the scale range. With default values (1.2 and 8), ORB covers a scale range of about 1.2^8 = 4.3x.
- ORB's scale invariance is not as robust as SIFT's — for large scale changes (more than ~3x), SIFT will typically outperform ORB.
- The `edgeThreshold` prevents detection at image borders where the descriptor patch would extend outside the image. If you're getting no keypoints near borders, this is why.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with rich texture and corners
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (50, 50, 50)

# Draw various shapes
cv2.rectangle(img, (30, 30), (170, 170), (255, 255, 255), 2)
cv2.rectangle(img, (50, 50), (150, 150), (180, 180, 180), 2)
cv2.rectangle(img, (200, 30), (350, 150), (200, 200, 200), -1)
cv2.circle(img, (480, 100), 70, (255, 200, 150), 2)
cv2.circle(img, (480, 100), 40, (200, 150, 255), 2)
cv2.putText(img, 'ORB', (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)
pts = np.array([[350, 220], [450, 250], [500, 350], [400, 380], [320, 320]], dtype=np.int32)
cv2.fillPoly(img, [pts], (100, 200, 100))
cv2.polylines(img, [pts], True, (200, 255, 200), 2)

# Add some texture with random lines
rng = np.random.RandomState(42)
for _ in range(15):
    x1, y1 = rng.randint(0, 600), rng.randint(0, 400)
    x2, y2 = x1 + rng.randint(-80, 80), y1 + rng.randint(-80, 80)
    color = tuple(int(c) for c in rng.randint(100, 255, 3))
    cv2.line(img, (x1, y1), (x2, y2), color, 1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- ORB with different nfeatures ---
orb_100 = cv2.ORB_create(nfeatures=100)
orb_500 = cv2.ORB_create(nfeatures=500)
orb_1500 = cv2.ORB_create(nfeatures=1500)

kp_100, des_100 = orb_100.detectAndCompute(gray, None)
kp_500, des_500 = orb_500.detectAndCompute(gray, None)
kp_1500, des_1500 = orb_1500.detectAndCompute(gray, None)

# --- Draw keypoints with rich info (size + orientation) ---
img_100 = cv2.drawKeypoints(img, kp_100, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_500 = cv2.drawKeypoints(img, kp_500, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_1500 = cv2.drawKeypoints(img, kp_1500, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- Plain keypoints (no size/orientation) for comparison ---
img_plain = cv2.drawKeypoints(img, kp_500, None, color=(0, 0, 255),
                              flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_100, f'nfeatures=100 ({len(kp_100)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_500, f'nfeatures=500 ({len(kp_500)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_1500, f'nfeatures=1500 ({len(kp_1500)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img_plain, f'Plain draw ({len(kp_500)} kp)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Build comparison grid
top_row = np.hstack([img_100, img_500])
bottom_row = np.hstack([img_1500, img_plain])
result = np.vstack([top_row, bottom_row])

# Print descriptor info
print(f'Keypoints (nfeatures=100):  {len(kp_100)}')
print(f'Keypoints (nfeatures=500):  {len(kp_500)}')
print(f'Keypoints (nfeatures=1500): {len(kp_1500)}')
if des_500 is not None:
    print(f'Descriptor shape: {des_500.shape}')
    print(f'Descriptor dtype: {des_500.dtype}')
    print(f'First descriptor (first 8 bytes): {des_500[0][:8]}')

cv2.imshow('ORB Descriptors', result)
```
