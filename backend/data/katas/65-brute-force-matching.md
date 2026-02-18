---
slug: 65-brute-force-matching
title: Brute-Force Matching
level: advanced
concepts: [cv2.BFMatcher, Hamming distance, L2 distance, crossCheck]
prerequisites: [63-orb-descriptors]
---

## What Problem Are We Solving?

Once you have computed feature descriptors for two images, you need to figure out **which descriptors in image 1 correspond to which descriptors in image 2**. This is the feature matching problem. **Brute-Force Matching** is the simplest approach: for each descriptor in image 1, it computes the distance to **every** descriptor in image 2 and picks the closest one. It guarantees finding the best match (unlike approximate methods) and is the foundation for understanding all matching techniques.

## Using cv2.BFMatcher

```python
bf = cv2.BFMatcher(normType, crossCheck)
```

| Parameter | Meaning |
|---|---|
| `normType` | Distance metric — `cv2.NORM_HAMMING` for binary descriptors (ORB), `cv2.NORM_L2` for float descriptors (SIFT) |
| `crossCheck` | If `True`, only returns matches where both descriptors are each other's best match |

## Choosing the Distance Metric

The distance metric **must** match the descriptor type:

```python
# For ORB (binary descriptors) — use Hamming distance
# Hamming distance counts the number of different bits
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# For SIFT (float descriptors) — use L2 (Euclidean) distance
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
```

Using the wrong distance metric (e.g., L2 with ORB) will produce meaningless matches. This is one of the most common feature matching bugs.

## Basic Matching with crossCheck

The `crossCheck` parameter enables a simple consistency check: a match (i, j) is only kept if descriptor `i` in image 1 is the closest to descriptor `j` in image 2 **and** descriptor `j` in image 2 is the closest to descriptor `i` in image 1:

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort by distance (best matches first)
matches = sorted(matches, key=lambda m: m.distance)
```

Each match object `m` has:
- `m.queryIdx` — index into `des1`
- `m.trainIdx` — index into `des2`
- `m.distance` — the distance between the two descriptors

## The Ratio Test (Lowe's Test)

A more robust filtering method is **Lowe's ratio test**. Instead of `crossCheck`, find the **two best** matches for each descriptor and reject matches where the best and second-best are too similar:

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # crossCheck must be False
matches_knn = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

The idea: if the best match is **much closer** than the second-best, it is likely a true match. If they are similar, the match is ambiguous and should be rejected. A ratio threshold of 0.75 is common; lower values are more strict.

> **Note:** `crossCheck=True` and `knnMatch` with `k=2` are **mutually exclusive** strategies. Use one or the other.

## Drawing Matches

OpenCV provides `cv2.drawMatches()` and `cv2.drawMatchesKnn()` to visualize matches side by side:

```python
# Draw single matches (from bf.match)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw KNN matches (from bf.knnMatch)
good = [[m] for m in good_matches]
result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:20], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

## Tips & Common Mistakes

- **Always** match the distance metric to the descriptor type: `NORM_HAMMING` for ORB/BRIEF (binary), `NORM_L2` for SIFT/SURF (float). This is the single most common mistake.
- `crossCheck=True` and `knnMatch(k=2)` are **not compatible** — crossCheck requires `bf.match()`. If you want the ratio test, set `crossCheck=False`.
- Sort matches by `m.distance` before drawing — this ensures you visualize the best matches first.
- The ratio test threshold of 0.75 works well in general, but you may need to adjust it. More distinctive scenes allow higher thresholds (0.8); repetitive textures need lower ones (0.6).
- If you get zero matches, check that: (1) both images have enough texture for feature detection, (2) you used the correct distance metric, and (3) your descriptors are not `None`.
- `drawMatches` places the two images side by side and draws lines between matched keypoints. The output image can be very wide, so resize for display if needed.
- Brute-force matching checks every descriptor against every other — its time complexity is O(N*M) where N and M are the number of descriptors. For large descriptor sets, consider FLANN matching instead.

## Starter Code

```python
import cv2
import numpy as np

# Create two related images (original and a rotated/shifted version)
img1 = np.zeros((300, 400, 3), dtype=np.uint8)
img1[:] = (50, 50, 50)
cv2.rectangle(img1, (30, 30), (150, 150), (255, 255, 255), 2)
cv2.circle(img1, (280, 80), 50, (200, 200, 255), 2)
cv2.putText(img1, 'A', (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (180, 180, 180), 3)
pts1 = np.array([[200, 180], [350, 180], [350, 270], [250, 270]], dtype=np.int32)
cv2.fillPoly(img1, [pts1], (100, 180, 100))
cv2.polylines(img1, [pts1], True, (150, 220, 150), 2)
rng = np.random.RandomState(42)
for _ in range(20):
    x, y = rng.randint(0, 400), rng.randint(0, 300)
    cv2.circle(img1, (x, y), rng.randint(2, 6), (rng.randint(100, 255), rng.randint(100, 255), rng.randint(100, 255)), -1)

# Create a transformed version (slight rotation + translation)
rows, cols = img1.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 0.9)
M[0, 2] += 30
M[1, 2] += 20
img2 = cv2.warpAffine(img1, M, (cols, rows), borderValue=(50, 50, 50))

# Add slight noise to img2
noise = np.random.randint(-15, 15, img2.shape, dtype=np.int16)
img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- Detect ORB features ---
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# --- Method 1: crossCheck matching ---
bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_cross = bf_cross.match(des1, des2)
matches_cross = sorted(matches_cross, key=lambda m: m.distance)

# --- Method 2: Ratio test matching ---
bf_ratio = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches_knn = bf_ratio.knnMatch(des1, des2, k=2)

good_ratio = []
for pair in matches_knn:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_ratio.append(m)

# --- Draw matches ---
img_cross = cv2.drawMatches(img1, kp1, img2, kp2, matches_cross[:20], None,
                            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_ratio = cv2.drawMatches(img1, kp1, img2, kp2, good_ratio[:20], None,
                            matchColor=(0, 255, 255), singlePointColor=(255, 0, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_cross, f'CrossCheck ({len(matches_cross)} matches)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_ratio, f'Ratio Test ({len(good_ratio)} matches)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Stack vertically
result = np.vstack([img_cross, img_ratio])

print(f'Keypoints in image 1: {len(kp1)}')
print(f'Keypoints in image 2: {len(kp2)}')
print(f'CrossCheck matches: {len(matches_cross)}')
print(f'Ratio test matches: {len(good_ratio)}')
if len(matches_cross) > 0:
    distances = [m.distance for m in matches_cross]
    print(f'CrossCheck distance range: {min(distances):.1f} - {max(distances):.1f}')

cv2.imshow('Brute-Force Matching', result)
```
