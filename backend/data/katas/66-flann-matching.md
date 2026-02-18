---
slug: 66-flann-matching
title: FLANN Matching
level: advanced
concepts: [cv2.FlannBasedMatcher, KD-tree, faster matching]
prerequisites: [65-brute-force-matching]
---

## What Problem Are We Solving?

Brute-force matching compares every descriptor in image 1 against every descriptor in image 2, giving O(N*M) complexity. For small feature sets (a few hundred), this is fine. But when you have **thousands of descriptors** or need to match in **real time**, brute-force becomes a bottleneck. **FLANN** (Fast Library for Approximate Nearest Neighbors) uses spatial data structures like **KD-trees** and **hierarchical clustering** to find approximate nearest neighbors much faster — often 10x or more.

## FLANN vs Brute Force

| Aspect | Brute Force | FLANN |
|---|---|---|
| Accuracy | Exact (guaranteed best match) | Approximate (may miss the best match) |
| Speed (small sets) | Fast | Similar or slower (setup overhead) |
| Speed (large sets) | Slow (O(N*M)) | Much faster (O(N*log(M))) |
| Configuration | Simple | Requires index/search params |

FLANN trades a small amount of matching accuracy for a large speed improvement. In practice, the approximation error is negligible for most applications.

## Index Parameters for Different Descriptor Types

FLANN requires an **index** configuration that depends on the descriptor type. This is the most important detail to get right:

```python
# For float descriptors (SIFT, SURF) — use KD-tree
index_params_sift = dict(algorithm=1, trees=5)  # algorithm 1 = FLANN_INDEX_KDTREE

# For binary descriptors (ORB, BRIEF) — use LSH (Locality Sensitive Hashing)
index_params_orb = dict(algorithm=6,             # algorithm 6 = FLANN_INDEX_LSH
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
```

| Algorithm | Constant | Use With |
|---|---|---|
| `1` (FLANN_INDEX_KDTREE) | KD-tree | SIFT, SURF (float descriptors) |
| `6` (FLANN_INDEX_LSH) | Locality Sensitive Hashing | ORB, BRIEF (binary descriptors) |

> **Critical:** Using KD-tree with binary descriptors or LSH with float descriptors will produce wrong results or errors.

## Search Parameters

The `search_params` dictionary controls how thoroughly FLANN searches the index:

```python
# checks: how many leaf nodes to examine (higher = more accurate, slower)
search_params = dict(checks=50)   # Default — good balance
search_params = dict(checks=100)  # More accurate, slower
search_params = dict(checks=10)   # Faster, less accurate
```

More checks means FLANN explores more of the index tree, approaching brute-force accuracy at the cost of speed.

## Creating and Using FlannBasedMatcher

```python
# For SIFT descriptors
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find 2 nearest matches for ratio test
matches = flann.knnMatch(des1, des2, k=2)
```

## Lowe's Ratio Test with FLANN

Just like with brute-force, apply the ratio test to filter good matches:

```python
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
```

The ratio test is especially important with FLANN because approximate matching can sometimes return spurious nearest neighbors. The ratio test catches these cases.

## Complete FLANN Example with SIFT

```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN parameters for SIFT
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Ratio test
good = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)
```

## Complete FLANN Example with ORB

```python
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# FLANN parameters for ORB (binary)
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for pair in matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)
```

## Tips & Common Mistakes

- The **most common mistake** is using the wrong index parameters for your descriptor type. KD-tree (algorithm=1) only works with float descriptors. LSH (algorithm=6) only works with binary descriptors.
- FLANN requires descriptors to be `np.float32` when using KD-tree. If your SIFT descriptors are somehow a different type, convert with `des = np.float32(des)`.
- Always check `len(pair) == 2` before unpacking ratio test pairs. If a descriptor has fewer than 2 neighbors (can happen with small descriptor sets), accessing `pair[1]` will raise an IndexError.
- For small descriptor sets (< 500), brute-force can actually be faster than FLANN because FLANN has setup overhead for building the index.
- The `trees` parameter in KD-tree index controls the number of randomized trees. More trees = more accuracy and memory, slower build time. 5 is a good default.
- FLANN does not support `crossCheck` like BFMatcher. Use the ratio test instead for filtering.
- If you get a FLANN error about descriptor types, ensure your descriptors are the correct numpy dtype (`float32` for SIFT, `uint8` for ORB).

## Starter Code

```python
import cv2
import numpy as np

# Create two related images with shared features
img1 = np.zeros((300, 400, 3), dtype=np.uint8)
img1[:] = (50, 50, 50)

# Draw identifiable features
cv2.rectangle(img1, (20, 20), (120, 120), (255, 255, 255), 2)
cv2.rectangle(img1, (30, 30), (110, 110), (200, 200, 200), 1)
cv2.circle(img1, (250, 70), 40, (200, 200, 255), 2)
cv2.circle(img1, (250, 70), 20, (150, 150, 255), -1)
cv2.putText(img1, 'FLANN', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
pts = np.array([[280, 160], [380, 160], [380, 280], [320, 280], [320, 220], [280, 220]], dtype=np.int32)
cv2.fillPoly(img1, [pts], (100, 180, 100))
cv2.polylines(img1, [pts], True, (150, 220, 150), 2)

# Add texture dots
rng = np.random.RandomState(42)
for _ in range(30):
    x, y = rng.randint(0, 400), rng.randint(0, 300)
    r = rng.randint(100, 255)
    cv2.circle(img1, (x, y), rng.randint(2, 5), (r, r, r), -1)

# Transform to create img2
rows, cols = img1.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 12, 0.95)
M[0, 2] += 20
M[1, 2] += 15
img2 = cv2.warpAffine(img1, M, (cols, rows), borderValue=(50, 50, 50))
noise = np.random.randint(-10, 10, img2.shape, dtype=np.int16)
img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- SIFT + FLANN (KD-tree) ---
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

index_params_sift = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann_sift = cv2.FlannBasedMatcher(index_params_sift, search_params)

matches_sift = flann_sift.knnMatch(des1_sift, des2_sift, k=2)

good_sift = []
for pair in matches_sift:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_sift.append(m)

# --- ORB + Brute Force (for comparison) ---
orb = cv2.ORB_create(nfeatures=500)
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches_orb = bf.knnMatch(des1_orb, des2_orb, k=2)

good_orb = []
for pair in matches_orb:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_orb.append(m)

# --- Draw matches ---
img_flann = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, good_sift[:25], None,
                            matchColor=(0, 255, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_bf = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, good_orb[:25], None,
                         matchColor=(0, 255, 255),
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_flann, f'FLANN+SIFT ({len(good_sift)} good matches)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_bf, f'BF+ORB ({len(good_orb)} good matches)', (5, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Stack vertically
result = np.vstack([img_flann, img_bf])

print(f'SIFT keypoints: img1={len(kp1_sift)}, img2={len(kp2_sift)}')
print(f'FLANN+SIFT good matches: {len(good_sift)}')
print(f'ORB keypoints: img1={len(kp1_orb)}, img2={len(kp2_orb)}')
print(f'BF+ORB good matches: {len(good_orb)}')

cv2.imshow('FLANN Matching', result)
```
