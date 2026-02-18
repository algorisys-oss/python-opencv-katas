---
slug: 62-fast-keypoints
title: FAST Keypoint Detection
level: advanced
concepts: [cv2.FastFeatureDetector, real-time corners, threshold]
prerequisites: [61-shi-tomasi-corners]
---

## What Problem Are We Solving?

Harris and Shi-Tomasi corners work well but involve computing gradients, structure tensors, and eigenvalues at every pixel — this is computationally expensive. **FAST** (Features from Accelerated Segment Test) takes a radically different approach: it detects corners by simply comparing the intensity of a candidate pixel against its surrounding **circle of 16 pixels**. This makes FAST **many times faster** than traditional detectors, making it the go-to choice for real-time applications like visual SLAM, augmented reality, and drone navigation.

## How FAST Works

For each candidate pixel `p` with intensity `Ip`, FAST examines 16 pixels arranged in a circle of radius 3 around `p`. A pixel is classified as a corner if there exists a **contiguous arc** of N or more pixels in the circle that are all brighter than `Ip + threshold` or all darker than `Ip - threshold`.

```
        16  1  2
     15          3
   14              4
   13     [p]      5
   12              6
     11          7
        10  9  8
```

The standard FAST variant uses N=12 (called FAST-12), meaning 12 out of 16 contiguous pixels must pass the test. FAST-9 (N=9) is also common and detects more keypoints.

## The Speed Advantage

FAST uses a **high-speed test** as a first check: it examines pixels 1, 5, 9, and 13 (the cardinal directions). If at least 3 of these 4 are not all brighter or all darker than the center, the pixel cannot be a FAST corner and is immediately rejected. This eliminates the vast majority of pixels without checking all 16 neighbors.

```python
# FAST is designed for speed — typical timings:
# Harris:      ~20ms on a 640x480 image
# Shi-Tomasi:  ~15ms
# FAST:        ~2ms
```

## Using cv2.FastFeatureDetector

```python
fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
keypoints = fast.detect(gray, None)
```

| Parameter | Meaning |
|---|---|
| `threshold` | Intensity difference threshold for the segment test (default 10) |
| `nonmaxSuppression` | If `True`, suppresses adjacent detections to get cleaner results (default `True`) |

The `detect()` method returns a list of `cv2.KeyPoint` objects, each containing `.pt` (x, y coordinates), `.size`, `.response` (corner score), and other attributes.

## The Threshold Parameter

The threshold controls how much brighter or darker surrounding pixels must be compared to the center pixel. It directly affects sensitivity:

```python
# Low threshold: detects many keypoints, including weak ones (noisy)
fast_sensitive = cv2.FastFeatureDetector_create(threshold=5)
kp_many = fast_sensitive.detect(gray, None)

# High threshold: detects only strong, clear corners
fast_selective = cv2.FastFeatureDetector_create(threshold=40)
kp_few = fast_selective.detect(gray, None)
```

For typical images, values between 10 and 30 work well. Very low thresholds on noisy images will produce thousands of spurious keypoints.

## Non-Maximum Suppression

Without non-maximum suppression, FAST tends to detect **multiple adjacent keypoints** at the same corner. Enabling it keeps only the strongest response in each local neighborhood:

```python
# Without NMS: many clustered keypoints
fast_no_nms = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)
kp_no_nms = fast_no_nms.detect(gray, None)

# With NMS: cleaner, sparser keypoints
fast_nms = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
kp_nms = fast_nms.detect(gray, None)
```

Always use `nonmaxSuppression=True` unless you have a specific reason not to.

## Choosing the FAST Type

OpenCV supports three FAST variants that differ in how many contiguous pixels are required:

```python
fast = cv2.FastFeatureDetector_create()

# TYPE_9_16: 9 out of 16 contiguous pixels (most keypoints)
fast.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

# TYPE_7_12: 7 out of 12 contiguous pixels
fast.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

# TYPE_5_8: 5 out of 8 contiguous pixels (fewest keypoints)
fast.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
```

## Drawing Keypoints

OpenCV provides `cv2.drawKeypoints()` to visualize detected keypoints:

```python
img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
```

## Tips & Common Mistakes

- FAST detects **corners only** — it does not compute descriptors. To match features, you need to pair FAST with a descriptor like ORB or BRIEF.
- FAST is not scale-invariant and not rotation-invariant by itself. ORB wraps FAST with a scale pyramid and orientation computation to add these properties.
- The threshold is an **absolute intensity difference**, not relative. Images with low contrast may need a lower threshold.
- Always use `nonmaxSuppression=True` for cleaner results. The only exception is if you want to study the raw detection behavior.
- FAST keypoints on uniform or textureless regions (like a white wall) are almost always noise — increase the threshold to suppress them.
- The `detect()` method takes an optional mask as the second argument. Pass a binary mask to restrict detection to a region of interest.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with various features
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (60, 60, 60)

# Draw shapes with corners and edges
cv2.rectangle(img, (30, 30), (170, 170), (255, 255, 255), 2)
cv2.rectangle(img, (200, 40), (330, 150), (180, 180, 180), -1)
pts = np.array([[420, 30], [560, 30], [560, 170], [490, 170], [490, 100], [420, 100]], dtype=np.int32)
cv2.polylines(img, [pts], True, (200, 220, 255), 2)
cv2.circle(img, (100, 300), 60, (255, 200, 150), 2)
cv2.putText(img, 'FAST', (230, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
pts_tri = np.array([[430, 230], [560, 370], [430, 370]], dtype=np.int32)
cv2.fillPoly(img, [pts_tri], (100, 200, 100))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Different thresholds ---
fast_low = cv2.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True)
fast_med = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
fast_high = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)

kp_low = fast_low.detect(gray, None)
kp_med = fast_med.detect(gray, None)
kp_high = fast_high.detect(gray, None)

# --- NMS comparison ---
fast_no_nms = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=False)
kp_no_nms = fast_no_nms.detect(gray, None)

# --- Draw keypoints on separate copies ---
img_low = cv2.drawKeypoints(img, kp_low, None, color=(0, 0, 255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_med = cv2.drawKeypoints(img, kp_med, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img_high = cv2.drawKeypoints(img, kp_high, None, color=(255, 0, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img_no_nms = cv2.drawKeypoints(img, kp_no_nms, None, color=(0, 255, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_low, f'thresh=5 ({len(kp_low)} kp)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_med, f'thresh=20 ({len(kp_med)} kp)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_high, f'thresh=40 ({len(kp_high)} kp)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img_no_nms, f'No NMS ({len(kp_no_nms)} kp)', (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Build comparison grid
top_row = np.hstack([img_low, img_med])
bottom_row = np.hstack([img_high, img_no_nms])
result = np.vstack([top_row, bottom_row])

print(f'Keypoints (threshold=5):  {len(kp_low)}')
print(f'Keypoints (threshold=20): {len(kp_med)}')
print(f'Keypoints (threshold=40): {len(kp_high)}')
print(f'Keypoints (no NMS):       {len(kp_no_nms)}')
print(f'Keypoints (with NMS):     {len(kp_med)}')

cv2.imshow('FAST Keypoint Detection', result)
```
