---
slug: 25-custom-kernels
title: Custom Kernels & Convolution
level: intermediate
concepts: [cv2.filter2D, kernel design, convolution]
prerequisites: [24-image-sharpening]
---

## What Problem Are We Solving?

Every image filter you've used so far — blur, sharpen, edge detection — is powered by **convolution** with a **kernel** (also called a filter or mask). Understanding how convolution works and how to design your own kernels gives you the power to create any spatial filter you can imagine: custom edge detectors, emboss effects, artistic filters, and more.

## What Is Convolution?

Convolution is a mathematical operation that slides a small matrix (the **kernel**) across every pixel of an image. At each position, it computes a weighted sum of the pixel values under the kernel:

```
For a 3x3 kernel K applied at pixel (x, y):

output(x, y) = sum of [ K(i, j) * image(x+i, y+j) ] for all i, j in kernel
```

Step by step for a single pixel:

```
Image region:          Kernel:
| 10  20  30 |       | -1  0  1 |
| 40  50  60 |   *   | -2  0  2 |
| 70  80  90 |       | -1  0  1 |

Result = (-1*10) + (0*20) + (1*30)
       + (-2*40) + (0*50) + (2*60)
       + (-1*70) + (0*80) + (1*90)
       = -10 + 0 + 30 - 80 + 0 + 120 - 70 + 0 + 90
       = 80
```

This process repeats for **every pixel** in the image, producing the output.

## Applying Custom Kernels with cv2.filter2D()

```python
output = cv2.filter2D(img, ddepth=-1, kernel=kernel)
```

| Parameter | Meaning |
|---|---|
| `img` | Input image |
| `ddepth` | Output depth. `-1` means same as input. Use `cv2.CV_64F` for signed results |
| `kernel` | NumPy array defining the convolution kernel |

The kernel must be a 2D NumPy array, typically with `dtype=np.float32`.

## The Identity Kernel

The identity kernel produces an output **identical** to the input — it's the "do nothing" kernel:

```python
identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)

result = cv2.filter2D(img, -1, identity)  # result == img
```

Only the center pixel has weight 1; all others are 0. This is the starting point for understanding all other kernels — every other kernel is a modification of identity.

## Edge Detection Kernels

Edge detection kernels highlight rapid changes in intensity. They have both positive and negative values that cancel out in flat regions but produce strong responses at edges:

```python
# Horizontal edge detection
horizontal_edges = np.array([[-1, -1, -1],
                              [ 0,  0,  0],
                              [ 1,  1,  1]], dtype=np.float32)

# Vertical edge detection
vertical_edges = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)

# Laplacian (detects edges in all directions)
laplacian = np.array([[ 0,  1,  0],
                       [ 1, -4,  1],
                       [ 0,  1,  0]], dtype=np.float32)
```

Edge kernels typically sum to **zero** — this means flat regions produce zero output (no edge), while transitions produce strong positive or negative values.

## Emboss Kernel

The emboss kernel creates a 3D raised/sunken effect by detecting edges at an angle and adding a bias:

```python
# Emboss from top-left
emboss = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]], dtype=np.float32)

embossed = cv2.filter2D(img, -1, emboss)

# Add 128 to shift from signed to visible range
embossed_visible = cv2.filter2D(img, cv2.CV_64F, emboss)
embossed_visible = np.clip(embossed_visible + 128, 0, 255).astype(np.uint8)
```

Different emboss directions:

```python
# Emboss from top-right
emboss_tr = np.array([[0, -1, -2],
                      [1,  1, -1],
                      [2,  1,  0]], dtype=np.float32)

# Emboss from bottom
emboss_bottom = np.array([[ 0,  1,  2],
                          [-1,  1,  1],
                          [-2, -1,  0]], dtype=np.float32)
```

## Designing Your Own Kernels

Rules of thumb for kernel design:

| Goal | Kernel Property |
|---|---|
| **Blur/smooth** | All positive values, sum = 1 |
| **Sharpen** | Center > sum of others, sum = 1 |
| **Edge detect** | Mix of positive and negative, sum = 0 |
| **Emboss** | Asymmetric positive/negative, sum ~ 1 |
| **Brighten** | All positive, sum > 1 |
| **Darken** | All positive, sum < 1 |

```python
# Custom blur: weighted center-heavy average
custom_blur = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=np.float32) / 16.0

# Custom edge enhancer: slight sharpen + edge emphasis
custom_enhance = np.array([[-0.5, -1, -0.5],
                           [-1,    7, -1],
                           [-0.5, -1, -0.5]], dtype=np.float32)
```

## Larger Kernels

Kernels don't have to be 3x3. Larger kernels affect wider neighborhoods:

```python
# 5x5 Gaussian-like blur
gauss_5x5 = np.array([[1,  4,  6,  4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]], dtype=np.float32) / 256.0

# 5x5 strong edge detection
big_edge = np.array([[-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, 24, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1]], dtype=np.float32)
```

## Tips & Common Mistakes

- Always use `dtype=np.float32` for kernels. Integer arrays can produce unexpected results due to truncation.
- Edge detection kernels produce **signed** values (positive and negative). Use `ddepth=cv2.CV_64F` to preserve negative values, then convert: `result = np.absolute(result).astype(np.uint8)`.
- With `ddepth=-1` and `uint8` input, negative output values are clipped to 0. This is fine for blur/sharpen but loses half the information for edge detection.
- The kernel sum determines brightness: sum=1 preserves brightness, sum=0 produces a dark image with edges, sum>1 brightens.
- Convolution at image borders uses zero-padding by default. You can change this with the `borderType` parameter of `cv2.filter2D()`.
- Larger kernels are slower: computation scales with kernel area. A 7x7 kernel is 5.4x slower than a 3x3.
- `cv2.filter2D()` actually performs **correlation**, not convolution (kernel is not flipped). For symmetric kernels, correlation and convolution are identical.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with various features
img = np.zeros((250, 350, 3), dtype=np.uint8)
img[:] = (180, 180, 180)

# Add shapes and detail
cv2.rectangle(img, (20, 20), (120, 110), (200, 70, 40), -1)
cv2.circle(img, (230, 65), 45, (40, 160, 70), -1)
cv2.line(img, (20, 140), (330, 140), (60, 60, 60), 2)
cv2.line(img, (175, 20), (175, 230), (60, 60, 60), 2)
for i in range(150, 240, 6):
    cv2.circle(img, (80, i + 30), 2, (80, 80, 80), -1)
cv2.putText(img, 'Kernels', (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

# --- Define custom kernels ---
# Identity (no change)
identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)

# Edge detection (Laplacian-style)
edge_kernel = np.array([[ 0,  1,  0],
                        [ 1, -4,  1],
                        [ 0,  1,  0]], dtype=np.float32)

# Emboss
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)

# Sharpen
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)

# Custom ridge detection
ridge_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype=np.float32)

# --- Apply kernels ---
identity_out = cv2.filter2D(img, -1, identity)

# Edge detection with proper depth handling
edges_raw = cv2.filter2D(img, cv2.CV_64F, edge_kernel)
edges = np.clip(np.absolute(edges_raw), 0, 255).astype(np.uint8)

# Emboss with offset for visibility
emboss_raw = cv2.filter2D(img, cv2.CV_64F, emboss_kernel)
embossed = np.clip(emboss_raw + 128, 0, 255).astype(np.uint8)

sharpened = cv2.filter2D(img, -1, sharpen_kernel)

# Ridge detection
ridge_raw = cv2.filter2D(img, cv2.CV_64F, ridge_kernel)
ridges = np.clip(np.absolute(ridge_raw), 0, 255).astype(np.uint8)

# Label helper
def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# Row 1: basic kernels
row1 = np.hstack([
    label(img, 'Original'),
    label(identity_out, 'Identity'),
    label(edges, 'Edge detect'),
])

# Row 2: artistic kernels
row2 = np.hstack([
    label(embossed, 'Emboss'),
    label(sharpened, 'Sharpen'),
    label(ridges, 'Ridge detect'),
])

result = np.vstack([row1, row2])

print('Kernel sums:')
print(f'  Identity: {identity.sum():.1f} (preserves brightness)')
print(f'  Edge:     {edge_kernel.sum():.1f} (zero = dark output)')
print(f'  Emboss:   {emboss_kernel.sum():.1f} (preserves brightness)')
print(f'  Sharpen:  {sharpen_kernel.sum():.1f} (preserves brightness)')
print(f'  Ridge:    {ridge_kernel.sum():.1f} (zero = dark output)')

cv2.imshow('Custom Kernels & Convolution', result)
```
