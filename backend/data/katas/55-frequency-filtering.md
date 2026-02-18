---
slug: 55-frequency-filtering
title: Frequency Domain Filtering
level: intermediate
concepts: [low-pass filter, high-pass filter, frequency masking]
prerequisites: [54-fourier-transform]
---

## What Problem Are We Solving?

Spatial domain filters (like `cv2.blur` or `cv2.Canny`) work by sliding kernels across pixels. But some filtering tasks are more intuitive in the **frequency domain**: want to remove noise? Block high frequencies. Want to sharpen edges? Block low frequencies. Frequency domain filtering gives you direct, surgical control over which spatial patterns survive and which get removed.

## The Frequency Filtering Pipeline

The general workflow for frequency domain filtering is:

1. Convert the image to frequency domain using DFT.
2. Shift zero frequency to center.
3. Multiply with a filter mask (the frequency filter).
4. Shift back.
5. Apply inverse DFT to get the filtered image.

```python
# Forward: spatial -> frequency
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Apply filter (element-wise multiplication)
filtered = dft_shift * mask

# Inverse: frequency -> spatial
f_ishift = np.fft.ifftshift(filtered)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
```

## Creating a Circular Low-Pass Filter

A low-pass filter **keeps low frequencies** (smooth regions) and **blocks high frequencies** (edges, noise). This produces a blurring effect. The filter is a circular mask centered on the zero-frequency point:

```python
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
radius = 30  # Cutoff frequency

mask = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= radius:
            mask[i, j] = 1
```

A more efficient way using NumPy:

```python
Y, X = np.ogrid[:rows, :cols]
dist = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
mask = np.zeros((rows, cols, 2), np.float32)
mask[dist <= radius] = 1
```

- Small radius = aggressive filtering, very blurry result.
- Large radius = mild filtering, subtle smoothing.

## Creating a Circular High-Pass Filter

A high-pass filter does the opposite: it **blocks low frequencies** and **keeps high frequencies**. This extracts edges and fine details:

```python
mask = np.ones((rows, cols, 2), np.float32)
mask[dist <= radius] = 0
```

This is simply `1 - low_pass_mask`. The result highlights edges while removing smooth regions.

## Applying the Filter and Inverse DFT

After multiplying the shifted DFT with the mask, convert back to the spatial domain:

```python
# Apply the frequency filter
filtered_dft = dft_shift * mask

# Inverse shift
f_ishift = np.fft.ifftshift(filtered_dft)

# Inverse DFT
img_back = cv2.idft(f_ishift)

# Compute magnitude (the actual pixel values)
result = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
```

The `cv2.idft()` returns a 2-channel array (real and imaginary). Taking the magnitude gives you the final grayscale image.

## Band-Pass Filtering

A band-pass filter keeps only frequencies within a specific range -- blocking both very low and very high frequencies:

```python
mask = np.zeros((rows, cols, 2), np.float32)
mask[(dist >= inner_radius) & (dist <= outer_radius)] = 1
```

This is useful for isolating texture patterns at a specific scale while removing both background and noise.

## Effect of Cutoff Radius

The radius of the circular mask directly controls the filtering strength:

```python
# Small radius low-pass: aggressive blur, only very smooth features survive
radius = 10

# Medium radius low-pass: moderate blur, major structures preserved
radius = 30

# Large radius low-pass: gentle blur, most detail preserved
radius = 80
```

## Tips & Common Mistakes

- The mask must have **2 channels** (matching the DFT output's real and imaginary parts). A single-channel mask won't broadcast correctly.
- Always use `np.fft.fftshift()` before filtering and `np.fft.ifftshift()` after. Skipping either one produces garbled results.
- The ideal (hard cutoff) circular filter can cause **ringing artifacts** -- visible ripples around edges. Gaussian or Butterworth filters have smoother transitions and fewer artifacts.
- After `cv2.idft()`, use `cv2.magnitude()` to get the actual image values. Don't just take channel 0 (the real part).
- Normalize the result to 0-255 for display, since `cv2.idft()` output may have arbitrary value ranges.
- Low-pass filtering in the frequency domain is mathematically equivalent to convolution with a sinc function in the spatial domain -- but it can be faster for very large kernels.
- For color images, convert to grayscale or process each channel separately.

## Starter Code

```python
import cv2
import numpy as np

# Create a test image with both smooth regions and sharp edges
h, w = 300, 300
img = np.full((h, w), 128, dtype=np.uint8)

# Smooth gradient region
for x in range(100, 200):
    img[20:120, x] = int(80 + 170 * (x - 100) / 100)

# Sharp edges: rectangles and lines
cv2.rectangle(img, (30, 150), (130, 250), 255, -1)
cv2.rectangle(img, (160, 160), (270, 270), 40, -1)
cv2.line(img, (0, 140), (300, 140), 0, 2)
cv2.circle(img, (220, 80), 40, 200, -1)

# Add some noise for high-frequency content
noise = np.random.randint(-20, 20, (h, w), dtype=np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# --- Compute DFT ---
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# --- Create distance matrix from center ---
crow, ccol = h // 2, w // 2
Y, X = np.ogrid[:h, :w]
dist = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)

# --- Low-pass filter (keep low frequencies = blur) ---
lp_radius = 30
lp_mask = np.zeros((h, w, 2), np.float32)
lp_mask[dist <= lp_radius] = 1

lp_filtered = dft_shift * lp_mask
lp_ishift = np.fft.ifftshift(lp_filtered)
lp_result = cv2.idft(lp_ishift)
lp_result = cv2.magnitude(lp_result[:, :, 0], lp_result[:, :, 1])
lp_result = cv2.normalize(lp_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- High-pass filter (keep high frequencies = edges) ---
hp_radius = 30
hp_mask = np.ones((h, w, 2), np.float32)
hp_mask[dist <= hp_radius] = 0

hp_filtered = dft_shift * hp_mask
hp_ishift = np.fft.ifftshift(hp_filtered)
hp_result = cv2.idft(hp_ishift)
hp_result = cv2.magnitude(hp_result[:, :, 0], hp_result[:, :, 1])
hp_result = cv2.normalize(hp_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Band-pass filter ---
bp_inner = 15
bp_outer = 60
bp_mask = np.zeros((h, w, 2), np.float32)
bp_mask[(dist >= bp_inner) & (dist <= bp_outer)] = 1

bp_filtered = dft_shift * bp_mask
bp_ishift = np.fft.ifftshift(bp_filtered)
bp_result = cv2.idft(bp_ishift)
bp_result = cv2.magnitude(bp_result[:, :, 0], bp_result[:, :, 1])
bp_result = cv2.normalize(bp_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Visualize the filter masks (single channel for display) ---
lp_mask_vis = (lp_mask[:, :, 0] * 255).astype(np.uint8)
hp_mask_vis = (hp_mask[:, :, 0] * 255).astype(np.uint8)
bp_mask_vis = (bp_mask[:, :, 0] * 255).astype(np.uint8)

# Compute magnitude spectrum for display
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
spectrum = 20 * np.log(magnitude + 1)
spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
label_pairs = [
    (img, 'Original'), (spectrum, 'Spectrum'),
    (lp_mask_vis, f'LP Mask r={lp_radius}'), (lp_result, 'Low-Pass'),
    (hp_mask_vis, f'HP Mask r={hp_radius}'), (hp_result, 'High-Pass'),
    (bp_mask_vis, f'BP Mask {bp_inner}-{bp_outer}'), (bp_result, 'Band-Pass'),
]

labeled = []
for image, label in label_pairs:
    out = image.copy()
    cv2.putText(out, label, (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    labeled.append(out)

# Build 4x2 grid
row1 = np.hstack([labeled[0], labeled[1]])
row2 = np.hstack([labeled[2], labeled[3]])
row3 = np.hstack([labeled[4], labeled[5]])
row4 = np.hstack([labeled[6], labeled[7]])
result = np.vstack([row1, row2, row3, row4])

result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

print(f'Image size: {h}x{w}')
print(f'Low-pass radius: {lp_radius} (keeps {int(np.pi * lp_radius**2)} freq bins)')
print(f'High-pass radius: {hp_radius} (blocks {int(np.pi * hp_radius**2)} freq bins)')
print(f'Band-pass range: {bp_inner}-{bp_outer}')

cv2.imshow('Frequency Domain Filtering', result_bgr)
```
