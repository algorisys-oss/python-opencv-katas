---
slug: 54-fourier-transform
title: Fourier Transform
level: intermediate
concepts: [cv2.dft, magnitude spectrum, frequency domain]
prerequisites: [02-color-spaces]
---

## What Problem Are We Solving?

Every image can be thought of in two ways: as a grid of pixel values (the **spatial domain**) or as a combination of waves at different frequencies (the **frequency domain**). The Fourier Transform converts an image from spatial to frequency domain, revealing patterns that are invisible when looking at pixels directly. This is fundamental for understanding noise, designing filters, analyzing textures, and compressing images.

## What Does "Frequency" Mean for Images?

In images, **frequency** refers to how quickly pixel values change:

- **Low frequency** = smooth, gradual changes (large uniform regions, soft gradients, backgrounds).
- **High frequency** = rapid, sharp changes (edges, fine textures, noise, text).

A plain wall is mostly low frequency. A zebra's stripes are high frequency. Most real images contain a mix of both.

## Computing the DFT with cv2.dft()

OpenCV's `cv2.dft()` computes the Discrete Fourier Transform. The input must be `float32`, and the output is a 2-channel array (real and imaginary parts):

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft_input = np.float32(gray)
dft = cv2.dft(dft_input, flags=cv2.DFT_COMPLEX_OUTPUT)
```

| Parameter | Meaning |
|---|---|
| `dft_input` | Grayscale image as `float32` |
| `flags` | `cv2.DFT_COMPLEX_OUTPUT` returns both real and imaginary parts |

The output `dft` has shape `(H, W, 2)` where channel 0 is the real part and channel 1 is the imaginary part.

## Computing the Magnitude Spectrum

The raw DFT output isn't visually meaningful. To visualize it, compute the **magnitude** and apply a log transform:

```python
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum = 20 * np.log(magnitude + 1)  # Log scale for visibility
```

- `cv2.magnitude(real, imag)` computes `sqrt(real^2 + imag^2)` per pixel.
- The `+1` prevents `log(0)`.
- The `20 *` scaling spreads the values for better visualization.

## Shifting Zero Frequency to Center

By default, `cv2.dft()` places the zero-frequency component (DC component, the average brightness) at the **top-left corner**. This is hard to interpret. `np.fft.fftshift()` moves it to the **center**:

```python
dft_shift = np.fft.fftshift(dft)
```

After shifting:
- The **center** of the spectrum = low frequencies (slow changes).
- The **edges** of the spectrum = high frequencies (fast changes, edges, noise).
- Bright spots indicate dominant frequency components.

## Reading the Magnitude Spectrum

The magnitude spectrum is a 2D image where:

```python
# Center = DC component (average brightness of the image)
# Moving outward = increasing frequency
# Horizontal direction = vertical patterns in the image
# Vertical direction = horizontal patterns in the image
```

A bright horizontal line through the center means the image has strong **vertical edges**. A bright vertical line means strong **horizontal edges**. Bright dots indicate repeating patterns at specific frequencies.

## Optimal DFT Size

DFT computation is fastest when the image dimensions are powers of 2, or products of small primes. OpenCV provides a helper:

```python
rows, cols = gray.shape
optimal_rows = cv2.getOptimalDFTSize(rows)
optimal_cols = cv2.getOptimalDFTSize(cols)
padded = np.zeros((optimal_rows, optimal_cols), dtype=np.float32)
padded[:rows, :cols] = gray
```

This zero-pads the image to an optimal size for faster computation.

## Using NumPy's FFT (Alternative)

NumPy provides a simpler interface that handles complex numbers directly:

```python
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift)
magnitude_spectrum = 20 * np.log(magnitude + 1)
```

This is equivalent to the OpenCV approach but uses Python complex numbers instead of a 2-channel array.

## Tips & Common Mistakes

- The input to `cv2.dft()` must be `float32`. Passing `uint8` causes an error.
- Always use `cv2.DFT_COMPLEX_OUTPUT` flag to get both real and imaginary parts -- you need both for the magnitude.
- Without the log transform, the magnitude spectrum appears almost entirely black because the DC component is vastly larger than all other frequencies.
- `np.fft.fftshift()` works on the OpenCV 2-channel array as well as on NumPy complex arrays.
- The Fourier Transform is **invertible** -- you can go back to the spatial domain with `cv2.idft()` or `np.fft.ifft2()`.
- For color images, apply the DFT to each channel separately, or convert to grayscale first. Most frequency analysis is done on grayscale.
- The magnitude spectrum is symmetric -- the left half mirrors the right half. This is a property of real-valued inputs.

## Starter Code

```python
import cv2
import numpy as np

# Create test images with different frequency content
h, w = 300, 300

# Image 1: Low frequency -- smooth gradient
gradient = np.zeros((h, w), dtype=np.uint8)
for x in range(w):
    gradient[:, x] = int(255 * x / w)

# Image 2: High frequency -- fine stripes
stripes = np.zeros((h, w), dtype=np.uint8)
for x in range(w):
    stripes[:, x] = 255 if (x // 4) % 2 == 0 else 0

# Image 3: Mixed -- shapes with edges on smooth background
mixed = np.full((h, w), 128, dtype=np.uint8)
cv2.rectangle(mixed, (50, 50), (150, 150), 255, -1)
cv2.circle(mixed, (220, 200), 60, 0, -1)
cv2.line(mixed, (0, 280), (300, 250), 200, 2)

# --- Compute DFT and magnitude spectrum for each image ---
def compute_spectrum(gray_img):
    dft_input = np.float32(gray_img)
    dft = cv2.dft(dft_input, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    spectrum = 20 * np.log(magnitude + 1)
    # Normalize to 0-255 for display
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return spectrum

spec_gradient = compute_spectrum(gradient)
spec_stripes = compute_spectrum(stripes)
spec_mixed = compute_spectrum(mixed)

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
images = [gradient, spec_gradient, stripes, spec_stripes, mixed, spec_mixed]
labels = ['Gradient', 'Spectrum', 'Stripes', 'Spectrum', 'Mixed', 'Spectrum']

for img_item, label in zip(images, labels):
    cv2.putText(img_item, label, (5, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# Build grid: each row = image + its spectrum
row1 = np.hstack([gradient, spec_gradient])
row2 = np.hstack([stripes, spec_stripes])
row3 = np.hstack([mixed, spec_mixed])
result = np.vstack([row1, row2, row3])

# Convert to BGR for display
result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

print(f'Image size: {h}x{w}')
print(f'Optimal DFT size: {cv2.getOptimalDFTSize(h)}x{cv2.getOptimalDFTSize(w)}')
print(f'Gradient spectrum range: [{spec_gradient.min()}, {spec_gradient.max()}]')
print(f'Stripes spectrum range: [{spec_stripes.min()}, {spec_stripes.max()}]')
print(f'Mixed spectrum range: [{spec_mixed.min()}, {spec_mixed.max()}]')

cv2.imshow('Fourier Transform', result_bgr)
```
