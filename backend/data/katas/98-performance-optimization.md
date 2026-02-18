---
slug: 98-performance-optimization
title: Performance Optimization
level: advanced
concepts: [cv2.UMat, vectorized NumPy, timing, avoiding loops]
prerequisites: [03-pixel-access]
---

## What Problem Are We Solving?

OpenCV operations on large images can be slow, especially when you chain many processing steps together. Worse, **writing pixel-level loops in Python** is catastrophically slow compared to vectorized operations. This kata covers practical techniques to make your OpenCV pipelines run faster: using `cv2.UMat` for GPU-accelerated processing, replacing Python loops with vectorized NumPy operations, and proper benchmarking with `cv2.getTickCount`.

Understanding performance is essential when moving from single-image experiments to real-time video processing where you need 30+ frames per second.

## Benchmarking with cv2.getTickCount

Before optimizing, you need to measure. `cv2.getTickCount()` returns a high-resolution tick counter, and `cv2.getTickFrequency()` converts ticks to seconds:

```python
t1 = cv2.getTickCount()
# ... operation to time ...
t2 = cv2.getTickCount()
elapsed_ms = (t2 - t1) / cv2.getTickFrequency() * 1000
```

This is more reliable than `time.time()` for short operations because it uses a high-resolution hardware counter. You can also use Python's `time.perf_counter()` for similar precision.

## Python Loops vs Vectorized NumPy

The single biggest performance mistake in OpenCV Python code is writing explicit pixel loops. Python's interpreter overhead makes per-pixel loops thousands of times slower than the equivalent vectorized operation:

```python
# SLOW: Python loop over every pixel
for y in range(h):
    for x in range(w):
        img[y, x] = 255 - img[y, x]

# FAST: Vectorized NumPy operation (same result)
img = 255 - img
```

NumPy operations are implemented in C and operate on entire arrays at once. The rule of thumb: if you find yourself writing a `for` loop over pixels, there is almost certainly a vectorized alternative.

## Common Vectorization Patterns

Many seemingly complex operations can be expressed without loops:

**Threshold manually** (equivalent to `cv2.threshold`):

```python
# Loop version (slow)
for y in range(h):
    for x in range(w):
        result[y, x] = 255 if gray[y, x] > 128 else 0

# Vectorized version (fast)
result = np.where(gray > 128, 255, 0).astype(np.uint8)
```

**Apply a brightness/contrast adjustment**:

```python
# Vectorized: multiply and clip in one step
adjusted = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
```

**Compute pixel-wise maximum of two images**:

```python
combined = np.maximum(img1, img2)
```

## GPU Acceleration with cv2.UMat

`cv2.UMat` (Universal Mat) is OpenCV's abstraction for GPU-backed image storage. When OpenCL is available, operations on `UMat` objects are automatically dispatched to the GPU:

```python
# Move image to GPU
gpu_img = cv2.UMat(img)

# Operations run on GPU transparently
gpu_gray = cv2.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
gpu_blur = cv2.GaussianBlur(gpu_gray, (15, 15), 0)
gpu_edges = cv2.Canny(gpu_blur, 50, 150)

# Move result back to CPU
result = gpu_edges.get()
```

| Aspect | `np.ndarray` (CPU) | `cv2.UMat` (GPU) |
|---|---|---|
| Storage | System RAM | GPU memory (via OpenCL) |
| NumPy operations | Yes | No (must use cv2 functions) |
| OpenCV functions | Yes | Yes (automatically accelerated) |
| Transfer cost | None | Upload + download overhead |

The key tradeoff: GPU acceleration has overhead for data transfer. It pays off when you chain **many** OpenCV operations on large images, because the computation savings outweigh the transfer cost. For a single small operation, the overhead can make it slower.

## Practical Tips for Speed

**Resize early**: If your source image is 4000x3000 but you only need a rough analysis, resize to 800x600 before processing. Most operations scale with pixel count.

```python
small = cv2.resize(img, None, fx=0.25, fy=0.25)
```

**Use appropriate data types**: `uint8` operations are faster than `float32`. Only convert to float when needed (e.g., for precise arithmetic).

**Prefer OpenCV functions over NumPy equivalents** when available. `cv2.add()` is faster than `np.add()` for images because it handles saturation and uses optimized SIMD instructions:

```python
# cv2.add saturates at 255 (correct for images)
result = cv2.add(img1, img2)
# np.add wraps around at 256 (incorrect for images)
result = np.add(img1, img2)  # 200 + 100 = 44, not 255!
```

## The Complete Pipeline

1. **Measure Baseline**: Time the original (unoptimized) pipeline
2. **Replace Loops**: Convert Python pixel loops to vectorized NumPy
3. **Use OpenCV Built-ins**: Replace custom implementations with cv2 functions
4. **Try UMat**: Move data to GPU for multi-step pipelines
5. **Resize Early**: Downscale before heavy processing
6. **Compare Timings**: Measure improvement at each step

## Tips & Common Mistakes

- Always warm up the GPU before benchmarking UMat operations. The first call includes OpenCL initialization overhead.
- UMat does not support direct NumPy indexing (`gpu_img[y, x]` won't work). Use only `cv2.*` functions with UMat.
- `cv2.UMat` falls back to CPU if OpenCL is not available, so your code stays portable -- just without GPU speedup.
- Don't mix `uint8` arithmetic with expectations of `int` behavior. `uint8(200) + uint8(100)` wraps to `44` in NumPy but saturates to `255` with `cv2.add()`.
- Benchmarking a single run can be misleading due to caching. Average over multiple runs for reliable measurements.
- Python's `time.time()` has millisecond resolution on some platforms. Use `cv2.getTickCount()` or `time.perf_counter()` for sub-millisecond timing.
- Profile before optimizing. The bottleneck may not be where you think it is.

## Starter Code

```python
import cv2
import numpy as np

# =============================================================
# Step 1: Create a test image for benchmarking
# =============================================================
img_h, img_w = 800, 1200
img = np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8)

# Add some structure to make it more realistic
cv2.rectangle(img, (100, 100), (500, 400), (0, 0, 200), -1)
cv2.circle(img, (800, 300), 150, (0, 200, 0), -1)
cv2.GaussianBlur(img, (5, 5), 2, dst=img)

print(f'Test image: {img_w}x{img_h} ({img.nbytes / 1024:.0f} KB)')
print(f'OpenCL available: {cv2.ocl.haveOpenCL()}')
print()

N_RUNS = 20  # Number of runs for averaging

# =============================================================
# Step 2: Benchmark Python loop vs vectorized inversion
# =============================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Slow: Python loop (on a smaller crop to keep it feasible) ---
crop_h, crop_w = 100, 100
crop = gray[:crop_h, :crop_w].copy()

t1 = cv2.getTickCount()
loop_result = np.zeros_like(crop)
for y in range(crop_h):
    for x in range(crop_w):
        loop_result[y, x] = 255 - crop[y, x]
t2 = cv2.getTickCount()
loop_time = (t2 - t1) / cv2.getTickFrequency() * 1000
loop_per_pixel = loop_time / (crop_h * crop_w) * 1e6  # nanoseconds per pixel

# --- Fast: Vectorized NumPy (on full image) ---
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    vec_result = 255 - gray
t2 = cv2.getTickCount()
vec_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS
vec_per_pixel = vec_time / (img_h * img_w) * 1e6  # nanoseconds per pixel

print('=== Pixel Inversion: Loop vs Vectorized ===')
print(f'Python loop ({crop_w}x{crop_h} crop): {loop_time:.2f} ms '
      f'({loop_per_pixel:.0f} ns/pixel)')
print(f'NumPy vectorized ({img_w}x{img_h} full): {vec_time:.2f} ms '
      f'({vec_per_pixel:.1f} ns/pixel)')
print(f'Speedup per pixel: ~{loop_per_pixel / max(vec_per_pixel, 0.001):.0f}x')
print()

# =============================================================
# Step 3: Benchmark threshold implementations
# =============================================================
# --- NumPy where ---
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    thresh_np = np.where(gray > 128, np.uint8(255), np.uint8(0))
t2 = cv2.getTickCount()
np_thresh_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

# --- OpenCV threshold ---
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    _, thresh_cv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
t2 = cv2.getTickCount()
cv_thresh_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

print('=== Thresholding: NumPy vs OpenCV ===')
print(f'np.where:        {np_thresh_time:.2f} ms')
print(f'cv2.threshold:   {cv_thresh_time:.2f} ms')
print(f'Speedup: {np_thresh_time / max(cv_thresh_time, 0.001):.1f}x')
print()

# =============================================================
# Step 4: Benchmark a multi-step pipeline (CPU vs UMat)
# =============================================================
def pipeline_cpu(image):
    """Multi-step processing pipeline on CPU."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return dilated

# CPU benchmark
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    cpu_result = pipeline_cpu(img)
t2 = cv2.getTickCount()
cpu_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

print('=== Multi-step Pipeline: CPU vs UMat (GPU) ===')
print(f'CPU pipeline: {cpu_time:.2f} ms')

# UMat (GPU) benchmark -- warm up first
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

    # Warm-up run (includes OpenCL initialization)
    gpu_img = cv2.UMat(img)
    _ = cv2.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

    def pipeline_gpu(image_umat):
        """Multi-step processing pipeline on GPU via UMat."""
        gray = cv2.cvtColor(image_umat, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blur, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        return dilated

    gpu_img = cv2.UMat(img)
    t1 = cv2.getTickCount()
    for _ in range(N_RUNS):
        gpu_result = pipeline_gpu(gpu_img)
    t2 = cv2.getTickCount()
    gpu_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS
    gpu_result_cpu = gpu_result.get()  # Transfer back to CPU

    print(f'UMat pipeline: {gpu_time:.2f} ms')
    print(f'Speedup: {cpu_time / max(gpu_time, 0.001):.1f}x')
else:
    print('OpenCL not available -- UMat falls back to CPU')
    gpu_result_cpu = cpu_result
    gpu_time = cpu_time

print()

# =============================================================
# Step 5: Benchmark resize-first optimization
# =============================================================
# Full resolution pipeline
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    full_result = pipeline_cpu(img)
t2 = cv2.getTickCount()
full_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

# Half resolution pipeline
img_half = cv2.resize(img, None, fx=0.5, fy=0.5)
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    half_result = pipeline_cpu(img_half)
t2 = cv2.getTickCount()
half_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

# Quarter resolution pipeline
img_quarter = cv2.resize(img, None, fx=0.25, fy=0.25)
t1 = cv2.getTickCount()
for _ in range(N_RUNS):
    quarter_result = pipeline_cpu(img_quarter)
t2 = cv2.getTickCount()
quarter_time = (t2 - t1) / cv2.getTickFrequency() * 1000 / N_RUNS

print('=== Resize-First Optimization ===')
print(f'Full res ({img_w}x{img_h}):      {full_time:.2f} ms')
print(f'Half res ({img_w//2}x{img_h//2}):     {half_time:.2f} ms '
      f'({full_time/max(half_time,0.001):.1f}x faster)')
print(f'Quarter res ({img_w//4}x{img_h//4}):   {quarter_time:.2f} ms '
      f'({full_time/max(quarter_time,0.001):.1f}x faster)')
print()

# =============================================================
# Step 6: Build visualization with results
# =============================================================
display_h, display_w = 300, 500
result_display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
result_display[:] = (30, 30, 30)

font = cv2.FONT_HERSHEY_SIMPLEX
y = 25

cv2.putText(result_display, 'Performance Benchmark Results', (10, y),
            font, 0.6, (0, 255, 0), 2)
y += 35

cv2.putText(result_display, f'Image: {img_w}x{img_h} | Runs: {N_RUNS}', (10, y),
            font, 0.4, (200, 200, 200), 1)
y += 30

cv2.putText(result_display, f'Loop inversion: {loop_time:.1f}ms (100x100 crop)', (10, y),
            font, 0.4, (150, 150, 255), 1)
y += 22
cv2.putText(result_display, f'NumPy inversion: {vec_time:.2f}ms (full image)', (10, y),
            font, 0.4, (150, 255, 150), 1)
y += 30

cv2.putText(result_display, f'np.where threshold: {np_thresh_time:.2f}ms', (10, y),
            font, 0.4, (150, 150, 255), 1)
y += 22
cv2.putText(result_display, f'cv2.threshold:      {cv_thresh_time:.2f}ms', (10, y),
            font, 0.4, (150, 255, 150), 1)
y += 30

cv2.putText(result_display, f'CPU pipeline:  {cpu_time:.2f}ms', (10, y),
            font, 0.4, (150, 150, 255), 1)
y += 22
cv2.putText(result_display, f'UMat pipeline: {gpu_time:.2f}ms', (10, y),
            font, 0.4, (150, 255, 150), 1)
y += 30

cv2.putText(result_display, f'Full: {full_time:.1f}ms | Half: {half_time:.1f}ms | '
            f'Quarter: {quarter_time:.1f}ms', (10, y),
            font, 0.4, (200, 200, 100), 1)

# Show pipeline output images
cpu_bgr = cv2.cvtColor(cpu_result, cv2.COLOR_GRAY2BGR)
cpu_small = cv2.resize(cpu_bgr, (display_w // 2, display_h // 2))
cv2.putText(cpu_small, 'Pipeline Output', (5, 18), font, 0.4, (0, 255, 0), 1)

# Input image preview
img_small = cv2.resize(img, (display_w // 2, display_h // 2))
cv2.putText(img_small, 'Input Image', (5, 18), font, 0.4, (0, 255, 0), 1)

images_row = np.hstack([img_small, cpu_small])

result = np.vstack([result_display, images_row])

cv2.imshow('Performance Optimization', result)
```
