---
slug: 99-cv-pipeline-framework
title: Building a CV Pipeline
level: advanced
concepts: [pipeline stages, composition, error handling, benchmarking]
prerequisites: [98-performance-optimization]
---

## What Problem Are We Solving?

As computer vision projects grow beyond single-script experiments, you need a structured way to compose processing steps. A **pipeline framework** lets you define individual stages (grayscale conversion, blur, edge detection, etc.) as reusable units, chain them together, add timing and logging automatically, and handle errors gracefully.

This is the pattern used in production CV systems: each stage has a clear input/output contract, pipelines are composable, and debugging is straightforward because each stage can be inspected individually.

## Designing Pipeline Stages as Functions

The simplest pipeline stage is a function that takes an image and returns a processed image. Using a consistent signature makes stages interchangeable:

```python
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def detect_edges(img, low=50, high=150):
    return cv2.Canny(img, low, high)
```

To handle stages that need parameters, use `functools.partial` or lambda wrappers:

```python
from functools import partial
blur_7x7 = partial(blur, ksize=7)
```

## Chaining Stages into a Pipeline

A pipeline is a sequence of stages applied in order. The output of each stage becomes the input of the next:

```python
def run_pipeline(image, stages):
    result = image
    for stage in stages:
        result = stage(result)
    return result

pipeline = [to_grayscale, blur_7x7, detect_edges]
output = run_pipeline(input_image, pipeline)
```

## Adding Timing and Logging

Wrapping each stage with timing produces a detailed performance profile:

```python
def run_pipeline_timed(image, stages):
    result = image
    timings = []
    for stage in stages:
        t1 = cv2.getTickCount()
        result = stage(result)
        t2 = cv2.getTickCount()
        ms = (t2 - t1) / cv2.getTickFrequency() * 1000
        timings.append((stage.__name__, ms))
    return result, timings
```

This tells you exactly where time is being spent, making optimization targeted rather than guesswork.

## Error Handling Patterns

Pipeline stages can fail -- an edge detection stage might receive a color image when it expects grayscale, or a contour stage might find no contours. Robust pipelines wrap each stage with error handling:

```python
def safe_run(image, stages):
    result = image
    for stage in stages:
        try:
            result = stage(result)
        except Exception as e:
            print(f'Stage {stage.__name__} failed: {e}')
            return None
    return result
```

For more sophisticated handling, stages can return a tuple of `(image, metadata)` where metadata carries information like detected contours, timing, or error flags.

## Composing Pipelines from Pipelines

Once you have a pipeline framework, you can compose higher-level pipelines from lower-level ones:

```python
preprocess = [to_grayscale, blur]
analysis = [detect_edges, find_contours]
full_pipeline = preprocess + analysis
```

This promotes reuse -- the same preprocessing pipeline can feed different analysis pipelines.

## The Complete Pipeline

1. **Define Stages**: Write individual processing functions with consistent signatures
2. **Create Pipeline**: List stages in order
3. **Add Instrumentation**: Wrap with timing, logging, and intermediate output capture
4. **Handle Errors**: Catch failures per-stage with meaningful messages
5. **Compose**: Build complex pipelines from simpler sub-pipelines
6. **Run and Inspect**: Execute and examine intermediate results

## Tips & Common Mistakes

- Keep stage functions pure (no side effects). Each stage should only transform its input and return a result, not modify global state.
- Use `functools.partial` rather than lambda for parameterized stages. Partials preserve the function name (`__name__`), which is useful for logging.
- Be explicit about data type expectations. Document whether a stage expects grayscale or color, `uint8` or `float32`.
- Don't silently swallow errors. Log them with the stage name and input shape so you can diagnose problems.
- For video pipelines, consider caching intermediate results when only part of the pipeline needs to re-run (e.g., if parameters change for only the last stage).
- When capturing intermediate results for debugging, store them in a dictionary keyed by stage name, not a list, for easier access.
- Test each stage independently before composing into a pipeline. This makes it clear whether a bug is in a specific stage or in the composition.

## Starter Code

```python
import cv2
import numpy as np
from functools import partial

# =============================================================
# Step 1: Define the pipeline framework
# =============================================================

class PipelineStage:
    """Wraps a processing function with metadata."""
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __call__(self, img):
        return self.func(img)

class CVPipeline:
    """Composable computer vision pipeline with timing and error handling."""

    def __init__(self, stages=None, name='Pipeline'):
        self.stages = stages or []
        self.name = name
        self.timings = []
        self.intermediates = {}
        self.errors = []

    def add(self, func, name=None):
        """Add a stage to the pipeline."""
        stage = PipelineStage(func, name)
        self.stages.append(stage)
        return self  # Allow chaining: pipeline.add(f1).add(f2)

    def run(self, image, capture_intermediates=False):
        """Execute the pipeline with timing and error handling."""
        self.timings = []
        self.intermediates = {}
        self.errors = []
        result = image.copy()

        for stage in self.stages:
            t1 = cv2.getTickCount()
            try:
                result = stage(result)
                t2 = cv2.getTickCount()
                ms = (t2 - t1) / cv2.getTickFrequency() * 1000
                self.timings.append((stage.name, ms))

                if capture_intermediates:
                    self.intermediates[stage.name] = result.copy()

            except Exception as e:
                t2 = cv2.getTickCount()
                ms = (t2 - t1) / cv2.getTickFrequency() * 1000
                self.timings.append((stage.name, ms))
                self.errors.append((stage.name, str(e)))
                print(f'  [ERROR] Stage "{stage.name}" failed: {e}')
                return None

        return result

    def report(self):
        """Print timing report."""
        total = sum(t for _, t in self.timings)
        print(f'\n  Pipeline: {self.name}')
        print(f'  {"Stage":<30} {"Time (ms)":>10} {"% Total":>10}')
        print(f'  {"-"*52}')
        for name, ms in self.timings:
            pct = (ms / total * 100) if total > 0 else 0
            bar = '#' * int(pct / 2)
            print(f'  {name:<30} {ms:>10.2f} {pct:>9.1f}% {bar}')
        print(f'  {"-"*52}')
        print(f'  {"TOTAL":<30} {total:>10.2f}')
        if self.errors:
            print(f'  Errors: {len(self.errors)}')
            for name, err in self.errors:
                print(f'    - {name}: {err}')
        return total

# =============================================================
# Step 2: Define reusable processing stages
# =============================================================

def to_grayscale(img):
    """Convert BGR to grayscale."""
    if len(img.shape) == 2:
        return img  # Already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, ksize=5):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny_edges(img, low=50, high=150):
    """Detect edges using Canny."""
    return cv2.Canny(img, low, high)

def dilate(img, ksize=3, iterations=1):
    """Dilate the image."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)

def adaptive_threshold(img):
    """Apply adaptive threshold."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

def find_and_draw_contours(img):
    """Find contours and draw them on a blank image."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    return result

def invert(img):
    """Invert image colors."""
    return cv2.bitwise_not(img)

def normalize_brightness(img):
    """Normalize brightness using CLAHE."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

# =============================================================
# Step 3: Create a synthetic test image
# =============================================================
img_h, img_w = 400, 600
test_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
test_img[:] = (180, 170, 160)

# Draw shapes to give the pipeline something to process
cv2.rectangle(test_img, (40, 40), (200, 200), (50, 50, 200), -1)
cv2.circle(test_img, (350, 120), 80, (50, 200, 50), -1)
cv2.rectangle(test_img, (450, 60), (560, 220), (200, 100, 50), -1)
cv2.ellipse(test_img, (150, 320), (100, 40), 30, 0, 360, (200, 200, 50), -1)
cv2.rectangle(test_img, (350, 250), (550, 370), (100, 50, 200), -1)

# Add text
cv2.putText(test_img, 'Pipeline Test', (180, 390),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Add some noise for realism
noise = np.random.randint(-15, 15, test_img.shape, dtype=np.int16)
test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

print('='*55)
print('  COMPUTER VISION PIPELINE FRAMEWORK DEMO')
print('='*55)

# =============================================================
# Step 4: Build and run Pipeline 1 -- Edge Detection Pipeline
# =============================================================
edge_pipeline = CVPipeline(name='Edge Detection')
edge_pipeline.add(to_grayscale, 'Grayscale')
edge_pipeline.add(partial(gaussian_blur, ksize=5), 'Gaussian Blur (5x5)')
edge_pipeline.add(partial(canny_edges, low=50, high=150), 'Canny Edges')
edge_pipeline.add(partial(dilate, ksize=3), 'Dilate')

edge_result = edge_pipeline.run(test_img, capture_intermediates=True)
edge_total = edge_pipeline.report()

# =============================================================
# Step 5: Build and run Pipeline 2 -- Threshold + Contours
# =============================================================
contour_pipeline = CVPipeline(name='Contour Analysis')
contour_pipeline.add(normalize_brightness, 'CLAHE Normalize')
contour_pipeline.add(to_grayscale, 'Grayscale')
contour_pipeline.add(partial(gaussian_blur, ksize=7), 'Gaussian Blur (7x7)')
contour_pipeline.add(adaptive_threshold, 'Adaptive Threshold')
contour_pipeline.add(invert, 'Invert')
contour_pipeline.add(find_and_draw_contours, 'Find & Draw Contours')

contour_result = contour_pipeline.run(test_img, capture_intermediates=True)
contour_total = contour_pipeline.report()

# =============================================================
# Step 6: Build and run Pipeline 3 -- Composed pipeline
#         (reuse preprocessing + custom analysis)
# =============================================================
composed = CVPipeline(name='Composed Pipeline')
# Reuse preprocessing stages
composed.add(normalize_brightness, 'CLAHE Normalize')
composed.add(to_grayscale, 'Grayscale')
composed.add(partial(gaussian_blur, ksize=3), 'Gaussian Blur (3x3)')
# Different analysis path
composed.add(partial(canny_edges, low=30, high=100), 'Canny (sensitive)')
composed.add(partial(dilate, ksize=5, iterations=2), 'Heavy Dilate')
composed.add(find_and_draw_contours, 'Find & Draw Contours')

composed_result = composed.run(test_img, capture_intermediates=True)
composed_total = composed.report()

# =============================================================
# Step 7: Test error handling with a broken stage
# =============================================================
def broken_stage(img):
    """This stage intentionally causes an error."""
    raise ValueError('Image dimensions are invalid (simulated error)')

error_pipeline = CVPipeline(name='Error Handling Test')
error_pipeline.add(to_grayscale, 'Grayscale')
error_pipeline.add(broken_stage, 'Broken Stage')
error_pipeline.add(partial(canny_edges), 'Canny (never reached)')

error_result = error_pipeline.run(test_img)
error_pipeline.report()

# =============================================================
# Step 8: Build visualization
# =============================================================
font = cv2.FONT_HERSHEY_SIMPLEX
panel_w, panel_h = 300, 200

def make_panel(img, label):
    """Resize image to panel size and add label."""
    if img is None:
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, 'ERROR', (100, 100), font, 0.7, (0, 0, 255), 2)
    else:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(img, (panel_w, panel_h))
    cv2.putText(panel, label, (5, 18), font, 0.45, (0, 255, 0), 1)
    return panel

# Row 1: Input + Edge stages
p_input = make_panel(test_img, 'Input')
p_edge = make_panel(edge_result, f'Edges ({edge_total:.1f}ms)')

# Get intermediates for display
edge_gray = edge_pipeline.intermediates.get('Grayscale')
p_gray = make_panel(edge_gray, 'Grayscale')

# Row 2: Contour pipeline stages
contour_thresh = contour_pipeline.intermediates.get('Adaptive Threshold')
p_thresh = make_panel(contour_thresh, 'Threshold')
p_contour = make_panel(contour_result, f'Contours ({contour_total:.1f}ms)')
p_composed = make_panel(composed_result, f'Composed ({composed_total:.1f}ms)')

row1 = np.hstack([p_input, p_gray, p_edge])
row2 = np.hstack([p_thresh, p_contour, p_composed])
result = np.vstack([row1, row2])

print(f'\nTotal pipelines executed: 4 (including error test)')
print(f'Total stages defined: {len(edge_pipeline.stages) + len(contour_pipeline.stages) + len(composed.stages)}')

cv2.imshow('Building a CV Pipeline', result)
```
