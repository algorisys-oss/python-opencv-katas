---
slug: 00-opencv-basics
title: OpenCV Basics
level: beginner
concepts: [what is OpenCV, computer vision, NumPy, pixels, coordinate system]
prerequisites: []
---

## What Is Computer Vision?

Computer vision is the field of teaching computers to **understand images and video**. Every time your phone unlocks with your face, a self-driving car reads a stop sign, or an app applies a photo filter — that's computer vision at work.

At its core, computer vision answers one question: **what's in this image, and where?**

## What Is OpenCV?

**OpenCV** (Open Source Computer Vision Library) is the most widely used computer vision library in the world. It was originally written in C++ and has bindings for Python, Java, and other languages.

In Python, you use it via the `cv2` module:

```python
import cv2
```

> **Why `cv2` and not `opencv`?** The `cv2` name comes from the second major version of the OpenCV C++ API. The Python module kept this name for historical reasons. There is no `cv1` module in Python.

### What Can OpenCV Do?

| Category | Examples |
|---|---|
| **Image basics** | Load, save, resize, crop, rotate |
| **Color** | Convert between color spaces (BGR, RGB, HSV, Grayscale) |
| **Drawing** | Lines, circles, rectangles, text on images |
| **Filtering** | Blur, sharpen, edge detection |
| **Detection** | Faces, objects, contours, features |
| **Video** | Read from files or webcam, process frame by frame |
| **Advanced** | Object tracking, image stitching, optical flow |

You don't need to learn all of this at once. This playground teaches one concept at a time, starting from the very basics.

## What Is a Digital Image?

A digital image is a **grid of tiny colored squares** called **pixels** (short for "picture elements"). Each pixel has a color value represented by numbers.

```
+-----+-----+-----+-----+
| 0,0 | 0,1 | 0,2 | 0,3 |   ← Row 0
+-----+-----+-----+-----+
| 1,0 | 1,1 | 1,2 | 1,3 |   ← Row 1
+-----+-----+-----+-----+
| 2,0 | 2,1 | 2,2 | 2,3 |   ← Row 2
+-----+-----+-----+-----+
  Col 0  Col 1  Col 2  Col 3
```

Key facts:
- The **top-left** corner is the origin `(0, 0)`.
- **Row** increases downward (this is the **y** direction).
- **Column** increases to the right (this is the **x** direction).
- A 1920x1080 HD image has **over 2 million pixels**.

## How OpenCV Stores Images — NumPy Arrays

When OpenCV loads an image, it stores it as a **NumPy array**. NumPy is Python's standard library for working with large arrays of numbers efficiently.

A **color image** is a 3D array with shape `(height, width, channels)`:

```
img.shape = (height, width, 3)
                              └── 3 color channels
```

A **grayscale image** is a 2D array with shape `(height, width)`:

```
img.shape = (height, width)
```

### Why NumPy?

NumPy is fast because it processes millions of numbers using optimized C code under the hood. Without it, looping through every pixel in Python would be painfully slow. Almost everything in OpenCV takes and returns NumPy arrays.

> **You don't need to be a NumPy expert to start.** You just need to know that an image is an array of numbers, and you can access any pixel using `img[row, col]`.

## Pixels and Color Channels

### Grayscale

Each pixel is a single number from **0** (black) to **255** (white):

```
  0 = pure black
128 = medium gray
255 = pure white
```

### Color (BGR)

Each pixel has **3 numbers** — one for each color channel. OpenCV uses **BGR** order (not RGB):

```
img[y, x] = [Blue, Green, Red]
```

| Channel | Index | Example: Pure Red |
|---|---|---|
| Blue | 0 | 0 |
| Green | 1 | 0 |
| Red | 2 | 255 |

So the pixel value `[0, 0, 255]` is **red** in OpenCV (zero blue, zero green, full red).

> **Why BGR instead of RGB?** When OpenCV was created in the late 1990s, camera hardware commonly stored bytes in BGR order. The library kept this convention for backward compatibility. This catches almost every beginner — remember: **Blue first, Red last**.

## The Coordinate System

This is one of the most common sources of confusion. OpenCV (and NumPy) use **row-major** order:

```
img[row, col]  =  img[y, x]
```

- **row** = **y** = vertical position (0 at top, increases downward)
- **col** = **x** = horizontal position (0 at left, increases rightward)

```
        x (columns) →
    ┌──────────────────────┐
    │ (0,0)          (0,w) │
y   │                      │
(rows) │                      │
↓   │ (h,0)          (h,w) │
    └──────────────────────┘
```

> **Critical rule:** When you write `img[a, b]`, `a` is always the **row** (y) and `b` is always the **column** (x). This is the opposite of what many people expect from `(x, y)` notation.

## Data Types — uint8

OpenCV images use the data type **`uint8`** by default:

- **u** = unsigned (no negative numbers)
- **int** = integer (whole numbers only)
- **8** = 8 bits = values from **0 to 255**

This means:
- Pixel values are always between **0** and **255**.
- If you try to set a pixel to 300, NumPy will **wrap around** (300 becomes 44), which creates bugs.
- If you try to set a pixel to -10, it wraps to 246.

```
0   = minimum intensity (black for grayscale)
255 = maximum intensity (white for grayscale)
```

> **Why 0–255?** Because 8 bits can represent 2^8 = 256 different values. This is the standard range for display hardware. For scientific work, OpenCV also supports `float32` and `float64`, but for learning, `uint8` is all you need.

## The Two Core Libraries You Need

Every kata in this playground uses just two imports:

```python
import cv2          # OpenCV — image processing functions
import numpy as np  # NumPy  — array creation and manipulation
```

| Library | What It Does | Example |
|---|---|---|
| `cv2` | Read, write, transform, and display images | `cv2.imread()`, `cv2.cvtColor()` |
| `numpy` | Create and manipulate the arrays that hold image data | `np.zeros()`, `np.uint8` |

You don't need to install anything extra — both come pre-installed in this playground.

## What You'll Learn Next

This kata had no live code — it was all concepts. Starting from the next kata, you'll write and run real OpenCV code in the editor.

Here's the learning path ahead:

1. **Image Loading & Display** — Create images from scratch, understand shapes and dtypes
2. **Color Spaces** — Convert between BGR, Grayscale, and HSV
3. **Pixel Access** — Read and write individual pixels and regions
4. **Resizing & Cropping** — Change image dimensions
5. **Drawing Primitives** — Lines, circles, rectangles, and text

Each kata builds on the previous one. Take them in order.

## Key Takeaways

- An image is a **NumPy array** of numbers.
- Color images have shape `(height, width, 3)` in **BGR** order.
- Grayscale images have shape `(height, width)` with values 0–255.
- Pixel access uses `img[row, col]` = `img[y, x]` — **row first, column second**.
- Pixel values are `uint8`: integers from **0 to 255**.
- You only need two imports: `cv2` and `numpy`.
