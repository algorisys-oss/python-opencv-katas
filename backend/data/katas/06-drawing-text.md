---
slug: 06-drawing-text
title: Drawing Text
level: beginner
concepts: [cv2.putText, font faces, getTextSize, baseline]
prerequisites: [04-drawing-lines-rectangles]
---

## What Problem Are We Solving?

You often need to **label** things on images — display a detection confidence score, annotate a region, show FPS on a video frame. `cv2.putText()` is how you render text onto an image.

## Putting Text on an Image

```python
cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
```

| Parameter | Meaning |
|---|---|
| `text` | The string to draw |
| `org` | **Bottom-left** corner of the text as `(x, y)` |
| `fontFace` | Font constant (see table below) |
| `fontScale` | Size multiplier (1.0 = base size, 2.0 = double) |
| `color` | BGR tuple |
| `thickness` | Stroke width of the letters (default: 1) |
| `lineType` | `cv2.LINE_AA` for smooth text (recommended) |

> **Important:** `org` is the **bottom-left** corner of the text, not the top-left. The text draws **upward** from this point. This catches many beginners — if you place `org` at `(10, 10)`, most of the text will be above the image and invisible.

## Available Fonts

OpenCV has a small set of built-in fonts (these are not system fonts — they're drawn pixel by pixel):

| Constant | Style |
|---|---|
| `cv2.FONT_HERSHEY_SIMPLEX` | Normal sans-serif |
| `cv2.FONT_HERSHEY_PLAIN` | Small sans-serif |
| `cv2.FONT_HERSHEY_DUPLEX` | Normal sans-serif (thicker) |
| `cv2.FONT_HERSHEY_COMPLEX` | Normal serif |
| `cv2.FONT_HERSHEY_TRIPLEX` | Normal serif (thicker) |
| `cv2.FONT_HERSHEY_SCRIPT_SIMPLEX` | Handwriting style |
| `cv2.FONT_HERSHEY_SCRIPT_COMPLEX` | Handwriting style (thicker) |

Add `cv2.FONT_ITALIC` with bitwise OR to make any font italic:

```python
font = cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC
```

## Measuring Text Before Drawing

To center text or place a background box behind it, you need to know its size **before** drawing:

```python
(text_w, text_h), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
```

Returns:
- `text_w`, `text_h` — width and height of the text in pixels
- `baseline` — distance from the bottom of the text to the lowest descender (letters like `g`, `p`, `y`)

## Centering Text

```python
text = "Hello"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1.5
thickness = 2

(tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

# Center of the image
cx = img.shape[1] // 2
cy = img.shape[0] // 2

# org = bottom-left of text, so offset by half the text size
org = (cx - tw // 2, cy + th // 2)
cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
```

## Drawing a Background Box Behind Text

A common pattern for readable labels:

```python
text = "Score: 0.95"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.7
thickness = 1
padding = 5

(tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

# Draw filled rectangle behind text
x, y = 10, 40  # where we want the text (org position)
cv2.rectangle(img, (x - padding, y + baseline + padding),
              (x + tw + padding, y - th - padding), (0, 0, 0), -1)
cv2.putText(img, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
```

## Tips & Common Mistakes

- `org` is the **bottom-left** of the text — place it at least `text_h` pixels from the top edge.
- Use `cv2.LINE_AA` for smooth, anti-aliased text. It looks much better than the default.
- `fontScale` is a multiplier, not a point size. Start with `0.5`–`1.0` for normal text.
- OpenCV fonts are bitmap-rendered, not TrueType. For fancy typography, use PIL/Pillow instead.
- `thickness` makes letters **bolder**, not bigger. Use `fontScale` to change size.
- Always use `getTextSize()` when you need precise text placement.

## Starter Code

```python
import cv2
import numpy as np

# Create a dark canvas
img = np.zeros((450, 550, 3), dtype=np.uint8)
img[:] = (30, 30, 30)

# --- Basic text ---
cv2.putText(img, 'Hello, OpenCV!', (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# --- Font showcase ---
fonts = [
    ('SIMPLEX', cv2.FONT_HERSHEY_SIMPLEX),
    ('PLAIN', cv2.FONT_HERSHEY_PLAIN),
    ('DUPLEX', cv2.FONT_HERSHEY_DUPLEX),
    ('COMPLEX', cv2.FONT_HERSHEY_COMPLEX),
    ('SCRIPT', cv2.FONT_HERSHEY_SCRIPT_SIMPLEX),
    ('ITALIC', cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC),
]

y = 90
for name, font in fonts:
    cv2.putText(img, f'{name}: OpenCV', (20, y), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    y += 35

# --- Centered text with background ---
text = 'Centered!'
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1.2
thick = 2
pad = 8

(tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
cx = img.shape[1] // 2
cy = 370

# Background box
cv2.rectangle(img, (cx - tw // 2 - pad, cy - th - pad),
              (cx + tw // 2 + pad, cy + baseline + pad), (0, 80, 0), -1)
# Text
cv2.putText(img, text, (cx - tw // 2, cy), font, scale, (0, 255, 0), thick, cv2.LINE_AA)

# --- Label with score ---
label = 'Confidence: 0.97'
(lw, lh), lb = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
lx, ly = 20, 430
cv2.rectangle(img, (lx - 4, ly - lh - 4), (lx + lw + 4, ly + lb + 4), (0, 0, 0), -1)
cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

print(f'Text size of "Centered!": {tw}x{th}, baseline: {baseline}')

cv2.imshow('Drawing Text', img)
```
