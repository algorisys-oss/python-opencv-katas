---
slug: 35-morphology-opening-closing
title: "Morphology: Opening & Closing"
level: intermediate
concepts: [cv2.morphologyEx, MORPH_OPEN, MORPH_CLOSE]
prerequisites: [34-morphology-dilation]
---

## What Problem Are We Solving?

Erosion removes noise but also shrinks your objects. Dilation fills holes but also bloats your objects. What if you want to **remove noise without changing object size**, or **fill holes without expanding objects**? That's exactly what **opening** and **closing** do — they combine erosion and dilation in sequence to achieve targeted cleanup.

## Opening: Erosion Then Dilation

**Opening** = erode first, then dilate with the same kernel.

```python
opened = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
```

What it does:
- The erosion step **removes small white noise** blobs (anything smaller than the kernel disappears)
- The dilation step **restores the original size** of the surviving objects

Opening is perfect for removing **small bright noise** on a dark background while keeping larger objects intact.

## Closing: Dilation Then Erosion

**Closing** = dilate first, then erode with the same kernel.

```python
closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
```

What it does:
- The dilation step **fills small black holes** and gaps inside white regions
- The erosion step **restores the original size** so objects don't stay bloated

Closing is perfect for filling **small dark holes** inside bright objects.

## Using cv2.morphologyEx()

```python
result = cv2.morphologyEx(src, op, kernel, iterations=1)
```

| Parameter | Meaning |
|---|---|
| `src` | Input image |
| `op` | Operation type: `cv2.MORPH_OPEN` or `cv2.MORPH_CLOSE` |
| `kernel` | Structuring element |
| `iterations` | Number of times to apply (default: 1) |

> **Note:** `iterations=2` with `MORPH_OPEN` means erode-dilate-erode-dilate (the full open operation is repeated), not erode-erode-dilate-dilate.

## Choosing the Right Kernel Size

- **Small kernel (3x3)** — removes only tiny noise/holes (1-2 pixels)
- **Medium kernel (5x5 to 7x7)** — removes small blobs, fills small gaps
- **Large kernel (9x9+)** — aggressive cleanup, but may merge or remove legitimate features

```python
small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
```

## Opening vs Closing: Quick Reference

| Operation | Removes | Preserves | Use when |
|---|---|---|---|
| Opening | Small white blobs | Large white objects | Noise dots on dark background |
| Closing | Small black holes | Large white objects | Holes inside objects |

## Combining Both

For images with both noise and holes, apply opening first (to remove noise), then closing (to fill remaining holes):

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
```

## Tips & Common Mistakes

- Always use the **same kernel** for the erosion and dilation steps — that's what preserves object size. `cv2.morphologyEx` handles this automatically.
- Opening cannot restore objects that were **completely eroded away**. If your objects are smaller than the kernel, they'll disappear.
- Closing cannot fix gaps **larger than the kernel**. If holes are bigger than the structuring element, they won't be filled.
- Don't confuse the order: **Open** = erode-dilate (removes noise), **Close** = dilate-erode (fills holes). Think: "Open the door to let noise out, close the door to keep holes filled."
- Use `MORPH_ELLIPSE` for natural-looking results. `MORPH_RECT` can leave blocky artifacts.
- If one pass isn't enough, increase `iterations` or use a larger kernel rather than calling the function multiple times manually.

## Starter Code

```python
import cv2
import numpy as np

# --- Create a binary image with both noise and holes ---
img = np.zeros((400, 600), dtype=np.uint8)

# Main objects
cv2.rectangle(img, (30, 30), (180, 180), 255, -1)
cv2.circle(img, (320, 110), 80, 255, -1)
cv2.rectangle(img, (450, 30), (570, 180), 255, -1)

# Objects in the bottom half
cv2.circle(img, (100, 310), 60, 255, -1)
cv2.rectangle(img, (230, 250), (400, 370), 255, -1)
cv2.ellipse(img, (510, 310), (70, 50), 0, 0, 360, 255, -1)

# Add small WHITE noise (opening will remove these)
for _ in range(100):
    x = np.random.randint(0, 600)
    y = np.random.randint(0, 400)
    r = np.random.randint(1, 4)
    cv2.circle(img, (x, y), r, 255, -1)

# Add small BLACK holes inside objects (closing will fill these)
for _ in range(40):
    x = np.random.randint(40, 170)
    y = np.random.randint(40, 170)
    cv2.circle(img, (x, y), 2, 0, -1)
for _ in range(40):
    x = np.random.randint(240, 390)
    y = np.random.randint(260, 360)
    cv2.circle(img, (x, y), 2, 0, -1)
for _ in range(30):
    cx, cy = 320, 110
    angle = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, 60)
    x = int(cx + r * np.cos(angle))
    y = int(cy + r * np.sin(angle))
    cv2.circle(img, (x, y), 2, 0, -1)

# --- Apply morphological operations ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
open_then_close = cv2.morphologyEx(
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
    cv2.MORPH_CLOSE, kernel
)

# --- Build comparison display ---
def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

font = cv2.FONT_HERSHEY_SIMPLEX
panels = [
    (to_bgr(img), 'Original (noise + holes)'),
    (to_bgr(opened), 'Opening (removes noise)'),
    (to_bgr(closed), 'Closing (fills holes)'),
    (to_bgr(open_then_close), 'Open then Close (both)'),
]

for panel, label in panels:
    cv2.putText(panel, label, (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

top_row = np.hstack([panels[0][0], panels[1][0]])
bottom_row = np.hstack([panels[2][0], panels[3][0]])
result = np.vstack([top_row, bottom_row])

print(f'White pixels - Original: {np.count_nonzero(img)}')
print(f'After Opening: {np.count_nonzero(opened)} (noise removed)')
print(f'After Closing: {np.count_nonzero(closed)} (holes filled)')
print(f'After Both: {np.count_nonzero(open_then_close)} (cleaned)')

cv2.imshow('Morphology: Opening & Closing', result)
```
