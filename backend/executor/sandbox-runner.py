"""
sandbox-runner.py — Isolated execution script.

This script is run as a subprocess by sandbox.py.
It intercepts cv2.imshow() and captures the image as a base64 PNG,
then prints it to stdout for the parent process to read.

Security model:
- This script runs as a subprocess (the process boundary IS the sandbox).
- User code is exec()'d with a restricted globals dict containing only
  cv2, np/numpy — so user code cannot directly access os, sys, etc.
- cv2.imshow is monkey-patched to capture images instead of displaying them.
- A 10-second timeout is enforced by the parent process (sandbox.py).
"""

import sys
import os
import base64
import tempfile

# ── Import cv2 and numpy (these are allowed) ──────────────────────────────────

import cv2
import numpy as np

# ── Intercept cv2.imshow ──────────────────────────────────────────────────────

_output_images: list = []


def _capture_imshow(winname, mat):
    """Replace cv2.imshow with image capture."""
    _output_images.append(mat.copy())


cv2.imshow = _capture_imshow
cv2.waitKey = lambda _=0: 0
cv2.destroyAllWindows = lambda: None
cv2.destroyWindow = lambda _: None
cv2.namedWindow = lambda *a, **kw: None
cv2.createTrackbar = lambda *a, **kw: None
cv2.getTrackbarPos = lambda name, *a, **kw: 100
cv2.setMouseCallback = lambda *a, **kw: None


# ── Fake VideoCapture for live camera katas ──────────────────────────────────
# In the sandbox there is no real camera. This fake class returns synthetic
# frames so that live-camera starter code runs without hanging.

_REAL_VIDEO_CAPTURE = cv2.VideoCapture


class _SandboxVideoCapture:
    """Drop-in replacement for cv2.VideoCapture that produces synthetic frames."""

    _MAX_FRAMES = 30  # Stop after this many frames to avoid timeout

    def __init__(self, source=0):
        self._frame_count = 0
        self._width = 640
        self._height = 480
        self._fps = 30.0
        self._opened = True
        # If source is a string (file path), try real VideoCapture
        if isinstance(source, str):
            self._real = _REAL_VIDEO_CAPTURE(source)
            self._use_real = self._real.isOpened()
        else:
            self._real = None
            self._use_real = False

    def isOpened(self):
        if self._use_real:
            return self._real.isOpened()
        return self._opened

    def read(self):
        if self._use_real:
            return self._real.read()
        if self._frame_count >= self._MAX_FRAMES:
            if self._frame_count == self._MAX_FRAMES:
                print(f"[Sandbox] Completed {self._MAX_FRAMES} frames. "
                      "Live camera katas should be tested locally with a real camera.")
                self._frame_count += 1  # Only print once
            return False, None
        self._frame_count += 1
        # Generate a synthetic frame with moving content
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:] = (40, 35, 30)
        # Moving circle to simulate motion
        t = self._frame_count
        cx = int(self._width * (0.3 + 0.4 * np.sin(t * 0.2)))
        cy = int(self._height * (0.3 + 0.2 * np.cos(t * 0.15)))
        cv2.circle(frame, (cx, cy), 40, (0, 180, 255), -1)
        # Frame counter text
        cv2.putText(frame, f"Sandbox frame {t}/{self._MAX_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "No real camera in sandbox", (10, self._height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return True, frame

    def get(self, prop_id):
        if self._use_real:
            return self._real.get(prop_id)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def set(self, prop_id, value):
        if self._use_real:
            return self._real.set(prop_id, value)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
        elif prop_id == cv2.CAP_PROP_FPS:
            self._fps = value
        return True

    def release(self):
        if self._use_real and self._real:
            self._real.release()
        self._opened = False


cv2.VideoCapture = _SandboxVideoCapture

# ── Read user code ────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("ERROR: No code file provided", file=sys.stderr)
    sys.exit(1)

code_file = sys.argv[1]
with open(code_file, "r") as f:
    user_code = f.read()

# ── Execute user code with restricted globals ─────────────────────────────────
# The globals dict only exposes cv2 and numpy — user code cannot access
# os, sys, subprocess, etc. through the globals namespace.

USER_ALLOWED_IMPORTS = {"cv2", "numpy", "np", "math", "random", "time", "collections"}

_builtin_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _user_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in USER_ALLOWED_IMPORTS:
        raise ImportError(
            f"🚫 Import '{name}' is not allowed in the sandbox.\n"
            f"Only `import cv2` and `import numpy as np` are permitted."
        )
    return _builtin_import(name, *args, **kwargs)


restricted_globals = {
    "__builtins__": __builtins__,
    "__import__": _user_import,
    "cv2": cv2,
    "np": np,
    "numpy": np,
}

try:
    exec(compile(user_code, "<kata>", "exec"), restricted_globals)
except Exception as e:
    print(f"EXEC_ERROR:{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

# ── Output result ─────────────────────────────────────────────────────────────

if _output_images:
    img = _output_images[-1]
    if img is not None and img.size > 0:
        success, buf = cv2.imencode(".png", img)
        if success:
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            print(f"IMAGE:{b64}")
        else:
            print("ERROR: Failed to encode image", file=sys.stderr)
    else:
        print("ERROR: Empty image", file=sys.stderr)
else:
    print("INFO: No cv2.imshow() called. Call cv2.imshow('result', your_image) to display output.")
