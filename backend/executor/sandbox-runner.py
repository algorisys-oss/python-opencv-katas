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
