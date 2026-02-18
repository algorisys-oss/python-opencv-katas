"""
sandbox.py — Orchestrates safe subprocess execution of user code.

Two execution modes:
1. Sandboxed (default) — monkey-patched cv2, 10s timeout, captures images
2. Local — runs code directly on the desktop with real camera access
"""

import subprocess
import tempfile
import os
import sys
import threading
from pathlib import Path

RUNNER_PATH = Path(__file__).parent / "sandbox-runner.py"
TIMEOUT_SECONDS = 10

# Track active local processes so we can stop them
_active_local_process: subprocess.Popen | None = None
_active_local_lock = threading.Lock()


def run_code(code: str) -> dict:
    """
    Execute user-submitted code safely in an isolated subprocess.

    Returns:
        {
            "image_b64": str | None,
            "logs": str,
            "error": str
        }
    """
    # Write code to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kata_"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, str(RUNNER_PATH), tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        image_b64 = None
        logs_lines = []
        error = ""

        for line in stdout.splitlines():
            if line.startswith("IMAGE:"):
                image_b64 = line[len("IMAGE:"):]
            elif line.startswith("INFO:"):
                logs_lines.append(line[len("INFO:"):])
            else:
                logs_lines.append(line)

        if stderr:
            for line in stderr.splitlines():
                if line.startswith("EXEC_ERROR:"):
                    error = _make_friendly_error(line[len("EXEC_ERROR:"):])
                else:
                    logs_lines.append(line)

        return {
            "image_b64": image_b64,
            "logs": "\n".join(logs_lines),
            "error": error,
        }

    except subprocess.TimeoutExpired:
        return {
            "image_b64": None,
            "logs": "",
            "error": f"⏱ Execution timed out after {TIMEOUT_SECONDS} seconds. "
                     "Check for infinite loops.",
        }
    except Exception as e:
        return {
            "image_b64": None,
            "logs": "",
            "error": f"Execution failed: {e}",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_local(code: str) -> dict:
    """
    Run code directly on the desktop with real camera, real cv2.imshow,
    and no timeout. Used for live camera katas.

    The process runs in the background — this function returns immediately.
    The OpenCV window appears on the user's desktop.
    """
    global _active_local_process

    # Stop any previously running local process
    stop_local()

    # Write code to a temp file (not auto-deleted — process needs it)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kata_live_"
    )
    tmp.write(code)
    tmp_path = tmp.name
    tmp.close()

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", tmp_path],
        )

        with _active_local_lock:
            _active_local_process = proc

        # Clean up temp file after process ends (in a background thread)
        def _cleanup():
            proc.wait()
            with _active_local_lock:
                global _active_local_process
                if _active_local_process is proc:
                    _active_local_process = None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        threading.Thread(target=_cleanup, daemon=True).start()

        return {
            "image_b64": None,
            "logs": "Running on your desktop — an OpenCV window should appear.\n"
                    "Press 'q' in the OpenCV window to quit.",
            "error": "",
        }

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return {
            "image_b64": None,
            "logs": "",
            "error": f"Failed to launch: {e}",
        }


def stop_local() -> dict:
    """Stop the currently running local process (if any)."""
    global _active_local_process
    with _active_local_lock:
        proc = _active_local_process
        _active_local_process = None

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        return {"stopped": True, "message": "Local process stopped."}

    return {"stopped": False, "message": "No local process was running."}


def _make_friendly_error(raw: str) -> str:
    """Convert cryptic Python errors into learner-friendly messages."""
    if "ImportError" in raw or "ModuleNotFoundError" in raw:
        return (
            f"🚫 Import blocked: {raw}\n"
            "Only `import cv2` and `import numpy as np` are allowed."
        )
    if "SyntaxError" in raw:
        return f"✏️ Syntax error in your code: {raw}"
    if "NameError" in raw:
        return f"❓ Name not found: {raw}\nDid you define this variable?"
    if "TypeError" in raw:
        return f"🔧 Type error: {raw}"
    if "AttributeError" in raw:
        return f"🔍 Attribute error: {raw}\nCheck the OpenCV function name."
    return f"❌ Error: {raw}"
