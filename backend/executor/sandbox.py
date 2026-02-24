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


def _write_uploaded_files(run_dir: Path, uploaded_files: list[tuple[str, bytes]] | None) -> None:
    """Persist uploaded files into the run directory with safe filenames."""
    if not uploaded_files:
        return
    for filename, content in uploaded_files:
        safe_name = Path(filename).name
        if not safe_name:
            continue
        (run_dir / safe_name).write_bytes(content)


def run_code(code: str, uploaded_files: list[tuple[str, bytes]] | None = None) -> dict:
    """
    Execute user-submitted code safely in an isolated subprocess.

    Returns:
        {
            "image_b64": str | None,
            "logs": str,
            "error": str
        }
    """
    try:
        with tempfile.TemporaryDirectory(prefix="kata_run_") as run_dir_str:
            run_dir = Path(run_dir_str)
            tmp_path = run_dir / "kata.py"
            tmp_path.write_text(code, encoding="utf-8")
            _write_uploaded_files(run_dir, uploaded_files)

            result = subprocess.run(
                [sys.executable, str(RUNNER_PATH), str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(run_dir),
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


def run_local(code: str, uploaded_files: list[tuple[str, bytes]] | None = None) -> dict:
    """
    Run code directly on the desktop with real camera, real cv2.imshow,
    and no timeout. Used for live camera katas.

    The process runs in the background — this function returns immediately.
    The OpenCV window appears on the user's desktop.
    """
    global _active_local_process

    # Stop any previously running local process
    stop_local()

    run_dir = tempfile.mkdtemp(prefix="kata_live_")
    run_dir_path = Path(run_dir)
    tmp_path = run_dir_path / "kata.py"
    tmp_path.write_text(code, encoding="utf-8")
    _write_uploaded_files(run_dir_path, uploaded_files)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(tmp_path)],
            cwd=str(run_dir_path),
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
                if tmp_path.exists():
                    os.unlink(tmp_path)
                for child in run_dir_path.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
                run_dir_path.rmdir()
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
            if tmp_path.exists():
                os.unlink(tmp_path)
            for child in run_dir_path.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            run_dir_path.rmdir()
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


