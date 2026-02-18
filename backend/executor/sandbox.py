"""
sandbox.py — Orchestrates safe subprocess execution of user code.

Writes user code to a temp file, runs sandbox-runner.py as a subprocess
with a timeout, and parses the output.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

RUNNER_PATH = Path(__file__).parent / "sandbox-runner.py"
TIMEOUT_SECONDS = 10


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
