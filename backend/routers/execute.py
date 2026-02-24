"""
Router: /api/execute
POST /api/execute — run user code (sandboxed or locally).
POST /api/execute/stop — stop a running local process.
"""

import code

from fastapi import APIRouter, File, Form, UploadFile
from backend.models.schemas import ExecuteResult
from backend.executor.sandbox import run_code, run_local, stop_local

router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("", response_model=ExecuteResult)
async def execute_code(
    code: str = Form(...),
    local: bool = Form(False),
    image: UploadFile | None = File(None),
) -> ExecuteResult:
    """
    Accept user Python/OpenCV code and run it.

    - local=False (default): sandboxed execution, returns image + logs.
    - local=True: launches on desktop with real camera, returns immediately.
    """
    uploaded_files: list[tuple[str, bytes]] = []
    if image is not None and image.filename:
        uploaded_files.append((image.filename, await image.read()))

    # Auto detect webcam usage
    if "cv2.VideoCapture" in code:
        result = run_local(code, uploaded_files)
    else:
        result = run_code(code, uploaded_files)

    return ExecuteResult(
        image_b64=result.get("image_b64"),
        logs=result.get("logs", ""),
        error=result.get("error", ""),
    )


@router.post("/stop")
async def stop_execution():
    """Stop a running local camera process."""
    return stop_local()
