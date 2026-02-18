"""
Router: /api/execute
POST /api/execute — run user code (sandboxed or locally).
POST /api/execute/stop — stop a running local process.
"""

from fastapi import APIRouter
from backend.models.schemas import ExecuteRequest, ExecuteResult
from backend.executor.sandbox import run_code, run_local, stop_local

router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("", response_model=ExecuteResult)
async def execute_code(req: ExecuteRequest) -> ExecuteResult:
    """
    Accept user Python/OpenCV code and run it.

    - local=False (default): sandboxed execution, returns image + logs.
    - local=True: launches on desktop with real camera, returns immediately.
    """
    if req.local:
        result = run_local(req.code)
    else:
        result = run_code(req.code)

    return ExecuteResult(
        image_b64=result.get("image_b64"),
        logs=result.get("logs", ""),
        error=result.get("error", ""),
    )


@router.post("/stop")
async def stop_execution():
    """Stop a running local camera process."""
    return stop_local()
