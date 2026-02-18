"""
Router: /api/execute
POST /api/execute — run user code safely and return image + logs.
"""

from fastapi import APIRouter
from backend.models.schemas import ExecuteRequest, ExecuteResult
from backend.executor.sandbox import run_code

router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("", response_model=ExecuteResult)
async def execute_code(req: ExecuteRequest) -> ExecuteResult:
    """
    Accept user Python/OpenCV code, run it in a sandbox,
    and return the output image (base64 PNG), logs, and any errors.
    """
    result = run_code(req.code)
    return ExecuteResult(
        image_b64=result.get("image_b64"),
        logs=result.get("logs", ""),
        error=result.get("error", ""),
    )
