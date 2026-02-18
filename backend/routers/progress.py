"""
Router: /api — user progress and code saving endpoints.
All endpoints require authentication via get_current_user.
"""

from fastapi import APIRouter, HTTPException, Depends

from backend.models.db import get_conn
from backend.models.schemas import CodeVersionSave, CodeVersionOut, ProgressOut
from backend.routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["progress"])


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Return current user profile."""
    return {"id": user["user_id"], "email": user["email"]}


@router.get("/me/progress", response_model=list[ProgressOut])
def get_progress(user: dict = Depends(get_current_user)):
    """List all completed katas for the current user."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT up.kata_id, k.slug as kata_slug, up.completed_at
        FROM user_progress up
        JOIN katas k ON k.id = up.kata_id
        WHERE up.user_id = ?
        ORDER BY up.completed_at DESC
    """, (user["user_id"],)).fetchall()
    conn.close()
    return [ProgressOut(kata_id=r["kata_id"], kata_slug=r["kata_slug"],
                        completed_at=r["completed_at"]) for r in rows]


@router.post("/katas/{slug}/save", response_model=CodeVersionOut)
def save_code(slug: str, body: CodeVersionSave, user: dict = Depends(get_current_user)):
    """Save a code version for a kata."""
    conn = get_conn()
    kata = conn.execute("SELECT id FROM katas WHERE slug = ?", (slug,)).fetchone()
    if not kata:
        conn.close()
        raise HTTPException(status_code=404, detail="Kata not found.")
    cur = conn.execute(
        "INSERT INTO user_code_versions (user_id, kata_id, code) VALUES (?, ?, ?)",
        (user["user_id"], kata["id"], body.code),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM user_code_versions WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    conn.close()
    return CodeVersionOut(id=row["id"], kata_id=row["kata_id"],
                          code=row["code"], saved_at=row["saved_at"])


@router.get("/katas/{slug}/saved")
def get_saved_code(slug: str, user: dict = Depends(get_current_user)):
    """Get most recent saved code for a kata."""
    conn = get_conn()
    kata = conn.execute("SELECT id FROM katas WHERE slug = ?", (slug,)).fetchone()
    if not kata:
        conn.close()
        raise HTTPException(status_code=404, detail="Kata not found.")
    row = conn.execute("""
        SELECT id, kata_id, code, saved_at FROM user_code_versions
        WHERE user_id = ? AND kata_id = ?
        ORDER BY saved_at DESC LIMIT 1
    """, (user["user_id"], kata["id"])).fetchone()
    conn.close()
    if not row:
        return None
    return CodeVersionOut(id=row["id"], kata_id=row["kata_id"],
                          code=row["code"], saved_at=row["saved_at"])


@router.post("/katas/{slug}/complete")
def mark_complete(slug: str, user: dict = Depends(get_current_user)):
    """Mark a kata as completed."""
    conn = get_conn()
    kata = conn.execute("SELECT id FROM katas WHERE slug = ?", (slug,)).fetchone()
    if not kata:
        conn.close()
        raise HTTPException(status_code=404, detail="Kata not found.")
    conn.execute(
        "INSERT OR IGNORE INTO user_progress (user_id, kata_id) VALUES (?, ?)",
        (user["user_id"], kata["id"]),
    )
    conn.commit()
    conn.close()
    return {"status": "completed"}


@router.delete("/katas/{slug}/complete")
def unmark_complete(slug: str, user: dict = Depends(get_current_user)):
    """Remove kata completion mark."""
    conn = get_conn()
    kata = conn.execute("SELECT id FROM katas WHERE slug = ?", (slug,)).fetchone()
    if not kata:
        conn.close()
        raise HTTPException(status_code=404, detail="Kata not found.")
    conn.execute(
        "DELETE FROM user_progress WHERE user_id = ? AND kata_id = ?",
        (user["user_id"], kata["id"]),
    )
    conn.commit()
    conn.close()
    return {"status": "removed"}
