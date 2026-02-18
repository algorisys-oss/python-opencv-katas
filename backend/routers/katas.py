"""
Router: /api/katas
GET /api/katas       — list all katas (summary)
GET /api/katas/{slug} — full kata detail
"""

import json
from fastapi import APIRouter, HTTPException
from backend.models.db import get_conn
from backend.models.schemas import KataListItem, KataDetail

router = APIRouter(prefix="/api/katas", tags=["katas"])


@router.get("", response_model=list[KataListItem])
def list_katas():
    """Return all katas as a summary list (id, slug, title, level, concepts)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, slug, title, level, concepts FROM katas ORDER BY slug"
    ).fetchall()
    conn.close()
    return [
        KataListItem(
            id=row["id"],
            slug=row["slug"],
            title=row["title"],
            level=row["level"],
            concepts=json.loads(row["concepts"]),
        )
        for row in rows
    ]


@router.get("/{slug}", response_model=KataDetail)
def get_kata(slug: str):
    """Return full kata detail including description, tips, and starter code."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, slug, title, level, concepts, content_json FROM katas WHERE slug = ?",
        (slug,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Kata '{slug}' not found.")

    content = json.loads(row["content_json"])
    return KataDetail(
        id=row["id"],
        slug=row["slug"],
        title=row["title"],
        level=row["level"],
        concepts=json.loads(row["concepts"]),
        body=content.get("body", ""),
        prerequisites=content.get("prerequisites", []),
        starter_code=content.get("starter_code", ""),
        demo_controls=content.get("demo_controls", []),
    )
