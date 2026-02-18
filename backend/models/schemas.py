"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr
from typing import Any


# ── Kata ──────────────────────────────────────────────────────────────────────

class KataListItem(BaseModel):
    id: int
    slug: str
    title: str
    level: str
    concepts: list[str]


class KataDetail(BaseModel):
    id: int
    slug: str
    title: str
    level: str
    concepts: list[str]
    prerequisites: list[str]
    starter_code: str
    body: str                    # Full Markdown body for frontend rendering
    demo_controls: list[dict[str, Any]]


# ── Execution ─────────────────────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    code: str
    local: bool = False  # True = run on desktop with real camera (no sandbox)


class ExecuteResult(BaseModel):
    image_b64: str | None = None
    logs: str = ""
    error: str = ""


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Progress & Code Saving ───────────────────────────────────────────────────

class CodeVersionSave(BaseModel):
    code: str


class CodeVersionOut(BaseModel):
    id: int
    kata_id: int
    code: str
    saved_at: str


class ProgressOut(BaseModel):
    kata_id: int
    kata_slug: str
    completed_at: str
