"""
Router: /api/auth
POST /api/auth/register — create account, return JWT
POST /api/auth/login    — verify credentials, return JWT
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Header
from passlib.context import CryptContext
from jose import jwt, JWTError

from backend.models.db import get_conn
from backend.models.schemas import UserRegister, UserLogin, Token

router = APIRouter(prefix="/api/auth", tags=["auth"])

# In production, load from env. For MVP this is fine.
SECRET_KEY = "opencv-katas-secret-change-in-prod"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash(password: str) -> str:
    return pwd_ctx.hash(password)


def _verify(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def _make_token(user_id: int, email: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


@router.post("/register", response_model=Token)
def register(body: UserRegister):
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM users WHERE email = ?", (body.email,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed = _hash(body.password)
    cur = conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (body.email, hashed),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return Token(access_token=_make_token(user_id, body.email))


@router.post("/login", response_model=Token)
def login(body: UserLogin):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE email = ?", (body.email,)
    ).fetchone()
    conn.close()

    if not row or not _verify(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return Token(access_token=_make_token(row["id"], body.email))


# ── Auth dependencies ────────────────────────────────────────────────────────

def get_current_user(authorization: str = Header(...)) -> dict:
    """Strict auth: raises 401 if token missing or invalid."""
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme.")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": int(payload["sub"]), "email": payload["email"]}
    except (JWTError, ValueError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_optional_user(authorization: str | None = Header(default=None)) -> dict | None:
    """Lenient auth: returns None for anonymous users."""
    if not authorization:
        return None
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            return None
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": int(payload["sub"]), "email": payload["email"]}
    except (JWTError, ValueError, KeyError):
        return None
