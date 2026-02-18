"""
SQLite database initialization and helpers.
Tables are created on startup; katas are seeded from Markdown files
with YAML frontmatter (*.md in backend/data/katas/).
"""

import sqlite3
import json
import re
from pathlib import Path

import frontmatter

DB_PATH = Path(__file__).parent.parent / "data" / "opencv-katas.db"
KATAS_DIR = Path(__file__).parent.parent / "data" / "katas"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    UNIQUE NOT NULL,
            password_hash TEXT  NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS katas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT    UNIQUE NOT NULL,
            title       TEXT    NOT NULL,
            level       TEXT    NOT NULL,
            concepts    TEXT    NOT NULL,
            content_json TEXT   NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_progress (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            kata_id     INTEGER NOT NULL REFERENCES katas(id),
            completed_at TEXT   DEFAULT (datetime('now')),
            UNIQUE(user_id, kata_id)
        );

        CREATE TABLE IF NOT EXISTS user_code_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            kata_id     INTEGER NOT NULL REFERENCES katas(id),
            code        TEXT    NOT NULL,
            saved_at    TEXT    DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    _seed_katas(conn)
    conn.close()


def _extract_starter_code(body: str) -> str:
    """Extract the ```python block under '## Starter Code', or the last one."""
    # Try the block right after the "## Starter Code" heading
    match = re.search(
        r"## Starter Code\s*```python\n(.*?)```", body, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    # Fallback: last ```python block in the file
    matches = re.findall(r"```python\n(.*?)```", body, re.DOTALL)
    return matches[-1].strip() if matches else ""


def _seed_katas(conn: sqlite3.Connection) -> None:
    """Load kata .md files into the katas table (upsert by slug)."""
    if not KATAS_DIR.exists():
        return

    cur = conn.cursor()
    for md_file in sorted(KATAS_DIR.glob("*.md")):
        post = frontmatter.load(str(md_file))
        meta = post.metadata
        body = post.content  # Markdown body (everything after frontmatter)

        slug = meta["slug"]
        title = meta["title"]
        level = meta["level"]
        concepts = meta.get("concepts", [])
        prerequisites = meta.get("prerequisites", [])
        starter_code = _extract_starter_code(body)

        content = {
            "slug": slug,
            "title": title,
            "level": level,
            "concepts": concepts,
            "prerequisites": prerequisites,
            "starter_code": starter_code,
            "body": body,          # Full Markdown body for frontend rendering
            "demo_controls": [],
        }

        cur.execute("""
            INSERT INTO katas (slug, title, level, concepts, content_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                title        = excluded.title,
                level        = excluded.level,
                concepts     = excluded.concepts,
                content_json = excluded.content_json
        """, (slug, title, level, json.dumps(concepts), json.dumps(content)))

    conn.commit()
