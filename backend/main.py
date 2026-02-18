"""
main.py — FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.db import init_db
from backend.routers import katas, execute, auth, progress

app = FastAPI(
    title="OpenCV Interactive Playground",
    description="Learn OpenCV through structured, visual katas.",
    version="0.1.0",
)

# Allow the SolidJS dev server (and any localhost origin) during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(katas.router)
app.include_router(execute.router)
app.include_router(auth.router)
app.include_router(progress.router)


@app.on_event("startup")
def on_startup():
    """Initialize DB and seed katas from JSON files on every startup."""
    init_db()


@app.get("/")
def root():
    return {"message": "OpenCV Playground API is running. Visit /docs for API reference."}
