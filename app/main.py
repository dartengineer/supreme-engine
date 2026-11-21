"""App entry point. Run with:  uvicorn app.main:app --reload

The database structure is managed by Alembic. After pulling changes that touch models.py, run:
    alembic upgrade head
"""
from fastapi import FastAPI

from app.config import settings
from app.routers import applications, auth, profile

app = FastAPI(title=settings.app_name)
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(applications.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name}
