"""App entry point. Run with:  uvicorn app.main:app --reload

The database structure is managed by Alembic. After pulling changes that touch models.py, run:
    alembic upgrade head
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers import applications, auth, documents, profile
from app.services import embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.warm_up_embeddings:
        # Loading the model can take a while on Windows; do it in the background so startup isn't blocked.
        threading.Thread(target=embeddings.warm_up, daemon=True, name="embeddings-warm-up").start()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(applications.router)
app.include_router(documents.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name}
