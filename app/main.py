"""App entry point. Run with:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import models  # noqa: F401  (imports the tables so create_all can see them)
from app.config import settings
from app.database import Base, engine
from app.routers import auth, profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creates any missing tables on startup. We'll replace this with Alembic migrations later.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(auth.router)
app.include_router(profile.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "app": settings.app_name}
