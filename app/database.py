"""Database connection and session handling."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings

if settings.database_url.startswith("sqlite"):
    # Only used by the tests: one shared in-memory database.
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
else:
    engine = create_engine(settings.database_url)

SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: opens a session per request and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()