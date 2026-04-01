"""
database.py  (app/db/database.py)
──────────────────────────────────
SQLAlchemy engine configuration for the ANAM-AI agent's PostgreSQL/pgvector
instance (same Docker container used by the RAG vector store).

Provides
--------
  Base         — declarative base for ORM models (imported by models.py)
  engine       — SQLAlchemy engine
  SessionLocal — session factory
  get_db()     — FastAPI dependency yielding a DB session
  create_tables() — auto-migrate all ORM models (called at startup)
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import config


# ── Declarative base (must be defined before models.py imports it) ─────────────
Base = declarative_base()

# ── Engine ────────────────────────────────────────────────────────────────────
engine = create_engine(
    config.postgres_url,
    pool_pre_ping=True,       # detect stale connections before use
    pool_size=5,
    max_overflow=10,
    echo=False,               # set True to log all SQL for debugging
)

# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Table auto-migration ───────────────────────────────────────────────────────
def create_tables() -> None:
    """
    Create all ORM tables if they don't already exist.

    Call once at application startup (see app/main.py startup event).
    Import models first so SQLAlchemy knows about them:
        from app.db import models  # noqa: F401
    """
    # Import models so their metadata is registered before create_all
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


# ── Context manager (for scripts / background tasks) ─────────────────────────
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a DB session with automatic commit/rollback."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── FastAPI dependency ─────────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for use in FastAPI route handlers.

    Usage:
        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
