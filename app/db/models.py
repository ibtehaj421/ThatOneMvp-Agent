"""
models.py  (app/db/models.py)
─────────────────────────────
SQLAlchemy ORM models for the ANAM-AI agent.

These tables are used when persistence beyond the in-memory session store is
needed (e.g., crash recovery, audit logs, future multi-server deployment).

Tables
------
  patient_sessions       — one row per chat session
  conversation_messages  — every turn (user + assistant) within a session
  collected_history      — structured CMAS slots extracted from the session
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


class InterviewSession(Base):
    """
    Maps directly to the \`interview_sessions\` table created by the Go backend.
    Uses a composite primary key (patient_id, session_seq) and stores the stateless
    context in JSON columns.
    """
    __tablename__ = "interview_sessions"

    patient_id = Column(Integer, primary_key=True, autoincrement=False)
    session_seq = Column(Integer, primary_key=True, autoincrement=False)

    status = Column(String(20), default="in_progress")
    dialogue_history = Column(JSON, default=list) # Stores Langchain array payload
    extracted_slots = Column(JSON, default=dict)  # Stores CMAS slots

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
