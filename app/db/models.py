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


class PatientSession(Base):
    __tablename__ = "patient_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(255), nullable=True, index=True)   # Optional in MVP
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    patient_name = Column(String(255), nullable=True)
    status = Column(String(50), default="active")                  # active | completed | abandoned | timed_out
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    messages = relationship(
        "ConversationMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.timestamp",
    )
    history_data = relationship(
        "CollectedHistory",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer,
        ForeignKey("patient_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(50), nullable=False)     # user | assistant | system
    content = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)            # renamed from `metadata` to avoid shadowing SA attribute
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("PatientSession", back_populates="messages")


class CollectedHistory(Base):
    __tablename__ = "collected_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer,
        ForeignKey("patient_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    category = Column(String(100), nullable=False)    # CMAS slot type
    field_name = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)              # JSON-serialised slot value
    confidence = Column(Integer, default=100)         # 0–100; reserved for future NLU confidence scoring
    extracted_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("PatientSession", back_populates="history_data")
