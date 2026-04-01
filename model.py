from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class PatientSession(Base):
    __tablename__ = "patient_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=False, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    ailment = Column(String(255), nullable=False)
    status = Column(String(50), default="active")  # active, completed, abandoned
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    history_data = relationship("CollectedHistory", back_populates="session", cascade="all, delete-orphan")

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("patient_sessions.id"), nullable=False)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("PatientSession", back_populates="messages")

class CollectedHistory(Base):
    __tablename__ = "collected_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("patient_sessions.id"), nullable=False)
    category = Column(String(100), nullable=False)  # chief_complaint, symptoms, duration, severity, etc.
    field_name = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Integer, default=100)  # 0-100
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("PatientSession", back_populates="history_data")

class MedicalKnowledge(Base):
    __tablename__ = "medical_knowledge"
    
    id = Column(Integer, primary_key=True, index=True)
    disease = Column(String(255), nullable=False, index=True)
    symptom = Column(String(255), nullable=False)
    follow_up_questions = Column(JSON, nullable=True)
    severity_indicators = Column(JSON, nullable=True)
    related_conditions = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)