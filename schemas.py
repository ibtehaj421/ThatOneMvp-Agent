from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class SessionCreateRequest(BaseModel):
    patient_id: int
    ailment: str
    patient_name: Optional[str] = None

class SessionCreateResponse(BaseModel):
    session_id: str
    patient_id: int
    ailment: str
    initial_message: str
    created_at: datetime

class MessageRequest(BaseModel):
    session_id: str
    message: str

class MessageResponse(BaseModel):
    session_id: str
    response: str
    extracted_data: Dict[str, Any]
    completeness: float
    should_end: bool
    timestamp: datetime

class SessionStatusResponse(BaseModel):
    session_id: str
    patient_id: int
    ailment: str
    status: str
    message_count: int
    completeness: float
    created_at: datetime
    updated_at: datetime

class SummaryRequest(BaseModel):
    session_id: str

class SummaryResponse(BaseModel):
    session_id: str
    summary: str
    extracted_data: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    generated_at: datetime

class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)