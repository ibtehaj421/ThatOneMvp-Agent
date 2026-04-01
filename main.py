from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from config import config
from database import db_manager
from schemas import (
    SessionCreateRequest, SessionCreateResponse,
    MessageRequest, MessageResponse,
    SessionStatusResponse, SummaryRequest, SummaryResponse,
    ConversationHistoryResponse, ErrorResponse
)
from services import SessionService

# Initialize FastAPI app
app = FastAPI(
    title="Medical History AI Agent API",
    description="AI-powered patient history-taking assistant",
    version="1.0.0"
)

# Configure CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    return next(db_manager.get_db())

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        db_manager.create_tables()
        config.validate()
        print("✅ Database tables created successfully")
        print(f"✅ Server starting on {config.API_HOST}:{config.API_PORT}")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical History AI Agent",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "ai_model": config.MODEL_NAME,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/sessions/create", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new patient history-taking session
    
    - **patient_id**: ID of the patient from your main database
    - **ailment**: The condition/disease to take history for
    - **patient_name**: Optional patient name for personalization
    """
    try:
        result = SessionService.create_session(
            db=db,
            patient_id=request.patient_id,
            ailment=request.ailment,
            patient_name=request.patient_name
        )
        return SessionCreateResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

@app.post("/sessions/message", response_model=MessageResponse)
async def send_message(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    """
    Send a patient message and receive AI response
    
    - **session_id**: The session ID from create_session
    - **message**: The patient's message/response
    """
    try:
        result = SessionService.send_message(
            db=db,
            session_id=request.session_id,
            message=request.message
        )
        return MessageResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )

@app.get("/sessions/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current status and progress of a session
    
    - **session_id**: The session ID
    """
    try:
        result = SessionService.get_session_status(db=db, session_id=session_id)
        return SessionStatusResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )

@app.post("/sessions/summary", response_model=SummaryResponse)
async def generate_summary(
    request: SummaryRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a clinical summary of the conversation
    
    This endpoint marks the session as completed and generates a professional
    summary suitable for healthcare providers.
    
    - **session_id**: The session ID
    """
    try:
        result = SessionService.generate_summary(db=db, session_id=request.session_id)
        return SummaryResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

@app.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the full conversation history for a session
    
    - **session_id**: The session ID
    """
    try:
        result = SessionService.get_conversation_history(db=db, session_id=session_id)
        return ConversationHistoryResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def end_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Manually end a session
    
    - **session_id**: The session ID
    """
    try:
        SessionService.end_session(db=db, session_id=session_id)
        return {"message": "Session ended successfully", "session_id": session_id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}"
        )

@app.get("/patients/{patient_id}/sessions")
async def get_patient_sessions(
    patient_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get all sessions for a specific patient
    
    - **patient_id**: The patient ID
    - **limit**: Maximum number of sessions to return (default: 10)
    """
    try:
        sessions = SessionService.get_patient_sessions(
            db=db,
            patient_id=patient_id,
            limit=limit
        )
        return {
            "patient_id": patient_id,
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get patient sessions: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )