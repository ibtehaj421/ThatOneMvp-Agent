from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
from models import PatientSession, ConversationMessage, CollectedHistory
from agent import agent
from config import config
import json

class SessionService:
    
    @staticmethod
    def create_session(db: Session, patient_id: int, ailment: str, patient_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new patient history-taking session"""
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session in database
        db_session = PatientSession(
            patient_id=patient_id,
            session_id=session_id,
            ailment=ailment,
            status="active"
        )
        db.add(db_session)
        db.flush()
        
        # Generate initial greeting
        initial_message = agent.start_conversation(ailment, patient_name)
        
        # Save initial message
        system_msg = ConversationMessage(
            session_id=db_session.id,
            role="assistant",
            content=initial_message,
            metadata={"type": "greeting"}
        )
        db.add(system_msg)
        db.commit()
        
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "ailment": ailment,
            "initial_message": initial_message,
            "created_at": db_session.created_at
        }
    
    @staticmethod
    def send_message(db: Session, session_id: str, message: str) -> Dict[str, Any]:
        """Process patient message and generate AI response"""
        
        # Get session
        db_session = db.query(PatientSession).filter(
            PatientSession.session_id == session_id,
            PatientSession.status == "active"
        ).first()
        
        if not db_session:
            raise ValueError("Session not found or inactive")
        
        # Check session timeout
        if db_session.updated_at < datetime.utcnow() - timedelta(minutes=config.SESSION_TIMEOUT_MINUTES):
            db_session.status = "abandoned"
            db.commit()
            raise ValueError("Session has timed out")
        
        # Save patient message
        user_msg = ConversationMessage(
            session_id=db_session.id,
            role="user",
            content=message
        )
        db.add(user_msg)
        db.flush()
        
        # Get conversation history
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == db_session.id
        ).order_by(ConversationMessage.timestamp).all()
        
        # Limit conversation history
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages[-config.MAX_CONVERSATION_HISTORY:]
        ]
        
        # Generate AI response
        response_data = agent.generate_response(
            patient_message=message,
            conversation_history=conversation_history[:-1],  # Exclude the message we just added
            ailment=db_session.ailment
        )
        
        # Save assistant response
        assistant_msg = ConversationMessage(
            session_id=db_session.id,
            role="assistant",
            content=response_data["response"],
            metadata=response_data.get("metadata", {})
        )
        db.add(assistant_msg)
        
        # Update collected history
        extracted_data = response_data.get("extracted_data", {})
        for category, value in extracted_data.items():
            # Check if this field already exists
            existing = db.query(CollectedHistory).filter(
                CollectedHistory.session_id == db_session.id,
                CollectedHistory.category == category
            ).first()
            
            if existing:
                existing.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                existing.extracted_at = datetime.utcnow()
            else:
                history_entry = CollectedHistory(
                    session_id=db_session.id,
                    category=category,
                    field_name=category,
                    value=json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                )
                db.add(history_entry)
        
        # Update session
        db_session.updated_at = datetime.utcnow()
        
        # If conversation should end, update status
        if response_data.get("should_end", False):
            db_session.status = "pending_completion"
        
        db.commit()
        
        return {
            "session_id": session_id,
            "response": response_data["response"],
            "extracted_data": extracted_data,
            "completeness": response_data.get("completeness", 0),
            "should_end": response_data.get("should_end", False),
            "timestamp": datetime.utcnow()
        }
    
    @staticmethod
    def get_session_status(db: Session, session_id: str) -> Dict[str, Any]:
        """Get session status and progress"""
        
        db_session = db.query(PatientSession).filter(
            PatientSession.session_id == session_id
        ).first()
        
        if not db_session:
            raise ValueError("Session not found")
        
        # Count messages
        message_count = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == db_session.id
        ).count()
        
        # Get collected history count
        history_count = db.query(CollectedHistory).filter(
            CollectedHistory.session_id == db_session.id
        ).count()
        
        # Calculate rough completeness
        expected_fields = 10  # Approximate expected fields
        completeness = min(100, (history_count / expected_fields) * 100)
        
        return {
            "session_id": session_id,
            "patient_id": db_session.patient_id,
            "ailment": db_session.ailment,
            "status": db_session.status,
            "message_count": message_count,
            "completeness": completeness,
            "created_at": db_session.created_at,
            "updated_at": db_session.updated_at
        }
    
    @staticmethod
    def generate_summary(db: Session, session_id: str) -> Dict[str, Any]:
        """Generate clinical summary of the session"""
        
        db_session = db.query(PatientSession).filter(
            PatientSession.session_id == session_id
        ).first()
        
        if not db_session:
            raise ValueError("Session not found")
        
        # Get all messages
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == db_session.id
        ).order_by(ConversationMessage.timestamp).all()
        
        conversation_history = [
            {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
            for msg in messages
        ]
        
        # Generate summary
        summary = agent.generate_summary(
            [{"role": msg.role, "content": msg.content} for msg in messages],
            db_session.ailment
        )
        
        # Get all collected data
        collected_data = db.query(CollectedHistory).filter(
            CollectedHistory.session_id == db_session.id
        ).all()
        
        extracted_data = {}
        for entry in collected_data:
            try:
                extracted_data[entry.category] = json.loads(entry.value)
            except:
                extracted_data[entry.category] = entry.value
        
        # Mark session as completed
        db_session.status = "completed"
        db_session.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "session_id": session_id,
            "summary": summary,
            "extracted_data": extracted_data,
            "conversation_history": conversation_history,
            "generated_at": datetime.utcnow()
        }
    
    @staticmethod
    def get_conversation_history(db: Session, session_id: str) -> Dict[str, Any]:
        """Get full conversation history"""
        
        db_session = db.query(PatientSession).filter(
            PatientSession.session_id == session_id
        ).first()
        
        if not db_session:
            raise ValueError("Session not found")
        
        messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == db_session.id
        ).order_by(ConversationMessage.timestamp).all()
        
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ],
            "total_messages": len(messages)
        }
    
    @staticmethod
    def end_session(db: Session, session_id: str) -> bool:
        """Manually end a session"""
        
        db_session = db.query(PatientSession).filter(
            PatientSession.session_id == session_id
        ).first()
        
        if not db_session:
            raise ValueError("Session not found")
        
        db_session.status = "completed"
        db_session.completed_at = datetime.utcnow()
        db.commit()
        
        return True
    
    @staticmethod
    def get_patient_sessions(db: Session, patient_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all sessions for a patient"""
        
        sessions = db.query(PatientSession).filter(
            PatientSession.patient_id == patient_id
        ).order_by(PatientSession.created_at.desc()).limit(limit).all()
        
        return [
            {
                "session_id": s.session_id,
                "ailment": s.ailment,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "completed_at": s.completed_at.isoformat() if s.completed_at else None
            }
            for s in sessions
        ]