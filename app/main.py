"""
main.py  (app/main.py)
──────────────────────
FastAPI application entry point for the ANAM-AI History Taking Agent.

Endpoints
─────────
  POST  /chat    — Main proxy endpoint connected to Go Backend. Expects patient_id, session_seq, message.

Stateless execution loop:
1. Fetch `InterviewSession` from the database.
2. Deserialize `dialogue_history` and `extracted_slots`.
3. Pass updated state to LangGraph.
4. Save the new context state back to DB.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any
import json

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.config import config
from app.agent.graph import compiled_graph
from app.agent.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, messages_from_dict, messages_to_dict

from app.db.database import get_db, create_tables
from app.db.models import InterviewSession

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ANAM-AI History Taking Agent",
    description=(
        "An AI-powered patient history-taking agent tailored for the Pakistani "
        "healthcare ecosystem. Runs statelessly via PostgreSQL."
    ),
    version="2.0.0-stateless-mvp",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Auto-migrate ORM tables on server start."""
    try:
        create_tables()
        print("✅ Database tables ready.")
        print(f"✅ Ollama model : {config.OLLAMA_MODEL_NAME} @ {config.OLLAMA_BASE_URL}")
        print(f"✅ Server       : {config.API_HOST}:{config.API_PORT}")
    except Exception as exc:
        print(f"⚠️  DB startup warning: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    patient_id: int
    session_seq: int
    message: str

class ChatResponse(BaseModel):
    reply: str
    is_complete: bool


GREETING_MESSAGE = (
    "Assalam-u-Alaikum! I'm ANAM, your AI medical history assistant. "
    "I'll ask you a few questions to help your doctor understand your condition better. "
    "Everything is confidential.\n\n"
    "To begin — what brings you in today? Please describe your main concern."
)

# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "healthy",
        "service": "ANAM-AI History Taking Agent",
        "version": "2.0.0-stateless-mvp",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Session"],
    summary="Stateless chat processor (called by Go Backend)",
)
async def session_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Progress the LangGraph state machine one turn statelessly.

    - Fetch exact session from DB.
    - Hydrate graph message and JSON state.
    - Run localize → context → generate.
    - Update state into DB.
    - Clear memory and answer.
    """
    
    # 1. Fetch the targeted session
    try:
        session_record = db.query(InterviewSession).filter_by(
            patient_id=request.patient_id, 
            session_seq=request.session_seq
        ).first()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(exc)}",
        )

    if not session_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with patient_id={request.patient_id} and seq={request.session_seq} not found.",
        )

    if session_record.status == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is already complete.",
        )

    # 2. Parse State
    history_data = session_record.dialogue_history
    if isinstance(history_data, str):
        try:
            history_data = json.loads(history_data)
        except json.JSONDecodeError:
            history_data = []
    
    slots_data = session_record.extracted_slots
    if isinstance(slots_data, str):
        try:
            slots_data = json.loads(slots_data)
        except json.JSONDecodeError:
            slots_data = {}

    messages: list[BaseMessage] = messages_from_dict(history_data) if history_data else []

    # Reinstall greeting if this is literally the very first chat exchange (DB initialized by Backend only)
    if not messages:
        messages.append(AIMessage(content=GREETING_MESSAGE))

    messages.append(HumanMessage(content=request.message))

    # Construct the GraphState payload structure required by Langgraph 
    # Provide the last question explicitly to inform the model context processing.
    last_question = ""
    for m in reversed(messages[:-1]): 
        if isinstance(m, AIMessage):
            last_question = m.content
            break

    state: GraphState = {
        "messages": messages,
        "json_state": slots_data,
        "rag_examples": [],
        "session_complete": False,
        "patient_input": request.message,
        "last_question": last_question
    }

    # 3. Model Inference Round-Trip (Graph execute)
    try:
        result_state = compiled_graph.invoke(state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline error: {str(exc)}",
        )

    # Extract final assistant reply out of graph execution buffer
    out_msgs = result_state.get("messages", [])
    reply = "I'm processing your response. Could you please repeat that?"
    for m in reversed(out_msgs):
        if isinstance(m, AIMessage):
            reply = m.content
            break

    # 4. Save Changes to Source of Truth!
    is_complete = result_state.get("session_complete", False)
    
    # Check if a SOAP node forced an extra "concluding SOAP report msg".
    if is_complete and "soap_note" in last_question.lower():
        # Edge case: If soap_node injected the SOAP.
        session_record.status = "completed"
    elif is_complete:
        session_record.status = "completed"
    else:
        session_record.status = "in_progress"

    # Serialize back as dict strings/arrays for SQLAlchemy
    session_record.dialogue_history = messages_to_dict(out_msgs)
    session_record.extracted_slots = result_state.get("json_state", {})

    db.commit()

    # 5. Return Output
    return ChatResponse(
        reply=reply,
        is_complete=is_complete,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
