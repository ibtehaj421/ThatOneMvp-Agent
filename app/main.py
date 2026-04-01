"""
main.py  (app/main.py)
──────────────────────
FastAPI application entry point for the ANAM-AI History Taking Agent.

Endpoints
─────────
  POST  /session/start   — Create a new session, return session_id + greeting
  POST  /session/chat    — Send a patient utterance, receive the agent's reply
  GET   /session/export  — Conclude the session and return the SOAP note

In-memory session store (MVP):
  Sessions are tracked in a module-level dict keyed by session_id. For
  production, replace with a Redis or DB-backed store.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.config import config
from app.agent.graph import compiled_graph
from app.agent.state import GraphState, CMASState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ANAM-AI History Taking Agent",
    description=(
        "An AI-powered patient history-taking agent tailored for the Pakistani "
        "healthcare ecosystem. Runs locally via Ollama + LangGraph."
    ),
    version="1.0.0-mvp",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
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
        from app.db.database import create_tables
        create_tables()
        print("✅ Database tables ready.")
        print(f"✅ Ollama model : {config.OLLAMA_MODEL_NAME} @ {config.OLLAMA_BASE_URL}")
        print(f"✅ Server       : {config.API_HOST}:{config.API_PORT}")
    except Exception as exc:
        # Non-fatal — agent still works with in-memory sessions if DB is down
        print(f"⚠️  DB startup warning: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store  (keyed: session_id → GraphState + created_at)
# ─────────────────────────────────────────────────────────────────────────────

# Each value is a tuple: (GraphState, created_at: datetime)
_sessions: dict[str, tuple[GraphState, datetime]] = {}

GREETING_MESSAGE = (
    "Assalam-u-Alaikum! I'm ANAM, your AI medical history assistant. "
    "I'll ask you a few questions to help your doctor understand your condition better. "
    "Everything is confidential.\n\n"
    "To begin — what brings you in today? Please describe your main concern."
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    patient_id: int | str | None = None
    patient_name: str | None = None


class StartSessionResponse(BaseModel):
    session_id: str
    message: str
    created_at: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    json_state: dict[str, Any]
    session_complete: bool
    timestamp: str


class ExportResponse(BaseModel):
    session_id: str
    soap_note: str
    json_state: dict[str, Any]
    generated_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "healthy",
        "service": "ANAM-AI History Taking Agent",
        "version": "1.0.0-mvp",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "ollama_model": config.OLLAMA_MODEL_NAME,
        "ollama_url": config.OLLAMA_BASE_URL,
        "active_sessions": len(_sessions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /session/start
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/session/start",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Session"],
    summary="Start a new history-taking session",
)
async def session_start(request: StartSessionRequest):
    """
    Create a fresh LangGraph session.

    Returns a `session_id` that must be passed to every subsequent `/session/chat`
    and `/session/export` call.
    """
    session_id = str(uuid.uuid4())

    # Personalise greeting if patient name provided
    greeting = GREETING_MESSAGE
    if request.patient_name:
        greeting = (
            f"Assalam-u-Alaikum, {request.patient_name}! I'm ANAM, your AI medical "
            "history assistant. I'll ask you a few questions to help your doctor "
            "understand your condition better.\n\n"
            "To begin — what brings you in today?"
        )

    # Initialise GraphState for this session
    initial_state: GraphState = {
        "messages": [AIMessage(content=greeting)],
        "json_state": {},
        "rag_examples": [],
        "session_complete": False,
        "patient_input": "",
        "last_question": greeting,
    }
    _sessions[session_id] = (initial_state, datetime.utcnow())

    return StartSessionResponse(
        session_id=session_id,
        message=greeting,
        created_at=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /session/chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/session/chat",
    response_model=ChatResponse,
    tags=["Session"],
    summary="Send a patient message and receive the agent's next question",
)
async def session_chat(request: ChatRequest):
    """
    Progress the LangGraph state machine one turn.

    - Injects the patient utterance into the graph.
    - Runs localize → context → generate.
    - Returns the agent's next question + current CMAS JSON state.

    If `session_complete` is True, call `/session/export` to get the SOAP note.
    """
    session_id = request.session_id

    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Call /session/start first.",
        )

    current_state, created_at = _sessions[session_id]

    # ── Session timeout check (ported from old services.py) ───────────────────
    timeout_delta = timedelta(minutes=config.SESSION_TIMEOUT_MINUTES)
    if datetime.utcnow() - created_at > timeout_delta:
        # Mark timed out but keep state so export still works
        timed_out_state = {**current_state, "session_complete": True}
        _sessions[session_id] = (timed_out_state, created_at)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=(
                f"Session has timed out after {config.SESSION_TIMEOUT_MINUTES} minutes. "
                "Call /session/export to retrieve whatever data was collected."
            ),
        )

    if current_state.get("session_complete", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is already complete. Call /session/export to retrieve the SOAP note.",
        )

    # Inject the new patient message into the state
    updated_state = {**current_state, "patient_input": request.message}

    try:
        # Run ONE step of the graph (localize → context → generate)
        # We drive the graph manually per-turn so that each HTTP call is a
        # single user-turn round-trip rather than running to completion.
        result_state = compiled_graph.invoke(updated_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline error: {str(exc)}",
        )

    # Persist the updated state (keep original created_at)
    _sessions[session_id] = (result_state, created_at)

    # Extract the last assistant message as the reply
    messages = result_state.get("messages", [])
    last_ai_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
        "I'm processing your response. Could you please repeat that?",
    )

    return ChatResponse(
        session_id=session_id,
        reply=last_ai_msg,
        json_state=result_state.get("json_state", {}),
        session_complete=result_state.get("session_complete", False),
        timestamp=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /session/export
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/session/export",
    response_model=ExportResponse,
    tags=["Session"],
    summary="Export the SOAP note for a completed (or ongoing) session",
)
async def session_export(session_id: str):
    """
    Force the session to conclude and generate/return the SOAP clinical note.

    Can be called at any point — even mid-session — to force an early export.
    The SOAP note will reflect whatever CMAS data has been collected so far.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    current_state, created_at = _sessions[session_id]

    # If the session isn't already complete, force it to conclude
    if not current_state.get("session_complete", False):
        from app.agent.nodes import soap_node
        current_state = soap_node(current_state)
        _sessions[session_id] = (current_state, created_at)

    # The SOAP note is the last AIMessage added by soap_node
    messages = current_state.get("messages", [])
    soap_note = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
        "No SOAP note generated.",
    )

    return ExportResponse(
        session_id=session_id,
        soap_note=soap_note,
        json_state=current_state.get("json_state", {}),
        generated_at=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /session/history
# ─────────────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    total_messages: int
    session_complete: bool
    json_state: dict[str, Any]


@app.get(
    "/session/history",
    response_model=SessionHistoryResponse,
    tags=["Session"],
    summary="Retrieve the full conversation history for a session",
)
async def session_history(session_id: str):
    """
    Return the complete turn-by-turn conversation log and current CMAS JSON
    state for a session (active or completed).

    Ported from the old services.py `get_conversation_history` method.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    current_state, _ = _sessions[session_id]
    messages: list[BaseMessage] = current_state.get("messages", [])

    history = [
        HistoryMessage(
            role="assistant" if isinstance(m, AIMessage) else "user",
            content=m.content,
        )
        for m in messages
    ]

    return SessionHistoryResponse(
        session_id=session_id,
        messages=history,
        total_messages=len(history),
        session_complete=current_state.get("session_complete", False),
        json_state=current_state.get("json_state", {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
