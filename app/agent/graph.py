"""
graph.py
────────
LangGraph state machine definition, edge wiring, and compiled graph export.

Topology
────────
    START
      │
      ▼
  localize_node              ← sanitize patient input
      │
      ▼
  context_node               ← fetch MediTOD few-shot examples from pgvector
      │
      ▼
  generate_node              ← call local Ollama, extract CMAS slots
      │
      ▼  (conditional edge)
      ├─ session_complete=True   ──► soap_node ──► END
      ├─ max turns exceeded      ──► soap_node ──► END  (safety fallback)
      └─ session_complete=False  ──► localize_node (loop)
"""

from __future__ import annotations

import os
import sys

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.agent.state import GraphState
from app.agent.nodes import localize_node, context_node, generate_node, soap_node
from app.config import config


# ── Router: decide whether to loop or conclude ────────────────────────────────

def _should_conclude(state: GraphState) -> str:
    """
    Conditional edge function used after the generate node.

    Concludes (→ soap) when:
      1. The LLM explicitly sets session_complete = True AND mandatory slots are filled.
      2. The conversation exceeds MAX_CONVERSATION_TURNS (hard safety limit).
    """
    session_complete = state.get("session_complete", False)
    json_state = state.get("json_state", {})
    
    # Validation: Ensure we have at least a chief complaint and one symptom
    has_chief = bool(json_state.get("chief_complaint"))
    has_pos_symptoms = len(json_state.get("positive_symptoms", [])) > 0
    
    if session_complete:
        if has_chief and has_pos_symptoms:
            return "soap"
        else:
            # If LLM tries to end without data, force it to continue
            print("[AUTO-RECOVER] 🔄 LLM tried to end session, but mandatory slots were missing. Forcing continuation.")
            return "end"

    # Hard turn limit — prevents runaway sessions
    messages: list[BaseMessage] = state.get("messages", [])
    if len(messages) >= config.MAX_CONVERSATION_TURNS:
        return "soap"

    # End the graph execution for this turn and yield back to the FastAPI user
    return "end"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("localize", localize_node)
    graph.add_node("context", context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("soap", soap_node)

    # Entry point
    graph.add_edge(START, "localize")

    # Linear pipeline: localize → context → generate
    graph.add_edge("localize", "context")
    graph.add_edge("context", "generate")

    # Conditional: after generate, either end the turn or conclude with soap
    graph.add_conditional_edges(
        "generate",
        _should_conclude,
        {
            "end": END,               # Stop execution, yield to user
            "soap": "soap",           # Write the SOAP note
        },
    )

    # Terminal transition
    graph.add_edge("soap", END)

    return graph


# ── Compiled graph (singleton) ────────────────────────────────────────────────
# The compiled graph is what FastAPI calls. We compile once at import time.
compiled_graph = build_graph().compile()
