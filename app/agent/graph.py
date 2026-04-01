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
      1. The LLM explicitly sets session_complete = True, OR
      2. The conversation exceeds MAX_CONVERSATION_TURNS (hard safety limit
         ported from the original agent.py 30-turn guard).
    """
    if state.get("session_complete", False):
        return "soap"

    # Hard turn limit — prevents runaway sessions
    messages: list[BaseMessage] = state.get("messages", [])
    if len(messages) >= config.MAX_CONVERSATION_TURNS:
        return "soap"

    return "localize"


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

    # Conditional: after generate, either loop or conclude
    graph.add_conditional_edges(
        "generate",
        _should_conclude,
        {
            "localize": "localize",   # loop back for next patient turn
            "soap": "soap",           # write the SOAP note
        },
    )

    # Terminal transition
    graph.add_edge("soap", END)

    return graph


# ── Compiled graph (singleton) ────────────────────────────────────────────────
# The compiled graph is what FastAPI calls. We compile once at import time.
compiled_graph = build_graph().compile()
