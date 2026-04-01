"""
state.py
────────
Defines the LangGraph GraphState and the CMAS (Comprehensive Medical Attribute
Schema) slot structure used by all nodes in the agent pipeline.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── CMAS Slot schema ──────────────────────────────────────────────────────────

class SymptomSlot(TypedDict, total=False):
    """One reported symptom and its attributes extracted by the LLM."""
    value: str            # e.g. "fever"
    onset: str            # e.g. "3 days ago"
    duration: str         # e.g. "persistent"
    severity: str         # e.g. "7/10"
    location: str         # e.g. "chest"
    progression: str      # e.g. "worsening"
    frequency: str        # e.g. "constant"


class CMASState(TypedDict, total=False):
    """Accumulated structured patient data in CMAS format."""
    chief_complaint: str
    positive_symptoms: list[SymptomSlot]
    negative_symptoms: list[SymptomSlot]
    patient_medical_history: list[str]
    family_medical_history: list[str]
    habits: dict[str, Any]          # {"smoking": "yes", "alcohol": "no", ...}
    medications: list[str]
    medical_tests: list[str]
    exposures: list[str]
    basic_information: dict[str, Any]   # {"name": ..., "age": ..., "gender": ...}


# ── LangGraph Graph State ─────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    The full state threaded through every LangGraph node.

    Fields
    ------
    messages        : Running chat history (user + assistant turns).
                      Uses the built-in `add_messages` reducer so that each
                      node can append without overwriting the whole list.
    json_state      : The accumulated CMAS JSON extracted from the conversation.
    rag_examples    : Few-shot MediTOD turns retrieved from pgvector for the
                      current patient utterance.
    session_complete: Set to True by the conclusion node when all required CMAS
                      slots are filled.
    patient_input   : The raw (possibly localized) text the patient just sent.
    last_question   : The last follow-up question the agent posed (to avoid
                      repeating).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    json_state: CMASState
    rag_examples: list[str]
    session_complete: bool
    patient_input: str
    last_question: str
