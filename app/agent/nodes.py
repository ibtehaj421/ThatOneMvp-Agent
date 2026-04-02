"""
nodes.py
────────
Individual LangGraph node functions.

Pipeline order (see graph.py):
    localize_node → context_node → generate_node → [conclusion_checker edge]
                                                         ↓ (if complete)
                                                     soap_node → END
                                                         ↓ (otherwise)
                                                     loop back → localize_node
"""

from __future__ import annotations

import json
import os
import sys
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.agent.state import GraphState, CMASState
from app.agent.llm import llm
from app.config import config


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Localize
# ─────────────────────────────────────────────────────────────────────────────

def localize_node(state: GraphState) -> GraphState:
    """
    Pass-through localization node (English MVP).

    Sanitizes the raw patient input, strips obvious noise, and stores the
    clean string in `state["patient_input"]`.

    In a future sprint this will route through a Urdu ↔ English translation
    pipeline via the localization module.
    """
    from app.utils.localization import localize_input

    raw = state.get("patient_input", "")
    clean = localize_input(raw)
    return {**state, "patient_input": clean}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — RAG Context
# ─────────────────────────────────────────────────────────────────────────────

def context_node(state: GraphState) -> GraphState:
    """
    Query pgvector with the patient's utterance.

    Retrieves the top-K MediTOD state-transition examples and stores them in
    `state["rag_examples"]` for the generative node to inject as few-shot guides.
    """
    from app.rag.vector_store import vector_store

    patient_input = state.get("patient_input", "")
    examples = vector_store.retrieve_few_shot_examples(patient_input)
    
    # ── Console log so you can visually verify RAG is working during chat ──
    print(f"\n[RAG] 🔍 Retrieved {len(examples)} relevant examples from MediTOD for input: '{patient_input}'")
    
    return {**state, "rag_examples": examples}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Generate (core LLM call)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are ANAM, a professional, highly empathetic AI medical history-taking assistant operating in a Pakistani hospital setting.
Your task is to conduct a structured patient history interview following a comprehensive clinical template. You must sound like a caring, polite human doctor, not a robotic questionnaire.

CURRENT CMAS SLOT STATE (JSON):
{json_state}

FEW-SHOT MEDITOD EXAMPLES (use these to guide your tone and next question):
{examples}

CORE RULES:
1. ASK ONE QUESTION AT A TIME: Never ask multiple questions in a single conversational turn.
2. EMPATHETIC PHRASING: Use natural, conversational transitions. Acknowledge patient distress (e.g., "I'm sorry to hear you're in pain. Can you tell me exactly when this started?").
3. DO NOT DIAGNOSE: Never prescribe, diagnose, or give medical advice.
4. EMERGENCY PROTOCOL: If a patient mentions chest pain, severe bleeding, difficulty breathing, or unconsciousness, immediately instruct them to seek emergency care and set "session_complete": true.
5. JSON FIRST, THEN CONVERSATION: ALWAYS output a valid JSON block FIRST, followed immediately by your conversational response.

INTERVIEW FLOW (Progress through these stages naturally):
1. INTRODUCTION & COMFORT: Check comfort and ask for the chief complaint.
2. SCREENING: Ask "Apart from this, have you noticed anything else?" until they say no.
3. SYMPTOM DEEP-DIVE (SOCRATES): Find the onset, severity, duration, location, progression, and triggers for each symptom.
4. I.C.E. (Ideas, Concerns, Expectations): Ask what they think it is, their worries, and what they hope to get from the visit.
5. SYSTEMS REVIEW: Ask about fever, weight changes, fatigue, bowel/urinary changes.
6. PAST/FAMILY/DRUG HISTORY: Ask about chronic conditions, surgeries, family history, and medications.
7. SOCIAL HISTORY: Ask about occupation, living situation, smoking, and alcohol.
8. CONCLUSION: Provide a quick recap, close the interview, and set "session_complete": true.

JSON SCHEMA (CMAS FORMAT):
Extract the data gathered in the conversation into this exact CMAS structure:
{{
  "extracted": {{
     "positive_symptoms": [{{ "value": "symptom_name", "onset": "...", "severity": "...", "duration": "..." }}],
     "negative_symptoms": [{{ "value": "symptom_name" }}],
     "patient_medical_history": ["..."],
     "family_medical_history": ["..."],
     "medications": ["..."],
     "habits": {{"smoking": "yes/no", "alcohol": "yes/no"}},
     "basic_information": {{"concerns_and_expectations": "..."}}
  }},
  "missing_attributes": ["<attr>", ...],
  "session_complete": false
}}

Note: Map any I.C.E. responses into the `basic_information` slot.
"""


def generate_node(state: GraphState) -> GraphState:
    """
    Submit the conversation context + CMAS state + RAG examples to Ollama.
    Parse the response into an updated json_state and the next question.
    Append both the user and assistant messages to the message history.
    """
    patient_input: str = state.get("patient_input", "")
    current_json_state: CMASState = state.get("json_state", {})
    rag_examples: list[str] = state.get("rag_examples", [])
    messages: list = list(state.get("messages", []))

    # Build the formatted system prompt
    json_state_str = json.dumps(current_json_state, indent=2) if current_json_state else "{}"
    examples_str = (
        "\n---\n".join(rag_examples) if rag_examples else "No examples retrieved."
    )
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        json_state=json_state_str,
        examples=examples_str,
    )

    # Assemble the full message list for the LLM call
    llm_messages: list = [SystemMessage(content=system_content)]
    # Replay existing dialogue (exclude system messages already in history)
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)
    # Add the new patient utterance
    llm_messages.append(HumanMessage(content=patient_input))

    try:
        ai_response = llm.invoke(llm_messages)
        raw_text: str = ai_response.content
    except Exception as exc:
        error_msg = (
            "I'm sorry, I'm having trouble connecting to my reasoning engine. "
            "Please try again in a moment."
        )
        new_messages = messages + [
            HumanMessage(content=patient_input),
            AIMessage(content=error_msg),
        ]
        return {
            **state,
            "messages": new_messages,
            "session_complete": False,
        }

    # ── Parse JSON block from the LLM's response ─────────────────────────────
    updated_json_state, session_complete, next_question = _parse_llm_response(
        raw_text, current_json_state
    )

    # ── Update message history ────────────────────────────────────────────────
    new_messages = messages + [
        HumanMessage(content=patient_input),
        AIMessage(content=next_question),
    ]

    return {
        **state,
        "messages": new_messages,
        "json_state": updated_json_state,
        "session_complete": session_complete,
        "last_question": next_question,
    }


def _parse_llm_response(
    raw_text: str,
    current_json_state: CMASState,
) -> tuple[CMASState, bool, str]:
    """
    Extract the JSON block and the follow-up question from the LLM output.

    The model is instructed to output JSON first, then the question. We use
    a regex to find the JSON block and treat the remainder as the next question.
    """
    # 1. Try to find a JSON object in the response
    json_match = re.search(r"\{[\s\S]*?\}", raw_text, re.DOTALL)
    updated_json_state: CMASState = dict(current_json_state)  # type: ignore[assignment]
    session_complete: bool = False

    if json_match:
        try:
            parsed = json.loads(json_match.group())
            session_complete = bool(parsed.get("session_complete", False))
            extracted = parsed.get("extracted", {})
            # Merge newly extracted slots into the running CMAS state
            for slot_type, slot_values in extracted.items():
                if slot_type not in updated_json_state:
                    updated_json_state[slot_type] = slot_values  # type: ignore[literal-required]
                elif isinstance(updated_json_state.get(slot_type), list):
                    updated_json_state[slot_type].extend(slot_values)  # type: ignore[literal-required]
                else:
                    updated_json_state[slot_type] = slot_values  # type: ignore[literal-required]
        except json.JSONDecodeError:
            pass  # Non-critical — state remains unchanged

    # 2. Extract the conversational question (text after the JSON block)
    if json_match:
        question_text = raw_text[json_match.end():].strip()
    else:
        question_text = raw_text.strip()

    # Fallback if there's nothing left after the JSON
    if not question_text:
        question_text = "Could you tell me more about how you're feeling?"

    return updated_json_state, session_complete, question_text


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — SOAP Formatter
# ─────────────────────────────────────────────────────────────────────────────

def soap_node(state: GraphState) -> GraphState:
    """
    Convert the accumulated CMAS json_state into a SOAP note.
    The formatted note is appended as a final AIMessage so the API can return it.
    """
    from app.utils.formatter import build_soap_note

    json_state: CMASState = state.get("json_state", {})
    soap_note = build_soap_note(json_state)
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=soap_note))

    return {**state, "messages": messages, "session_complete": True}
