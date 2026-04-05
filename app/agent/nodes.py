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

# CMAS slot names (must match CMASState). Models often emit these at top level instead of under "extracted".
_CMAS_SLOT_KEYS = frozenset(
    {
        "chief_complaint",
        "positive_symptoms",
        "negative_symptoms",
        "patient_medical_history",
        "family_medical_history",
        "habits",
        "medications",
        "medical_tests",
        "exposures",
        "basic_information",
    }
)
_JSON_META_KEYS = frozenset({"session_complete", "missing_attributes", "extracted"})
from app.agent.llm import llm
from app.config import config

# After a short rhetorical "? ", real questions usually start with these words.
_FOLLOWUP_QUESTION_START = re.compile(
    r"^(When|What|Where|Who|Why|How|Can|Do|Does|Did|Have|Has|Is|Are|Could|Would|Should)\b",
    re.I,
)


def _last_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content or "")
    return ""


def _missing_sections_status(json_state: CMASState) -> str:
    """Short checklist for the model: what is still empty in CMAS (one topic per turn)."""
    missing: list[str] = []
    if not (json_state.get("medications") or []):
        missing.append("current medications (or confirm none / unsure)")
    if not (json_state.get("family_medical_history") or []):
        missing.append("family medical history (or none known)")
    if not (json_state.get("patient_medical_history") or []):
        missing.append("past medical history")
    habits = json_state.get("habits") or {}
    if not habits.get("smoking") or not habits.get("alcohol"):
        missing.append("smoking/alcohol (habits)")
    basic = json_state.get("basic_information") or {}
    if not basic.get("concerns_and_expectations"):
        missing.append("patient concerns/expectations (I.C.E.)")
    if not missing:
        return "Core sections are largely filled; if the patient has nothing to add after your recap, conclude."
    return "Next missing topics (pick ONE question for the highest-priority item only): " + "; ".join(missing)


def _user_requests_closure(user_text: str) -> bool:
    t = user_text.strip().lower()
    if not t:
        return False
    needles = (
        "give me report",
        "give me the report",
        "clinical report",
        "soap",
        "end it",
        "end now",
        "end the interview",
        "end the session",
        "finish the interview",
        "stop the interview",
        "i'm done",
        "im done",
        "that's all",
        "thats all",
        "nothing else to add",
        "nothing more to add",
    )
    return any(n in t for n in needles)


def _user_declines_after_recap(last_ai: str, user_text: str) -> bool:
    """True when assistant asked 'anything else?' and the patient signals they are finished."""
    if not last_ai:
        return False
    last = last_ai.lower()
    if "anything else" not in last and "add anything" not in last:
        return False
    u = user_text.strip().lower().rstrip(".!?")
    if u in ("no", "nope", "nothing", "none", "no thanks", "no thank you", "nah"):
        return True
    # Short negatives like "no i dont" after recap; avoid long answers ("no i dont drink...")
    if u.startswith("no") and len(u) <= 18:
        return True
    return False


def _should_force_session_complete(user_text: str, messages: list) -> bool:
    return _user_requests_closure(user_text) or _user_declines_after_recap(
        _last_ai_text(messages), user_text
    )


def _enforce_single_question_reply(text: str) -> str:
    """
    Post-process the conversational reply so the patient sees at most one question.
    Keeps leading empathy paragraphs that do not contain a question mark.
    """
    text = (text or "").strip()
    if not text or text.count("?") <= 1:
        return text

    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts:
        return text

    idx = next((i for i, p in enumerate(parts) if "?" in p), None)
    if idx is None:
        return text

    head = "\n\n".join(parts[:idx]).strip()
    last_para = parts[idx]

    fq = last_para.find("?")
    if fq < 0:
        return text

    after_first_q = last_para[fq + 1 :].lstrip()
    # Keep "Really? When did it start?" as one turn; do not merge "Rate 1-10? Any nausea?".
    short_rhetorical_lead = fq <= 8
    if (
        last_para.count("?") >= 2
        and short_rhetorical_lead
        and _FOLLOWUP_QUESTION_START.match(after_first_q)
    ):
        sq = last_para.find("?", fq + 1)
        trimmed = last_para[: sq + 1].strip() if sq != -1 else last_para[: fq + 1].strip()
    else:
        trimmed = last_para[: fq + 1].strip()

    if head:
        return f"{head}\n\n{trimmed}".strip()
    return trimmed


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
You are ANAM, a warm, professional pre-visit history-taking assistant in a Pakistani hospital setting.
You collect information so a human clinician can take over — you are NOT the treating clinician in the room.

CURRENT CMAS SLOT STATE (JSON):
{json_state}

SECTIONS STILL TO COVER (internal guide — do not read this list aloud verbatim):
{missing_sections_status}

FEW-SHOT MEDITOD EXAMPLES (tone and pacing only — do not copy unsafe clinical behavior):
{examples}

CORE RULES:
1. EXACTLY ONE QUESTION PER TURN: The conversational text after the JSON must contain only ONE question mark and ONE history question. Do not add follow-ups with "Also," "Additionally," "And," or a second sentence that asks something else. No numbered lists (1. 2.) or bullet lists of questions.
2. INTAKE ONLY: Do NOT perform or role-play physical exams, vitals, imaging, or procedures. Do NOT say you are checking blood pressure, doing an eye exam, ordering MRI/CT, or narrating results/placeholders like "[insert reading]".
3. NO PLANNING AS CLINICIAN: Do NOT recommend specific tests, treatments, or workflows as if you will carry them out. If asked whether tests hurt or what will happen, say briefly that their doctor will explain and choose what is appropriate.
4. EMPATHY: At most one short empathy sentence without a question, then your single question — or only the question if empathy is redundant.
5. DO NOT DIAGNOSE OR PRESCRIBE.
6. EMERGENCY: If the patient reports chest pain, severe bleeding, trouble breathing, or unconsciousness, tell them to seek emergency care immediately and set "session_complete": true.
7. JSON EVERY TURN (required): Every assistant message MUST contain one valid JSON object (with "extracted" and "session_complete") or the history will not save. Put JSON first, then your single conversational follow-up (one question only).

INTERVIEW FLOW (natural order; skip sections already well captured in JSON):
1. Chief complaint and symptom details (SOCRATES).
2. Relevant negatives, ICE (ideas/concerns/expectations), systems review as needed.
3. Past medical history, family history, medications, habits/social context.
4. Brief recap, then ask if they want to add anything else.

CONCLUSION (CRITICAL):
- If your recap asks whether they want to add anything and they clearly say no / nothing else / that's all (or they ask for the report / to end), set "session_complete": true.
- When "session_complete": true, reply with ONE short thank-you line only (no new questions, no exam, no test discussion). The system will attach the structured note separately.

JSON SCHEMA (CMAS FORMAT):
Extract the data gathered in the conversation into this exact CMAS structure:
{{
  "extracted": {{
     "chief_complaint": "...",
     "positive_symptoms": [{{ "value": "symptom_name", "onset": "...", "severity": "...", "duration": "...", "location": "...", "progression": "...", "frequency": "..." }}],
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

Note: Map I.C.E. into `basic_information.concerns_and_expectations`. Use negative_symptoms for denied associated symptoms.
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
    missing_status = _missing_sections_status(current_json_state)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        json_state=json_state_str,
        examples=examples_str,
        missing_sections_status=missing_status,
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
    except Exception:
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

    chief_ok = bool(updated_json_state.get("chief_complaint"))
    sym_ok = len(updated_json_state.get("positive_symptoms") or []) > 0
    if session_complete and not (chief_ok and sym_ok):
        session_complete = False

    if _should_force_session_complete(patient_input, messages) and chief_ok and sym_ok:
        session_complete = True
        next_question = (
            "Thank you — your history has been recorded for your doctor to review."
        )
    else:
        next_question = _enforce_single_question_reply(next_question)

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


def _normalize_symptom_list(items: list) -> list[dict]:
    """LLMs sometimes emit ['fever', 'headache'] instead of [{value: ...}, ...]."""
    out: list[dict] = []
    for sym in items:
        if isinstance(sym, str) and sym.strip():
            out.append({"value": sym.strip()})
        elif isinstance(sym, dict):
            out.append(sym)
    return out


def _merge_cmas_extracted(
    updated_json_state: CMASState,
    extracted: dict,
) -> None:
    """Merge one batch of slot updates (from nested "extracted" or top-level JSON) into state."""
    for slot_type, new_values in extracted.items():
        if slot_type in _JSON_META_KEYS:
            continue
        if slot_type in ("positive_symptoms", "negative_symptoms") and isinstance(new_values, list):
            new_values = _normalize_symptom_list(new_values)
        if slot_type not in updated_json_state:
            updated_json_state[slot_type] = new_values  # type: ignore[literal-required]
            continue

        current_val = updated_json_state.get(slot_type)

        if slot_type in ("positive_symptoms", "negative_symptoms") and isinstance(new_values, list):
            existing_list = (
                _normalize_symptom_list(list(current_val))
                if isinstance(current_val, list)
                else []
            )
            symptom_map = {s.get("value", "").lower(): s for s in existing_list if s.get("value")}

            for sym in new_values:
                if not isinstance(sym, dict):
                    continue
                name = sym.get("value", "").lower()
                if not name:
                    continue

                if name in symptom_map:
                    symptom_map[name].update({k: v for k, v in sym.items() if v})
                else:
                    symptom_map[name] = sym

            updated_json_state[slot_type] = list(symptom_map.values())  # type: ignore[literal-required]

        elif isinstance(current_val, list) and isinstance(new_values, list):
            for val in new_values:
                if val not in current_val:
                    current_val.append(val)
            updated_json_state[slot_type] = current_val  # type: ignore[literal-required]

        elif isinstance(current_val, dict) and isinstance(new_values, dict):
            current_val.update(new_values)
            updated_json_state[slot_type] = current_val  # type: ignore[literal-required]

        else:
            updated_json_state[slot_type] = new_values  # type: ignore[literal-required]


def _normalize_llm_text_for_json(raw_text: str) -> str:
    """Normalize quotes and strip common markdown fences so JSONDecoder can parse."""
    t = (raw_text or "").replace("\r\n", "\n")
    t = (
        t.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    if "```" in t:
        lines = [ln for ln in t.split("\n") if not ln.strip().startswith("```")]
        t = "\n".join(lines)
    return t


def _lenient_json_tail(s: str) -> str:
    """Remove trailing commas before } or ] — common in LLM JSON."""
    return re.sub(r",(\s*[\]}])", r"\1", s)


def _score_cmas_dict(obj: dict) -> int:
    """Prefer objects that look like our CMAS payload over incidental dicts."""
    score = 0
    ext = obj.get("extracted")
    if isinstance(ext, dict):
        score += 20 + 2 * len(ext)
    for k in _CMAS_SLOT_KEYS:
        if k in obj:
            score += 8
    if "session_complete" in obj:
        score += 1
    score += min(len(json.dumps(obj)), 400) // 40
    return score


def _extract_json_dict_candidates(
    base: str,
) -> list[tuple[int, int, dict, str]]:
    """
    Find every JSON object via JSONDecoder.raw_decode (string-safe).
    Tries a trailing-comma–stripped copy as well; indices are always relative to `variant`.
    """
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict, str]] = []
    variants = [base]
    lenient = _lenient_json_tail(base)
    if lenient != base:
        variants.append(lenient)
    for variant in variants:
        n = len(variant)
        for i in range(n):
            if variant[i] != "{":
                continue
            try:
                obj, end = decoder.raw_decode(variant, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and end > i:
                candidates.append((i, end, obj, variant))
    return candidates


def _pick_best_cmas_candidate(
    candidates: list[tuple[int, int, dict, str]],
) -> tuple[int, int, dict, str] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda x: _score_cmas_dict(x[2]))


def _parse_llm_response(
    raw_text: str,
    current_json_state: CMASState,
) -> tuple[CMASState, bool, str]:
    """
    Extract CMAS JSON from the model reply (any position). Uses JSONDecoder.raw_decode
    so braces inside JSON strings do not break parsing. Picks the highest-scoring dict
    when multiple objects appear (e.g. nested noise).
    """
    updated_json_state: CMASState = dict(current_json_state)  # type: ignore[assignment]
    session_complete: bool = False
    question_text = ""

    normalized = _normalize_llm_text_for_json(raw_text)
    candidates = _extract_json_dict_candidates(normalized)
    best = _pick_best_cmas_candidate(candidates)

    if best is not None:
        start_idx, end_idx, parsed, src = best
        try:
            session_complete = bool(parsed.get("session_complete", False))
            nested = parsed.get("extracted", {})
            if not isinstance(nested, dict):
                nested = {}
            _merge_cmas_extracted(updated_json_state, nested)
            top_slots = {k: v for k, v in parsed.items() if k in _CMAS_SLOT_KEYS}
            _merge_cmas_extracted(updated_json_state, top_slots)
        except Exception as e:
            print(f"[ERROR] ❌ Unexpected error during CMAS merge: {str(e)}")
        question_text = (src[:start_idx] + src[end_idx:]).strip()
    else:
        question_text = normalized.strip()
        if raw_text and len(normalized.strip()) > 0:
            print(
                f"[WARN] ⚠ No JSON dict parsed from LLM reply (len={len(normalized)}). "
                f"Head: {normalized[:180]!r}..."
            )

    if not question_text:
        question_text = (
            "Thanks for that. Is there anything else you'd like to add about this right now?"
            if best is not None
            else "Could you tell me more about how you're feeling?"
        )

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
