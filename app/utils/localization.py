"""
localization.py
───────────────
Localization pipeline for ANAM-AI.

MVP Phase (Basic English):
    Acts as a pass-through — normalizes whitespace and strips control
    characters only. No translation is performed.

Future Sprint (Roman Urdu / Code-switching):
    Replace `localize_input` with a proper transliteration pipeline that:
      1. Detects the script (Latin / Nastaliq / mixed).
      2. Transliterates Roman Urdu → Urdu Unicode (via UrduHack or AI4Bharat).
      3. Translates Urdu → English (via a local Ollama call or a fine-tuned
         m2m100 model).
    The outbound counterpart `localize_output` would reverse the pipeline.
"""

import re


def localize_input(raw_text: str) -> str:
    """
    Sanitize and normalize raw patient input for English MVP.

    Steps
    -----
    1. Strip leading/trailing whitespace.
    2. Collapse interior runs of whitespace/newlines to single spaces.
    3. Remove non-printable control characters (keeps Unicode letters).

    Returns
    -------
    str
        Cleaned English text ready for the generate node.
    """
    if not raw_text:
        return ""
    # Remove non-printable control characters (but keep newlines for now)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    # Collapse multiple newlines
    text = re.sub(r"\n{2,}", " ", text)
    return text


def localize_output(ai_text: str) -> str:
    """
    Post-process the AI-generated question before returning it to the patient.

    MVP Phase: no-op — returns text unchanged.
    Future Sprint: translate English → patient's preferred language.
    """
    return ai_text
