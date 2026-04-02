"""
ingest_meditod.py
─────────────────
One-time script to parse dialogs.json and store its conversational chunks
in the pgvector table used for few-shot RAG examples.

Chunking strategy: Hybrid Turn-based Sliding Window + Metadata Enrichment
  - Slides a window of RAG_WINDOW_SIZE utterances over each dialogue,
    advancing RAG_WINDOW_STEP utterances per step (overlap = SIZE - STEP).
  - Each chunk is prefixed with a metadata header extracted from the JSON
    annotations (keywords → symptoms, dialog_state → slot types, nlu → intents).
  - This ensures no Doctor question is orphaned from its Patient answer, and
    that the embedding model indexes medical context alongside the raw dialogue.

Run once:
    python -m app.rag.ingest_meditod
"""

import os
import sys
import re
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# ── Resolve the app package root regardless of cwd ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving sentence flow."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_window_metadata(window: list[dict]) -> dict:
    """
    Scan a list of utterance dicts and extract:
      - symptoms   : union of all 'keywords' values from every utterance
      - slot_types : slot categories from patient 'dialog_state' fields
      - intents    : NLU intent strings from patient 'nlu' fields
    """
    symptoms: set[str] = set()
    slot_types: set[str] = set()
    intents: set[str] = set()

    for u in window:
        # Keywords (both speaker types may have these)
        for kw in u.get("keywords", []):
            if kw:
                symptoms.add(kw.lower())

        # dialog_state and nlu only appear on patient utterances
        if u.get("speaker") == "patient":
            for slot_key in u.get("dialog_state", {}).keys():
                slot_types.add(slot_key)

            for nlu_entry in u.get("nlu", []):
                intent = nlu_entry.get("intent", "")
                if intent:
                    intents.add(intent)

    return {
        "symptoms": sorted(symptoms),
        "slot_types": sorted(slot_types),
        "intents": sorted(intents),
    }


def _build_metadata_prefix(meta: dict) -> str:
    """
    Render the metadata dict as a compact one-line header that is prepended
    to the chunk text so the embedding model indexes it alongside the dialogue.

    Example:
        [Symptoms: coughing, dyspnea | Slots: positive_symptom | Intents: inform]
    """
    parts = []
    if meta["symptoms"]:
        parts.append(f"Symptoms: {', '.join(meta['symptoms'])}")
    if meta["slot_types"]:
        parts.append(f"Slots: {', '.join(meta['slot_types'])}")
    if meta["intents"]:
        parts.append(f"Intents: {', '.join(meta['intents'])}")

    return f"[{' | '.join(parts)}]" if parts else ""


def _window_chunks(
    utterances: list[dict],
    window_size: int,
    window_step: int,
    dialog_id: str,
) -> list[Document]:
    """
    Slide a window over the utterances list and produce one Document per window.

    Each Document:
      - page_content : metadata prefix header + formatted dialogue turns
      - metadata     : structured dict with dialog_id, utterance_range,
                       symptoms, slot_types, intents
    """
    docs: list[Document] = []
    n = len(utterances)
    start = 0

    while start < n:
        end = min(start + window_size, n)
        window = utterances[start:end]

        # ── Extract metadata for this window ──────────────────────────────────
        meta = _extract_window_metadata(window)
        prefix = _build_metadata_prefix(meta)

        # ── Format dialogue turns ─────────────────────────────────────────────
        turn_lines = []
        for u in window:
            speaker = str(u.get("speaker", "unknown")).capitalize()
            text = _clean_text(u.get("text", ""))
            if text:
                turn_lines.append(f"{speaker}: {text}")

        if not turn_lines:
            start += window_step
            continue

        dialogue_block = "\n".join(turn_lines)
        page_content = f"{prefix}\n{dialogue_block}" if prefix else dialogue_block

        docs.append(Document(
            page_content=page_content,
            metadata={
                "dialog_id": dialog_id,
                "utterance_range": f"{start}-{end - 1}",
                "symptoms": meta["symptoms"],
                "slot_types": meta["slot_types"],
                "intents": meta["intents"],
            },
        ))

        # Advance the window — stop if we've already reached the end
        if end == n:
            break
        start += window_step

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_meditod() -> list[Document]:
    """Load dialogs.json and produce sliding-window + metadata-enriched chunks."""
    json_path = os.path.abspath(config.MEDITOD_JSON_PATH)
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"dialogs.json not found at: {json_path}\n"
            "Place dialogs.json in the ThatOneMvp-Agent/data/ folder."
        )

    print(f"📄 Loading dialogs from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    window_size = config.RAG_WINDOW_SIZE
    window_step = config.RAG_WINDOW_STEP
    limit = config.MEDITOD_INGEST_LIMIT

    print(f"⚙️  Chunking strategy: window_size={window_size}, window_step={window_step} "
          f"(overlap={window_size - window_step} utterances)")

    all_chunks: list[Document] = []
    dialog_count = 0

    for dialog_id, dialog_data in data.items():
        if limit is not None and dialog_count >= limit:
            print(f"🛑 Reached ingestion limit of {limit} dialogs.")
            break

        utterances = dialog_data.get("utterances", [])
        if not utterances:
            continue

        chunks = _window_chunks(utterances, window_size, window_step, dialog_id)
        all_chunks.extend(chunks)
        dialog_count += 1

    print(f"   Processed {dialog_count} dialogues → {len(all_chunks)} chunks total.")
    return all_chunks


def build_vector_store(chunks: list[Document]) -> PGVector:
    """Embed chunks and upsert into the pgvector collection."""
    print(f"🔤 Loading embedding model: {config.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print(f"🐘 Connecting to pgvector at: {config.postgres_url}")
    print(f"🚀 Ingesting {len(chunks)} chunks into collection '{config.PGVECTOR_COLLECTION}' …")

    store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.PGVECTOR_COLLECTION,
        connection=config.postgres_url,
        use_jsonb=True,
        pre_delete_collection=True,   # idempotent re-ingest
    )
    print("✅ Ingestion complete.")
    return store


def main() -> None:
    chunks = load_and_chunk_meditod()
    build_vector_store(chunks)


if __name__ == "__main__":
    main()
