"""
ingest_meditod.py
─────────────────
One-time script to parse MediTOD.pdf and store its state-transition chunks
in the pgvector table used for few-shot RAG examples.

Run once:
    python -m app.rag.ingest_meditod
"""

import os
import sys
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# ── Resolve the app package root regardless of cwd ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import config


def _clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_chunk_meditod() -> list[Document]:
    """Load MediTOD.pdf and split into dialogue-turn-sized chunks."""
    pdf_path = os.path.abspath(config.MEDITOD_PDF_PATH)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"MediTOD.pdf not found at: {pdf_path}\n"
            "Place MediTOD.pdf in the ThatOneMvp-Agent/ root folder."
        )

    print(f"📄 Loading MediTOD from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()
    print(f"   Loaded {len(raw_pages)} pages.")

    # Clean text per page
    for doc in raw_pages:
        doc.page_content = _clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,          # ~300 tokens — tight enough for a single turn example
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)
    print(f"   Created {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks: list[Document]) -> PGVector:
    """Embed chunks and upsert into the pgvector collection."""
    print(f"🔤 Loading embedding model: {config.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print(f"🐘 Connecting to pgvector at: {config.postgres_url}")
    store = PGVector(
        embeddings=embeddings,
        collection_name=config.PGVECTOR_COLLECTION,
        connection=config.postgres_url,
        use_jsonb=True,
    )

    print(f"🚀 Ingesting {len(chunks)} chunks into collection '{config.PGVECTOR_COLLECTION}' …")
    # PGVector.from_documents re-creates the collection if it already exists
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
