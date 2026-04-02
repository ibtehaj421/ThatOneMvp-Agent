"""
vector_store.py
───────────────
Thin interface around pgvector for retrieving MediTOD few-shot examples.
Called by the LangGraph context node at each turn.
"""

import os
import sys
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import config


class MediTODVectorStore:
    """Lazy-initialised wrapper around the PGVector-backed MediTOD collection."""

    def __init__(self) -> None:
        self._store: Optional[PGVector] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def _get_store(self) -> PGVector:
        """Lazily connect to pgvector on first use."""
        if self._store is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
            self._store = PGVector(
                embeddings=self._embeddings,
                collection_name=config.PGVECTOR_COLLECTION,
                connection=config.postgres_url,
                use_jsonb=True,
            )
        return self._store

    def retrieve_few_shot_examples(
        self,
        patient_utterance: str,
        top_k: int | None = None,
    ) -> list[str]:
        """
        Perform a similarity search against MediTOD chunks.

        Returns a list of the top-K document page_content strings that the
        LangGraph generative node will inject as few-shot examples.
        """
        k = top_k or config.RAG_TOP_K
        try:
            store = self._get_store()
            results = store.similarity_search(patient_utterance, k=k)
            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                symptoms = meta.get("symptoms", [])
                slot_types = meta.get("slot_types", [])
                dialog_id = meta.get("dialog_id", "?")
                urange = meta.get("utterance_range", "?")
                print(
                    f"   [RAG chunk {i}] dialog={dialog_id} "
                    f"turns={urange} symptoms={symptoms} slots={slot_types}"
                )
            return [doc.page_content for doc in results]
        except Exception as exc:
            # Gracefully degrade — return an empty list so the agent can still
            # function without RAG context if the DB is unavailable.
            print(f"[VectorStore] WARNING: similarity search failed — {exc}")
            return []


# Module-level singleton — imported by nodes.py
vector_store = MediTODVectorStore()
