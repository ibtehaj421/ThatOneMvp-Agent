import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Ollama (local LLM) ────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "anam-agent")

    # ── PostgreSQL + pgvector (Docker) ─────────────────────────────────────────
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5435"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "anam_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "anam_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "anam_password")

    # ── RAG / pgvector ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    PGVECTOR_COLLECTION: str = "meditod_examples"
    RAG_TOP_K: int = 3

    # ── Chunking: sliding window over dialogue utterances ──────────────────────
    # Number of utterances per chunk (e.g. 6 = ~3 doctor + 3 patient turns)
    RAG_WINDOW_SIZE: int = int(os.getenv("RAG_WINDOW_SIZE", "6"))
    # How many utterances to advance per step — overlap = WINDOW_SIZE - WINDOW_STEP
    RAG_WINDOW_STEP: int = int(os.getenv("RAG_WINDOW_STEP", "4"))

    # ── LLM Generation ─────────────────────────────────────────────────────────
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.3

    # ── FastAPI ────────────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── Session ────────────────────────────────────────────────────────────────
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_CONVERSATION_TURNS: int = 30

    # ── SOAP / Completion thresholds ───────────────────────────────────────────
    # Minimum CMAS slots that must be filled before the agent can conclude
    REQUIRED_SLOTS: list = field(default_factory=lambda: [
        "chief_complaint", "onset", "severity", "duration"
    ])

    # ── MediTOD document path (for ingest) ─────────────────────────────────────
    # Resolved relative to this file: app/ → ../ → ThatOneMvp-Agent/ → data/dialogs.json
    MEDITOD_JSON_PATH: str = os.getenv(
        "MEDITOD_JSON_PATH",
        str(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "dialogs.json")))
    )

    # Set to a positive integer to limit ingested dialog examples, or None / 0 for no limit
    MEDITOD_INGEST_LIMIT: Optional[int] = int(os.getenv("MEDITOD_INGEST_LIMIT", "0")) or None

    # ── Ollama Modelfile path ──────────────────────────────────────────────────
    # Used in setup docs / CI scripts:  ollama create anam-agent -f <this path>
    MODELFILE_PATH: str = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "Modelfile_Qwen")
    )

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def validate(self) -> bool:
        """Validate that essential config values are present."""
        if not self.POSTGRES_PASSWORD:
            raise ValueError("POSTGRES_PASSWORD must be set.")
        return True


# Singleton
config = Config()
