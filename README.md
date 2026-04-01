# ANAM-AI Patient History Taking Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118+-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.ai)
[![pgvector](https://img.shields.io/badge/pgvector-0.4+-blue.svg)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

**An AI-powered patient history-taking agent tailored for the Pakistani healthcare ecosystem, designed to automate clinical SOAP note generation.**

</div>

---

## 🎯 Overview

The **ANAM-AI History Taking Agent** addresses the critical gap between high-volume patient intake and disjointed Electronic Medical Records (EMR) systems. Designed for Junior Doctors and Medical Officers managing 60–100 patients daily in high-volume OPDs, this agent dramatically reduces documentation time.

> **Language (MVP Phase):** This MVP operates in **Basic English** to validate the core agentic workflow, RAG integration, and LangGraph state machine before tackling Roman Urdu / code-switching NLP.

### 🚀 Key Features

- **🤖 Local Ollama Inference** — Runs entirely offline using a custom `qwen2.5:14b` model via Ollama. Zero cloud dependency, maximum patient data privacy.
- **🔄 LangGraph State Machine** — A cyclic `localize → context → generate` loop that drives a structured CMAS history-taking interview turn-by-turn.
- **📚 RAG via pgvector** — Ingests `MediTOD.pdf` into a local PostgreSQL + pgvector store and retrieves top-K few-shot examples per patient utterance to guide slot extraction.
- **🧩 CMAS Slot Extraction** — Each turn the LLM outputs a JSON block updating the Comprehensive Medical Attribute Schema (chief complaint, onset, severity, duration, medications, etc.).
- **📋 SOAP Note Generation** — On session completion (or manual export), the accumulated CMAS JSON is rendered into a formatted clinical SOAP note for the physician.
- **⏱️ Session Timeout + Turn Limit** — Sessions auto-expire after 60 minutes or 30 turns, preventing runaway conversations.

---

## 🏗️ System Architecture

```
Patient Utterance
       │
       ▼
  localize_node          ← strip noise / future: Urdu → English
       │
       ▼
  context_node           ← pgvector similarity → top-3 MediTOD examples
       │
       ▼
  generate_node          ← Ollama qwen2.5:14b → JSON (CMAS slots) + next question
       │
  [conditional edge]
       ├─ session_complete=True  ──► soap_node ──► SOAP Note
       ├─ turn limit reached     ──► soap_node ──► SOAP Note
       └─ still collecting       ──► localize_node (loop to next turn)
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI 0.118 + Uvicorn |
| Agent Orchestration | LangGraph 1.1 + LangChain 1.2 |
| Local LLM | Ollama (`qwen2.5:14b` via `anam-agent` Modelfile) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | PostgreSQL 16 + pgvector (Docker) |
| ORM | SQLAlchemy 2.0 |
| Language | Python 3.11+ |

---

## 📁 Repository Structure

```
ThatOneMvp-Agent/
│
├── app/                            # Entire agent lives here
│   ├── Modelfile_Qwen              # Ollama model definition (qwen2.5:14b + CMAS prompt)
│   ├── config.py                   # Centralized config (Ollama, pgvector, paths)
│   ├── main.py                     # FastAPI app & all endpoints
│   ├── requirements.txt            # Minimal pip dependencies
│   │
│   ├── agent/                      # LangGraph state machine
│   │   ├── graph.py                # Graph topology, conditional edges, compiled_graph
│   │   ├── nodes.py                # localize / context / generate / soap nodes
│   │   ├── state.py                # GraphState TypedDict + CMAS slot schemas
│   │   └── llm.py                  # ChatOllama wrapper (singleton)
│   │
│   ├── rag/                        # Retrieval-Augmented Generation
│   │   ├── ingest_meditod.py       # One-time: PDF → chunks → pgvector embeddings
│   │   └── vector_store.py         # retrieve_few_shot_examples() interface
│   │
│   ├── utils/
│   │   ├── localization.py         # Input sanitizer (pass-through for English MVP)
│   │   └── formatter.py            # CMAS JSON → SOAP note template engine
│   │
│   └── db/
│       ├── database.py             # SQLAlchemy engine, Base, create_tables(), get_db()
│       └── models.py               # ORM models: PatientSession, ConversationMessage, CollectedHistory
│
├── data/
│   └── MediTOD.pdf                 # Source document for RAG (CMAS medical ontology)
│
├── docker/
│   ├── docker-compose.yml          # PostgreSQL 16 + pgvector container
│   └── init.sql                    # Enables the vector extension on first boot
│
├── .gitignore
├── README.md
└── requirements.txt                # Full pip freeze (reference)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for the pgvector database)
- **Ollama** installed and running — [ollama.ai](https://ollama.ai)

### Step 1 — Clone & create a virtual environment

```bash
cd ThatOneMvp-Agent

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r app/requirements.txt
```

### Step 3 — Start the pgvector database

```bash
cd docker
docker compose up -d
cd ..
```

> The container uses the default credentials `anam_user / anam_password / anam_db` (configurable via `.env`).

### Step 4 — Register the Ollama model

```bash
ollama create anam-agent -f app/Modelfile_Qwen
```

Verify it's available:

```bash
ollama list
```

### Step 5 — Seed the RAG vector store

This only needs to be run **once** (or whenever `data/MediTOD.pdf` changes):

```bash
python -m app.rag.ingest_meditod
```

### Step 6 — Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## ⚙️ Environment Configuration

Create a `.env` file in the `ThatOneMvp-Agent/` root directory:

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=anam-agent

# PostgreSQL / pgvector (must match docker-compose.yml)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=anam_db
POSTGRES_USER=anam_user
POSTGRES_PASSWORD=anam_password

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000

# Override data path if needed
# MEDITOD_PDF_PATH=/absolute/path/to/MediTOD.pdf
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/session/start` | Create a new session → returns `session_id` + greeting |
| `POST` | `/session/chat` | Send patient utterance → returns agent reply + CMAS JSON |
| `GET` | `/session/history` | Retrieve full conversation log + current CMAS state |
| `GET` | `/session/export` | Conclude session → returns formatted SOAP note |
| `GET` | `/health` | Service health check |

### Example flow

```bash
# 1. Start a session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"patient_name": "Ahmed"}'

# 2. Chat (use the session_id from step 1)
curl -X POST http://localhost:8000/session/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "I have had a fever for 3 days"}'

# 3. Export SOAP note
curl "http://localhost:8000/session/export?session_id=<id>"
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Local Ollama only | Pakistan hospital networks are unreliable; zero cloud dependency |
| pgvector over Pinecone | Fully local, no API keys, same Docker container as the ORM DB |
| LangGraph state machine | Ensures structured CMAS extraction across multiple turns without losing state |
| In-memory session store (MVP) | Avoids persistence complexity for single-session demo; swap for Redis in production |
| Basic English first | Validates the entire pipeline before adding Urdu NLP complexity |

---

## 📊 Research Context

This agent is part of **ANAM-AI Phase 2** targeting Pakistani OPD settings.

- **Inconsistent Internet** → handled by routing all LLM calls to local Ollama.
- **Hallucination Risk** → physician must review the SOAP note before it enters the EMR.
- **High Volume** → stateless FastAPI + in-memory sessions handle concurrent interviews.

---

## 👥 Authors

**[22i-0767, 22i-0911, 22i-0891, 22i-0928]**
