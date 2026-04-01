# ANAM-AI Patient History Taking Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-blue.svg)](https://python.langchain.com)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

**An AI-powered patient history taking agent tailored for the Pakistani healthcare ecosystem, utilizing localized Roman Urdu natural language processing to automate clinical note generation.**

</div>

## 🎯 Overview

The **ANAM-AI History Taking Agent** addresses the critical gap between high-volume patient intake and disjointed Electronic Medical Records (EMR) systems. Designed specifically for Junior Doctors and Medical Officers managing 60–100 patients daily in high-volume OPDs, this agent drastically reduces documentation time using specialized speech-to-text models that understand Roman Urdu and medical code-switching.

### 🚀 Key Features

- **🗣️ Roman Urdu NLP** - Fine-tuned to understand Roman Urdu medical code-switching (e.g., "pait mein dard" alongside "abdominal pain").
- **⚡ AI Risk Prediction & Extraction** - Uses LLMs (via Groq and LangChain) to parse speech-to-text transcripts and structure clinical notes.
- **🛡️ Privacy & Verification Logic** - Implements a mandatory "Verify" step to prevent AI hallucinations and clinical errors before data is saved.
- **📚 Centralized Knowledge Base** - Uses `pgvector` and `Supabase` to retain specific medical guidelines and local disease prevalence contexts.

### 🏗️ System Architecture

Our system uses a modular AI agent pattern:

- **Ingestion Module** (`ingest.py`): Processes and chunks incoming text or parsed medical transcripts.
- **Agent Core** (`agent.py` & `main.py`): Runs the orchestration logic and serves as the FastAPI backend entry point.
- **Vector Store** (`vector-store.py` & `database.py`): Handles storing and querying embeddings using PostgreSQL.
- **Model Interface** (`model.py` & `services.py`): Connects to the Groq API for rapid inference and LangChain processing.

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+** - Primary development language
- **FastAPI** - High-performance web framework for the agent API
- **LangChain & LangGraph** - Framework for developing the LLM application architecture
- **Groq API** - Ultra-fast LLM inference
- **Sentence Transformers** - For generating semantic embeddings
- **Supabase / pgvector** - Database and vector similarity search

## 📁 Repository Structure

```
.
├─ agent.py             # Agent execution logic
├─ database.py          # Database connection and setup
├─ ingest.py            # Document ingestion for vector store
├─ main.py              # FastAPI application entry point
├─ model.py             # LLM model configuration
├─ requirements.txt     # Python dependencies
├─ schemas.py           # Pydantic schemas for data validation
├─ services.py          # Business logic and external API integrations
└─ vector-store.py      # Vector database operations
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Supabase Account** (or local PostgreSQL with pgvector)
- **Groq API Key**

### Installation

1. **Clone the repository and enter the directory**
   ```bash
   cd ThatOneMvp-Agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
DATABASE_URL=your_postgres_connection_string
```

### Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API documentation will be available at `http://localhost:8000/docs`.

## 📊 Research Context

This agent is developed as part of **ANAM-AI (Phase 2 & 3)**. The application's goal is to transition healthcare out of a paper-based, fragmented system into a digital, proactive system.

**Key constraints managed:**
- **Inconsistent Internet:** Focus on eventual offline-sync integrations with the Go backend.
- **High-Risk Hallucinations:** Eliminated by requiring physician verification.

## 👥 Authors

- **[22i-0767, 22i-0911, 22i-0891, 22i-0928]** 
