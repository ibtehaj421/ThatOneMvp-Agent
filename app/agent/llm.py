"""
llm.py
──────
Local Ollama provider setup using the ChatOllama LangChain wrapper.
Exports a reusable `llm` instance used by the generative node.
"""

import os
import sys

from langchain_ollama import ChatOllama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.config import config


def get_llm() -> ChatOllama:
    """
    Return a ChatOllama instance pointing at the locally running Ollama server.

    The model `anam-agent` must have been created beforehand with:
        ollama create anam-agent -f Modelfile_Qwen
    """
    return ChatOllama(
        model=config.OLLAMA_MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_predict=config.MAX_TOKENS,
    )


# Module-level singleton — cheap to create, ChatOllama is stateless
llm = get_llm()
