"""Application settings loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
HISTORY_DIR = BASE_DIR / "chat_history"
ASSETS_DIR = BASE_DIR / "assets"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
DISCLAIMER = (
    "**Medical disclaimer:** This educational tool is not a substitute for a "
    "qualified healthcare professional. It does not diagnose conditions, "
    "prescribe medicines, or recommend dosages. Seek urgent medical care for "
    "emergencies."
)


def load_environment() -> str:
    """Load .env and return a validated Gemini API key."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return api_key


def is_demo_mode_requested() -> bool:
    """Return True if the user explicitly asked to skip Gemini and use demo answers."""
    load_dotenv(BASE_DIR / ".env")
    return os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}
