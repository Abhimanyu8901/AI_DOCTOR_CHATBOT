"""Reusable UI, serialisation, and text helpers."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_history(messages: list[dict[str, Any]], directory: Path) -> Path:
    """Persist chat messages as JSON and return the saved file path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"medical_chat_{timestamp()}.json"
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def history_download(messages: list[dict[str, Any]]) -> str:
    """Format messages as a readable Markdown transcript."""
    lines = ["# AI Doctor Chatbot conversation", ""]
    for message in messages:
        role = "You" if message["role"] == "user" else "AI Doctor"
        lines.extend([f"## {role}", message["content"], ""])
    return "\n".join(lines)


def clamp_confidence(relevance_score: float) -> int:
    """Convert a FAISS relevance score to a display percentage."""
    return max(0, min(100, round(relevance_score * 100)))
