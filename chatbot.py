"""Grounded Gemini response service."""
from __future__ import annotations

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from config import DISCLAIMER, GEMINI_MODEL, is_demo_mode_requested, load_environment
from rag import NO_ANSWER, MedicalRetriever, RetrievedChunk, build_context, build_prompt

LOGGER = logging.getLogger(__name__)


class MedicalChatbot:
    """Retrieves trusted content before every Gemini generation.

    If no valid GOOGLE_API_KEY is configured (or DEMO_MODE=true is set in .env),
    the bot automatically runs in demo mode: it still performs real FAISS
    retrieval over the ingested PDFs, but instead of calling Gemini it returns
    the top retrieved passage directly. This is handy for classroom demos or
    offline runs where you don't want to use a real API key yet.
    """

    def __init__(self) -> None:
        self.retriever = MedicalRetriever()
        self.demo_mode = False
        api_key = None
        try:
            if is_demo_mode_requested():
                raise ValueError("DEMO_MODE=true was set in .env")
            api_key = load_environment()
        except ValueError as exc:
            self.demo_mode = True
            LOGGER.warning("Starting in demo mode (no live Gemini calls): %s", exc)
        if not self.demo_mode:
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL, google_api_key=api_key, temperature=0,
            )

    def answer(self, question: str, history: list[dict[str, str]]) -> tuple[str, list[RetrievedChunk]]:
        """Retrieve evidence then generate a constrained educational response."""
        chunks = self.retriever.search(question)
        if self.demo_mode:
            return self._demo_answer(chunks), chunks
        context = build_context(chunks)
        conversation = "\n".join(
            f"{message['role'].title()}: {message['content']}" for message in history[-6:]
        )
        try:
            response = self.llm.invoke(build_prompt(question, context, conversation))
            answer = str(response.content).strip()
            # A deterministic safeguard in addition to the prompt instruction.
            if "Medical disclaimer:" not in answer:
                answer = f"{answer}\n\n{DISCLAIMER}"
            return answer, chunks
        except Exception as exc:
            LOGGER.exception("Gemini request failed")
            raise RuntimeError("The response service is unavailable. Please try again shortly.") from exc

    def _demo_answer(self, chunks: list[RetrievedChunk]) -> str:
        """Build an answer straight from the retrieved evidence, no LLM call."""
        if not chunks:
            return f"{NO_ANSWER}\n\n{DISCLAIMER}"
        top = chunks[0]
        metadata = top.document.metadata
        source = metadata.get("source_name") or metadata.get("source", "the knowledge base")
        page = int(metadata.get("page", 0)) + 1
        snippet = top.document.page_content.strip().replace("\n", " ")
        if len(snippet) > 700:
            snippet = snippet[:700].rsplit(" ", 1)[0] + "…"
        return (
            f"**{source}, page {page}:**\n\n"
            f"{snippet}\n\n"
            f"{DISCLAIMER}"
        )
