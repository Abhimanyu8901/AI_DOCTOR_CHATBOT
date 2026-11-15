from __future__ import annotations
import logging
from langchain_groq import ChatGroq
from config import (DISCLAIMER, GROQ_MODEL, is_demo_mode_requested, load_environment)
from rag import NO_ANSWER, MedicalRetriever, RetrievedChunk, build_context, build_prompt

LOGGER = logging.getLogger(__name__)


class MedicalChatbot:

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
            self.llm = ChatGroq(
                model=GROQ_MODEL, groq_api_key=api_key, temperature=0, max_retries=2,
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
