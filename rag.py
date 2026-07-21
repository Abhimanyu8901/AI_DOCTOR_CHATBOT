"""Retrieval and prompt-building services for the medical chatbot."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR

NO_ANSWER = "I couldn't find reliable information about that in my medical knowledge base."


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved document segment with its relevance score."""
    document: Document
    relevance: float


class MedicalRetriever:
    """Loads a local FAISS store and performs similarity retrieval."""

    def __init__(self) -> None:
        if not (VECTORSTORE_DIR / "index.faiss").exists():
            raise FileNotFoundError("Knowledge base not found. Run `python ingest.py` first.")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = FAISS.load_local(
            str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )

    def search(self, question: str, k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the top relevant chunks; scores are normalised to 0..1."""
        results = self.store.similarity_search_with_relevance_scores(question, k=k)
        return [RetrievedChunk(document=doc, relevance=float(score)) for doc, score in results]


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Create a clearly sourced context block for the LLM."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.document.metadata
        source = metadata.get("source_name") or metadata.get("source", "Unknown source")
        page = metadata.get("page", 0)
        blocks.append(f"[Source {index}: {source}, page {page + 1}]\n{chunk.document.page_content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str, conversation: str = "") -> str:
    """Construct a strict grounded-answer prompt."""
    return f"""You are an AI medical education assistant. Answer ONLY using the provided medical context.
Never use outside knowledge, invent facts, diagnose, prescribe medicine, or recommend dosages.
If the context does not directly support a reliable answer, reply exactly: {NO_ANSWER}
Use simple language. Organize a supported answer with short headings and bullets when helpful.
Always include this exact final disclaimer: Medical disclaimer: Educational information only; consult a qualified healthcare professional for diagnosis, treatment, or emergencies.

Conversation (only for resolving references; it is not medical evidence):
{conversation or 'None'}

Medical context:
{context}

User question: {question}
"""
