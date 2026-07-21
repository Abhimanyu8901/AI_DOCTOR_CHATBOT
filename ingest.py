"""Build a local FAISS index from PDFs placed in the data directory."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL,
                    VECTORSTORE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def load_pdfs(data_dir: Path) -> list:
    """Load every readable PDF under *data_dir*, adding source metadata."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Add trusted medical PDFs first.")
    documents = []
    for pdf_path in pdfs:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            for page in pages:
                page.metadata["source_name"] = pdf_path.name
            documents.extend(pages)
            LOGGER.info("Loaded %s (%d pages)", pdf_path.name, len(pages))
        except Exception as exc:
            LOGGER.warning("Skipping unreadable PDF %s: %s", pdf_path.name, exc)
    if not documents:
        raise RuntimeError("No readable content could be extracted from the PDFs.")
    return documents


def build_vectorstore() -> int:
    """Create and save FAISS index; return the number of text chunks."""
    DATA_DIR.mkdir(exist_ok=True)
    documents = load_pdfs(DATA_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    LOGGER.info("Saved %d chunks to %s", len(chunks), VECTORSTORE_DIR)
    return len(chunks)


if __name__ == "__main__":
    try:
        count = build_vectorstore()
        print(f"Success: indexed {count} chunks in {VECTORSTORE_DIR}")
    except Exception as error:
        LOGGER.error("Ingestion failed: %s", error)
        raise SystemExit(1) from error
