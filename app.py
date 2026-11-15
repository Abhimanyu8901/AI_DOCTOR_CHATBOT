from __future__ import annotations
import logging
import streamlit as st
from chatbot import MedicalChatbot
from config import DISCLAIMER, HISTORY_DIR, VECTORSTORE_DIR
from ingest import build_vectorstore
from utils.helper import clamp_confidence, history_download, write_history
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

FAQS = {
    "What can this tool do?": "It explains information found in the trusted PDFs in its knowledge base.",
    "Can it diagnose me?": "No. It cannot diagnose conditions or replace a clinician.",
    "What should I do in an emergency?": "Contact local emergency services or seek urgent in-person medical care now.",
}
def initialise_state() -> None:
    """Set Streamlit session defaults."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("theme", "Light")
    st.session_state.setdefault("bot", None)

@st.cache_resource(show_spinner=False)
def create_bot() -> MedicalChatbot:
    """Build the knowledge base if missing, then initialize the chatbot."""
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        build_vectorstore()

    return MedicalChatbot()

def render_sources(sources: list) -> None:
    """Show cited evidence, confidence, and raw retrieved chunks."""
    with st.expander("Retrieved medical sources and evidence"):
        for number, chunk in enumerate(sources, start=1):
            metadata = chunk.document.metadata
            filename = metadata.get("source_name") or metadata.get("source", "Unknown source")
            page = int(metadata.get("page", 0)) + 1
            st.markdown(f"**[{number}] {filename} — page {page}**  ")
            st.caption(f"Retrieval confidence: {clamp_confidence(chunk.relevance)}%")
            st.code(chunk.document.page_content, language=None)


def apply_theme(theme: str) -> None:
    """Apply a compact, responsive medical blue colour scheme."""
    dark = theme == "Dark"
    background, card, text = ("#071a2d", "#102a43", "#f0f7ff") if dark else ("#f7fbff", "#ffffff", "#142b4a")
    st.markdown(
        f"""<style>
        .stApp {{background: {background}; color: {text};}}
        [data-testid='stChatMessage'] {{background: {card}; border: 1px solid #cfe5ff; border-radius: 12px;}}
        .medical-header {{color:#1479c9; font-weight:700;}}
        footer {{visibility:hidden;}}
        @media (max-width: 640px) {{ .block-container {{padding: 1rem;}} }}
        </style>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render and operate the Streamlit chat UI."""
    st.set_page_config(page_title="AI Doctor Chatbot", page_icon="🩺", layout="wide")
    initialise_state()
    apply_theme(st.session_state.theme)

    with st.sidebar:
        st.header("🩺 AI Doctor Chatbot")
        st.caption("Retrieval-Augmented Medical Education")
        selected = st.radio("Appearance", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
        if selected != st.session_state.theme:
            st.session_state.theme = selected
            st.rerun()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        transcript = history_download(st.session_state.messages)
        st.download_button("Download chat", transcript, "medical_chat.md", "text/markdown", use_container_width=True)
        if st.button("Save chat history", use_container_width=True):
            saved = write_history(st.session_state.messages, HISTORY_DIR)
            st.success(f"Saved {saved.name}")
        st.divider()
        st.subheader("FAQ")
        for question, response in FAQS.items():
            with st.expander(question):
                st.write(response)
        st.divider()
        st.subheader("About")
        st.caption("Answers are grounded in PDFs you provide. Run `python ingest.py` whenever documents change.")

    st.title("AI Doctor Chatbot with RAG")
    st.caption("Ask an educational health question. Every response is grounded in your local medical document library.")
    st.info(DISCLAIMER)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    with st.form("medical_question", clear_on_submit=True):
        question = st.text_input("Your medical education question", placeholder="Type a health question…")
        submitted = st.form_submit_button("Send", type="primary")
    if not submitted:
        st.markdown("<p style='text-align:center'>For emergencies, seek immediate professional medical help.</p>", unsafe_allow_html=True)
        return
    question = question.strip()
    if not question:
        st.warning("Please enter a question before sending.")
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching trusted medical documents and preparing a response…"):
                bot = st.session_state.bot or create_bot()
                st.session_state.bot = bot
                answer, sources = bot.answer(question, st.session_state.messages[:-1])
            st.markdown(answer)
            render_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Unexpected application failure")
            st.error("An unexpected error occurred. Check the application logs and try again.")
    st.markdown("<p style='text-align:center'>" + DISCLAIMER + "</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
