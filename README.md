# AI Doctor Chatbot with RAG

A local, retrieval-augmented Streamlit chatbot for **medical education**. It searches trusted PDFs you supply, retrieves the five most relevant passages from a FAISS index, and asks Gemini to answer only from that evidence. It is not a diagnostic or treatment service.

## Features

- PDF ingestion with `PyPDFLoader`, semantic chunks, MiniLM embeddings, and persistent FAISS storage
- Strict evidence-grounded Gemini responses, citations, retrieved chunks, and relevance confidence
- Chat history, short conversational memory, clear/save/download controls, FAQ, About panel, responsive light/dark interface
- API-key and knowledge-base validation, logging, typed modular code, and user-facing error handling
- Medical disclaimer on every response. The app never prescribes medicines, recommends dosages, or diagnoses.

## Folder structure

```text
AI_Doctor_Chatbot/
├── app.py                 # Streamlit interface
├── ingest.py              # PDF-to-FAISS ingestion command
├── rag.py                 # Retrieval and grounded prompting
├── chatbot.py             # Gemini service
├── config.py
├── requirements.txt
├── data/                  # Put trusted medical PDFs here
├── vectorstore/           # Generated local index (ignored by Git)
├── chat_history/          # Saved chats (ignored by Git)
└── utils/helper.py
```

## Installation

Requires Python 3.11 or newer.

```powershell
cd AI_Doctor_Chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `GOOGLE_API_KEY` to a Gemini API key from Google AI Studio. `GEMINI_MODEL` defaults to `gemini-2.5-flash` and may be changed there if needed.

## Add documents and build the knowledge base

Place reputable, readable medical PDFs in `data/`; for example, official public-health guidance. Then run:

```powershell
python ingest.py
```

The command extracts pages, creates overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and writes `index.faiss` and its metadata to `vectorstore/`. Run it again after adding or replacing PDFs.

## Run the application

```powershell
streamlit run app.py
```

Open the local URL Streamlit displays. Ask a question, inspect the evidence expander under each response, use the sidebar to switch theme, clear history, download it, or save a JSON copy.

## Screenshots

Start the app and add screenshots of your configured interface to `screenshots/` if desired. They are intentionally not bundled because the displayed sources depend on your own medical PDF library.

## Safety and limitations

This tool provides educational summaries grounded in supplied PDFs only. It cannot evaluate your individual situation and may say it cannot find reliable information when the documents do not cover a question. For symptoms, diagnosis, treatment decisions, or emergencies, contact a licensed healthcare professional or local emergency service.

## Future improvements

- Curated source registry with document versioning and review dates
- Authentication, audit logs, and encrypted storage for any real deployment
- Source-quality scoring and clinical safety evaluation suite
- OCR pipeline for scanned PDFs and multilingual document collections
