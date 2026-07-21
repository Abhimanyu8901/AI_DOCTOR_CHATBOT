# AI Doctor Chatbot with RAG
## Folder structure

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
