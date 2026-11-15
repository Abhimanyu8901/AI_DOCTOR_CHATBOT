# AI Doctor Chatbot with RAG || [Live Link](https://ai-doctor-chatbott.streamlit.app/)

An AI-powered medical education chatbot built using Retrieval-Augmented Generation (RAG). The application uses Groq API (Llama 3.3 70B) to generate context-aware responses based on information retrieved from medical PDF documents.

It combines Large Language Models (LLMs), semantic search, and vector databases to provide educational answers grounded in a custom medical knowledge base.

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
