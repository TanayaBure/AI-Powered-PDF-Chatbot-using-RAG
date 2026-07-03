# AI-Powered PDF Chatbot using RAG

A local, privacy-focused Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and chat with them. Built using Python, Flask, LangChain, FAISS, Hugging Face Embeddings, and Ollama (Llama 3.2).

## Key Achievements & Features

- **Developed a Retrieval-Augmented Generation (RAG) application** to answer user questions from uploaded PDF documents.
- **Implemented document ingestion, text chunking, vector embeddings, and semantic search** using FAISS.
- **Integrated a local LLM** through Ollama (`llama3.2`) to generate context-aware responses.
- **Built a Flask-based web interface** for PDF upload and interactive question answering with a modern dark-mode, glassmorphism design.
- **Used Git for version control** and followed a clean, modular project architecture.
- **100% Local & Private**: No data leaves your machine. Both embeddings and LLM generation run locally.
- **Fast Retrieval**: Powered by FAISS (vector database) and HuggingFace's `all-MiniLM-L6-v2` embedding model.

---

## Prerequisites

1. **Python 3.8+** installed.
2. **Ollama** installed and running on your system:
   - Download Ollama from [ollama.com](https://ollama.com/).
   - Start the Ollama application.
   - Pull the Llama 3.2 model:
     ```bash
     ollama pull llama3.2
     ```

---

## Installation & Setup

1. **Clone or Navigate to the project directory**:
   ```bash
   cd AI-PDF-Chatbot
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

1. Make sure Ollama is running in the background and you have pulled the model (`llama3.2`).
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Project Structure

```
AI-PDF-Chatbot/
│
├── app.py                  # Flask web server & routes
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── chatbot/
│   ├── __init__.py
│   ├── pdf_loader.py       # PDF text extraction & splitting
│   ├── embeddings.py       # Local HuggingFace embeddings config
│   └── rag_pipeline.py     # Vector store & Ollama LLM RAG pipeline
│
├── templates/
│   └── index.html          # Web UI layout
│
├── static/
│   └── style.css           # Premium dark-theme styles
│
└── uploads/                # Directory where uploaded PDFs are stored
```
