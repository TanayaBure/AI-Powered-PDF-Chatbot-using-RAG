import sys
import os

# Add the current directory to path so we can import chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=========================================")
print("AI-PDF-Chatbot Code Verification Script")
print("=========================================")

# 1. Check Project Modules
try:
    from chatbot.pdf_loader import load_and_split_pdf
    from chatbot.embeddings import get_embedding_model
    from chatbot.rag_pipeline import RAGPipeline
    print("[SUCCESS] Project modules imported successfully!")
except Exception as e:
    print(f"[FAIL] Failed to import project modules: {e}")
    sys.exit(1)

# 2. Check External Dependencies
dependencies = {
    "flask": "Flask (Backend)",
    "langchain": "LangChain (Core)",
    "langchain_community": "LangChain Community",
    "langchain_huggingface": "LangChain HuggingFace",
    "langchain_ollama": "LangChain Ollama",
    "faiss": "FAISS (Vector Database)",
    "pypdf": "PyPDF (PDF Loader)",
    "sentence_transformers": "Sentence Transformers (Embeddings)"
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"[OK] {name} is installed.")
    except ImportError:
        print(f"[MISSING] {name} is not installed.")
        missing.append(module)

print("=========================================")
if missing:
    print(f"Status: Code is structurally sound, but {len(missing)} dependencies are missing.")
    print("To install them, run: pip install -r requirements.txt")
else:
    print("Status: All dependencies are satisfied! Ready to run.")
print("=========================================")
