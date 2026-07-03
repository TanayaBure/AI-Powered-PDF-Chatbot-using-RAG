import os
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    """
    Mock embeddings class to prevent crashes on Vercel when no API keys are provided.
    Generates dummy vectors of size 384.
    """
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
        
    def embed_query(self, text):
        return [0.1] * 384

def get_embedding_model():
    """
    Returns an embeddings model instance.
    - If OPENAI_API_KEY is present, uses OpenAIEmbeddings.
    - If HF_TOKEN is present, uses HuggingFaceHubEmbeddings (cloud API).
    - If on Vercel (without keys), uses MockEmbeddings to allow demo usage.
    - Otherwise, falls back to local HuggingFaceEmbeddings.
    """
    # 1. Check for OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
    
    # 2. Check for HuggingFace Cloud
    elif os.environ.get("HF_TOKEN"):
        from langchain_community.embeddings import HuggingFaceHubEmbeddings
        return HuggingFaceHubEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ.get("HF_TOKEN")
        )
    
    # 3. Check if running on Vercel (without keys) -> Fallback to Mock
    elif os.environ.get("VERCEL"):
        return MockEmbeddings()
    
    # 4. Fallback to local HuggingFace
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
