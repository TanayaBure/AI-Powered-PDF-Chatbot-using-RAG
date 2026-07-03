import os

def get_embedding_model():
    """
    Returns an embeddings model instance.
    - If OPENAI_API_KEY is present, uses OpenAIEmbeddings.
    - If HF_TOKEN is present or running on Vercel, uses HuggingFaceHubEmbeddings (cloud API).
    - Otherwise, falls back to local HuggingFaceEmbeddings (requires local torch/sentence-transformers).
    """
    # 1. Check for OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
    
    # 2. Check for HuggingFace Cloud
    elif os.environ.get("HF_TOKEN") or os.environ.get("VERCEL"):
        from langchain_community.embeddings import HuggingFaceHubEmbeddings
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required for HuggingFace embeddings on Vercel.")
        return HuggingFaceHubEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
    
    # 3. Fallback to local HuggingFace (imported dynamically to avoid importing torch/transformers on Vercel)
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
