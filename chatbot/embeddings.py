from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a HuggingFaceEmbeddings instance using the lightweight and 
    highly effective 'all-MiniLM-L6-v2' model, running on CPU.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # True helps with cosine similarity
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
