from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    # Initialize Hugging Face embedding model
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
