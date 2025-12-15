from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from embeddings import get_embeddings

def load_faq_dataset(file_path):
    documents = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            question, answer = line.strip().split("|")
            content = f"Question: {question}\nAnswer: {answer}"
            documents.append(Document(page_content=content))

    return documents


def create_faiss_store(file_path):
    embeddings = get_embeddings()
    documents = load_faq_dataset(file_path)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
