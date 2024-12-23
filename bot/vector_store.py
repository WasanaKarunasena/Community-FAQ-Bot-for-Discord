from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load your FAQ dataset
def load_faq_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    faq_data = []
    for line in lines:
        question, answer = line.strip().split('|')
        faq_data.append({"question": question, "answer": answer})
    return faq_data

# Embed and store FAQ in FAISS
def setup_faiss(faq_data):
    embeddings = HuggingFaceEmbeddings()
    texts = [item["question"] for item in faq_data]
    faiss_store = FAISS.from_texts(texts, embeddings)
    return faiss_store, faq_data
