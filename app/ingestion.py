import os
import re
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import (
    VECTORSTORE_DIR,
    BM25_CORPUS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def clean_medical_text(text: str) -> str:
    """Normalizes extracted text, fixing multi-column hyphenations and extra whitespace[cite: 2]."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ingest_verified_file(pdf_path: str) -> int:
    """Extracts, chunks, embeds, and permanently saves to FAISS & BM25 on disk."""
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # Pre-process extracted text
    for doc in raw_docs:
        doc.page_content = clean_medical_text(doc.page_content)

    # Chunking preserving sentence context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)
    new_passages = [c.page_content for c in chunks]

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss_index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")

    # 1. Update or Create Permanent FAISS Vector Store on Disk
    if os.path.exists(faiss_index_file):
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)

    # 2. Update or Create Sparse BM25 Postings on Disk
    corpus = []
    if os.path.exists(BM25_CORPUS_PATH):
        with open(BM25_CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)

    corpus.extend(new_passages)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)

    return len(chunks)