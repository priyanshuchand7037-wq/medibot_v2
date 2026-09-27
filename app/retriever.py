import os
import pickle
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.config import (
    VECTORSTORE_DIR,
    BM25_CORPUS_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# In-Memory RAM Caches for Instant User Queries
cached_faiss: FAISS = None
cached_bm25: BM25Okapi = None
cached_corpus: List[str] = []

def load_knowledge_base():
    """Reads disk indexes directly into server RAM during boot[cite: 1]."""
    global cached_faiss, cached_bm25, cached_corpus
    faiss_index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")

    if os.path.exists(faiss_index_file):
        cached_faiss = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ FAISS Master Index loaded into server RAM[cite: 1].")

    if os.path.exists(BM25_CORPUS_PATH):
        with open(BM25_CORPUS_PATH, "rb") as f:
            cached_corpus = pickle.load(f)
            tokenized_corpus = [doc.lower().split() for doc in cached_corpus]
            cached_bm25 = BM25Okapi(tokenized_corpus)
        print("✓ BM25 Sparse Index initialized in server RAM[cite: 1].")

def hybrid_retrieve(query: str, top_n: int = 3) -> str:
    """Executes dense + sparse search with Cross-Encoder re-ranking[cite: 1]."""
    if not cached_faiss or not cached_bm25:
        return ""

    # Dense vector similarity (FAISS top-10)
    dense_docs = cached_faiss.similarity_search(query, k=10)
    dense_passages = [d.page_content for d in dense_docs]

    # Sparse lexical keyword matching (BM25 top-10)
    tokenized_query = query.lower().split()
    bm25_passages = cached_bm25.get_top_n(tokenized_query, cached_corpus, n=10)

    # Candidate deduplication
    candidates = list(set(dense_passages + bm25_passages))

    # Cross-Encoder re-ranking
    pairs = [(query, passage) for passage in candidates]
    scores = reranker.predict(pairs)
    ranked = [passage for _, passage in sorted(zip(scores, candidates), reverse=True)]

    return "\n\n".join(ranked[:top_n])