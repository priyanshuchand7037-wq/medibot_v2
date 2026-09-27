import os
from dotenv import load_dotenv

load_dotenv()

# Authentication
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin_medibot_secret_2026")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
VECTORSTORE_DIR = os.path.join(STORAGE_DIR, "master_vectorstore")
BM25_CORPUS_PATH = os.path.join(STORAGE_DIR, "bm25_corpus.pkl")

# Text Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Model Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"