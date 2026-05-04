"""Project-wide configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# HotPotQA dataset
HOTPOTQA_SPLIT = os.getenv("HOTPOTQA_SPLIT", "train")
HOTPOTQA_SUBSET = os.getenv("HOTPOTQA_SUBSET", "fullwiki")
HOTPOTQA_SAMPLE_SIZE = int(os.getenv("HOTPOTQA_SAMPLE_SIZE", "500"))

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retriever
RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Evaluation
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))

# ── GraphRAG ──────────────────────────────────────────────────────
GRAPH_DIR = BASE_DIR / "graph_store"
GRAPH_DIR.mkdir(exist_ok=True)
GRAPH_VECTORSTORE_DIR = BASE_DIR / "graph_vectorstore"
GRAPH_VECTORSTORE_DIR.mkdir(exist_ok=True)

# Entity / relationship extraction
GRAPH_EXTRACT_MODEL = os.getenv("GRAPH_EXTRACT_MODEL", LLM_MODEL)
GRAPH_EXTRACT_TEMPERATURE = float(os.getenv("GRAPH_EXTRACT_TEMPERATURE", "0"))

# Graph traversal
GRAPH_TRAVERSAL_DEPTH = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "2"))
GRAPH_TOP_K_ENTITIES = int(os.getenv("GRAPH_TOP_K_ENTITIES", "5"))
GRAPH_COMMUNITY_RESOLUTION = float(os.getenv("GRAPH_COMMUNITY_RESOLUTION", "1.0"))
