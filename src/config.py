import os
from pathlib import Path

# ==============================================================================
# 1. Project Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
ASSETS_DIR = BASE_DIR / "assets"

# Automatically create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Ollama Configuration (Models detected on your machine)
# ==============================================================================
OLLAMA_HOST = "http://localhost:11434"

# Dense Embeddings: Lightweight & Blazing fast (from your ollama list)
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Specialist Drafter model (Small, ultra-fast 3B model for parallel drafting)
DRAFTER_MODEL = "llama3.2:3b"

# Generalist Verifier model (8B or 14B larger judge)
VERIFIER_MODEL = "llama3.1:latest"  # Tip: swap with "phi4:latest" anytime if you want a 14B judge!

# ==============================================================================
# 3. Speculative RAG Hyperparameters
# ==============================================================================
# Phase 1: Retrieval & Chunking
CHUNK_SIZE = 350                 # Words per chunk
CHUNK_OVERLAP = 50               # Overlap between consecutive chunks
TOP_N_RETRIEVAL = 12             # Top chunks selected via Reciprocal Rank Fusion
RRF_K = 60                       # Reciprocal Rank Fusion constant

# Phase 2: Diversity Partitioning & Allocation
NUM_DRAFTS_M = 3                 # 3 parallel drafts (optimal for laptop GPU/CPU concurrency)
MIN_K = 2                        # Lower bound of semantic clusters
MAX_K = 3                        # Upper bound of semantic clusters (guards against retrieval noise)

# Phase 5: Adaptive Fusion Gating
# If confidence margin between Top-1 and Top-2 is narrow (< 0.15), synthesize both!
ADAPTIVE_FUSION_THRESHOLD = 0.15
SYNTHESIS_MIN_CONFIDENCE = 0.60
