import re
from typing import List, Dict, Any, Tuple
import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi

from src.config import (
    CHROMA_DB_DIR,
    OLLAMA_HOST,
    EMBEDDING_MODEL,
    TOP_N_RETRIEVAL,
    RRF_K
)
from src.retrieval.document_loader import load_all_documents

# Initialize Ollama Client
ollama_client = ollama.Client(host=OLLAMA_HOST)

def get_embedding(text: str) -> List[float]:
    """Generates dense vector embedding using local Ollama model."""
    try:
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[Embedding Error] Failed to embed text with {EMBEDDING_MODEL}: {e}")
        raise e

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())

class HybridSearchEngine:
    def __init__(self, collection_name: str = "specu_rag_docs"):
        """Initializes ChromaDB (Dense) and BM25 (Sparse) retrieval indices."""
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Dense vectors for Speculative RAG"}
        )
        self.bm25_index = None
        self.doc_chunks: List[Dict[str, Any]] = []

    def index_documents(self, force_reload: bool = False):
        """
        Loads all documents from data folder, computes embeddings,
        and saves them into ChromaDB and BM25 index.
        """
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reload:
            print(f"[HybridSearch] ChromaDB already contains {existing_count} chunks. Loading from DB...")
            # Retrieve existing chunks from ChromaDB for BM25 building
            stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self.doc_chunks = []
            for i, doc_id in enumerate(stored["ids"]):
                self.doc_chunks.append({
                    "id": doc_id,
                    "content": stored["documents"][i],
                    "source": stored["metadatas"][i].get("source", "unknown"),
                    "embedding": stored["embeddings"][i] if stored["embeddings"] is not None else None
                })
        else:
            # Fresh index from files
            raw_chunks = load_all_documents()
            if not raw_chunks:
                print("[HybridSearch] No documents available to index.")
                return

            print(f"[HybridSearch] Computing embeddings for {len(raw_chunks)} chunks using {EMBEDDING_MODEL}...")
            ids = []
            documents = []
            metadatas = []
            embeddings = []

            for chunk in raw_chunks:
                emb = get_embedding(chunk["content"])
                ids.append(chunk["id"])
                documents.append(chunk["content"])
                metadatas.append({"source": chunk["source"]})
                embeddings.append(emb)

                chunk["embedding"] = emb

            # Store in ChromaDB
            if existing_count > 0:
                self.collection.delete(ids=self.collection.get()["ids"])

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            self.doc_chunks = raw_chunks
            print(f"[HybridSearch] Successfully indexed {len(ids)} chunks in ChromaDB.")

        # Build BM25 Inverted Index
        corpus = [tokenize_for_bm25(doc["content"]) for doc in self.doc_chunks]
        self.bm25_index = BM25Okapi(corpus)
        print(f"[HybridSearch] BM25 keyword index ready with {len(corpus)} documents.\n")

    def search_dense(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Dense semantic search using ChromaDB."""
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.doc_chunks)),
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        dense_hits = []
        if results and results["ids"] and results["ids"][0]:
            for rank, doc_id in enumerate(results["ids"][0]):
                dense_hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][rank],
                    "source": results["metadatas"][0][rank].get("source", "unknown"),
                    "distance": results["distances"][0][rank],
                    "embedding": results["embeddings"][0][rank] if results["embeddings"] else None,
                    "dense_rank": rank + 1
                })
        return dense_hits

    def search_sparse_bm25(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Sparse keyword search using BM25Okapi."""
        if not self.bm25_index or not self.doc_chunks:
            return []

        tokenized_query = tokenize_for_bm25(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        # Sort indices in descending order of BM25 score
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(self.doc_chunks))]

        sparse_hits = []
        for rank, idx in enumerate(top_indices):
            chunk = self.doc_chunks[idx]
            sparse_hits.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "source": chunk.get("source", "unknown"),
                "bm25_score": float(scores[idx]),
                "embedding": chunk.get("embedding"),
                "bm25_rank": rank + 1
            })
        return sparse_hits

    def hybrid_search_rrf(self, query: str, top_n: int = TOP_N_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Combines Dense & BM25 results using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score(d) = 1/(60 + Rank_dense) + 1/(60 + Rank_bm25)
        Returns top_n re-ranked chunks with their embeddings attached.
        """
        dense_results = self.search_dense(query, top_k=top_n)
        bm25_results = self.search_sparse_bm25(query, top_k=top_n)

        # Map to store combined chunk metadata & RRF score
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Add Dense ranks
        for item in dense_results:
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + item["dense_rank"]))

        # 2. Add BM25 ranks
        for item in bm25_results:
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + item["bm25_rank"]))

        # 3. Sort chunks descending by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_chunks = []
        for rank, cid in enumerate(sorted_cids[:top_n]):
            chunk_info = chunk_map[cid]
            chunk_info["rrf_score"] = rrf_scores[cid]
            chunk_info["final_rank"] = rank + 1

            # Ensure embedding is present (fetch if missing)
            if chunk_info.get("embedding") is None:
                chunk_info["embedding"] = get_embedding(chunk_info["content"])

            final_chunks.append(chunk_info)

        return final_chunks
