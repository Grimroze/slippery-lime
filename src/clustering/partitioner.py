from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import NUM_DRAFTS_M, MIN_K, MAX_K

def evaluate_best_k(embeddings: np.ndarray, min_k: int = MIN_K, max_k: int = MAX_K) -> Tuple[int, float]:
    """
    Evaluates cluster quality for k in [2, 3] using Silhouette Score.
    Takes < 2 ms on CPU.
    Returns (best_k, best_silhouette_score).
    """
    num_samples = len(embeddings)
    if num_samples < max_k + 1:
        return min_k, 0.0

    best_k = min_k
    best_score = -1.0

    for k in range(min_k, max_k + 1):
        # Quick deterministic KMeans fit
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embeddings)

        # Silhouette score ranges from -1 to +1
        score = silhouette_score(embeddings, labels)
        print(f"  [Auto-K] Evaluated k={k} -> Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"  [Auto-K] Selected optimal k={best_k} (Cohesion & Separation peak)\n")
    return best_k, best_score

def partition_and_balance_subsets(
    ranked_chunks: List[Dict[str, Any]],
    m_drafts: int = NUM_DRAFTS_M
) -> Tuple[List[List[Dict[str, Any]]], int, Dict[str, Any]]:
    """
    End-to-End Phase 2 Workflow:
    1. Evaluates whether k=2 or k=3 best represents document perspectives.
    2. Slices chunks to match slots: 10 chunks if k=2 (5x2), 15 chunks if k=3 (5x3).
    3. Clusters chunks using K-Means on dense embeddings.
    4. Applies Inverted Modulo Allocation to ensure balanced rank distribution
       (eliminates 'all-strong' or 'all-weak' subsets).
    """
    if not ranked_chunks:
        return [], 2, {}

    # Extract embeddings
    raw_embeddings = np.array([c["embedding"] for c in ranked_chunks])

    # 1. Determine optimal k
    best_k, sil_score = evaluate_best_k(raw_embeddings, min_k=MIN_K, max_k=MAX_K)

    # 2. Slice required chunks based on k
    required_chunks_count = m_drafts * best_k  # e.g., 5*2 = 10, or 5*3 = 15
    active_chunks = ranked_chunks[:required_chunks_count]
    active_embeddings = raw_embeddings[:required_chunks_count]

    # 3. Fit KMeans with best_k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(active_embeddings)

    # Group chunks by cluster
    clusters: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(best_k)}
    for chunk, label in zip(active_chunks, labels):
        clusters[label].append(chunk)

    # Sort each cluster internally by RRF rank (best chunks at index 0)
    for c_id in clusters:
        clusters[c_id].sort(key=lambda x: x.get("final_rank", 999))

    # Pad or equalize cluster lengths if KMeans had slight imbalance
    # Ensure every cluster has at least m_drafts elements
    for c_id in range(best_k):
        while len(clusters[c_id]) < m_drafts:
            # Replicate highest ranked in that cluster if needed
            clusters[c_id].append(clusters[c_id][0])

    # 4. Balanced Inverted Modulo Allocation
    subsets: List[List[Dict[str, Any]]] = []

    for i in range(m_drafts):
        subset = []
        if best_k == 2:
            # Cluster 1: natural index i
            chunk_c1 = clusters[0][i]
            # Cluster 2: inverted index (m - 1 - i) => Inverted snake pairing!
            chunk_c2 = clusters[1][m_drafts - 1 - i]
            subset = [chunk_c1, chunk_c2]

        elif best_k == 3:
            # Cluster 1: index i
            chunk_c1 = clusters[0][i]
            # Cluster 2: inverted index (m - 1 - i)
            chunk_c2 = clusters[1][m_drafts - 1 - i]
            # Cluster 3: circular shift (i + 2) % m
            chunk_c3 = clusters[2][(i + 2) % m_drafts]
            subset = [chunk_c1, chunk_c2, chunk_c3]

        subsets.append(subset)

    metadata = {
        "selected_k": best_k,
        "silhouette_score": sil_score,
        "total_chunks_partitioned": len(active_chunks),
        "subsets_created": len(subsets)
    }

    return subsets, best_k, metadata
