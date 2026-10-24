import time
from typing import Dict, Any, List
from src.config import (
    TOP_N_RETRIEVAL,
    ADAPTIVE_FUSION_THRESHOLD,
    SYNTHESIS_MIN_CONFIDENCE
)
from src.retrieval.hybrid_search import HybridSearchEngine
from src.clustering.partitioner import partition_and_balance_subsets
from src.generation.drafter import run_parallel_drafters
from src.generation.verifier import verify_all_drafts, format_final_fluent_response

class SpeculativeRAGPipeline:
    def __init__(self):
        """Initializes the End-to-End Speculative RAG Engine."""
        print("[Pipeline] Initializing Speculative RAG Pipeline...")
        self.search_engine = HybridSearchEngine()
        self.search_engine.index_documents(force_reload=False)
        print("[Pipeline] System ready for queries!\n")

    async def query(self, query_str: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes the full 5-Phase Speculative RAG workflow on a user query.
        Returns final answer, winning draft, telemetry, and phase breakdown.
        """
        pipeline_start = time.perf_counter()
        telemetry = {}

        if verbose:
            print("=" * 80)
            print(f"QUERY: {query_str}")
            print("=" * 80)

        # -------------------------------------------------------------
        # Phase 1: Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        p1_start = time.perf_counter()
        ranked_chunks = self.search_engine.hybrid_search_rrf(query_str, top_n=TOP_N_RETRIEVAL)
        p1_time = time.perf_counter() - p1_start
        telemetry["phase_1_retrieval_ms"] = round(p1_time * 1000, 2)

        if verbose:
            print(f"\n[Phase 1] Hybrid Retrieval & RRF Complete ({telemetry['phase_1_retrieval_ms']} ms)")
            print(f"  - Retrieved {len(ranked_chunks)} fused chunks.")
            for c in ranked_chunks[:3]:
                print(f"    • Rank {c['final_rank']}: {c['id']} (RRF: {c['rrf_score']:.4f})")

        # -------------------------------------------------------------
        # Phase 2: Diversity Partitioning & Balanced Allocation
        # -------------------------------------------------------------
        p2_start = time.perf_counter()
        subsets, chosen_k, cluster_meta = partition_and_balance_subsets(ranked_chunks)
        p2_time = time.perf_counter() - p2_start
        telemetry["phase_2_clustering_ms"] = round(p2_time * 1000, 2)
        telemetry["chosen_k_clusters"] = chosen_k

        if verbose:
            print(f"\n[Phase 2] Dynamic Clustering & Modulo Allocation ({telemetry['phase_2_clustering_ms']} ms)")
            print(f"  - Evaluated optimal k: {chosen_k} (Silhouette: {cluster_meta['silhouette_score']:.4f})")
            print(f"  - Partitioned into {len(subsets)} balanced subsets (no weak subsets!).")

        # -------------------------------------------------------------
        # Phase 3: Parallel Asynchronous Drafting (Specialist Drafter)
        # -------------------------------------------------------------
        p3_start = time.perf_counter()
        candidate_drafts = await run_parallel_drafters(query_str, subsets)
        p3_time = time.perf_counter() - p3_start
        telemetry["phase_3_drafting_ms"] = round(p3_time * 1000, 2)

        if verbose:
            print(f"\n[Phase 3] Parallel Drafting Complete ({telemetry['phase_3_drafting_ms']} ms)")
            for d in candidate_drafts:
                print(f"  - Draft {d['draft_id']}: \"{d['answer'][:60]}...\"")

        # -------------------------------------------------------------
        # Phase 4: Single-Pass Verification (Generalist Verifier)
        # -------------------------------------------------------------
        p4_start = time.perf_counter()
        scored_drafts = await verify_all_drafts(query_str, candidate_drafts)
        p4_time = time.perf_counter() - p4_start
        telemetry["phase_4_verification_ms"] = round(p4_time * 1000, 2)

        if verbose:
            print(f"\n[Phase 4] Draft Verification & Scoring ({telemetry['phase_4_verification_ms']} ms)")
            for d in scored_drafts:
                print(f"  - Draft {d['draft_id']} [{d['decision']}]: Score={d['composite_score']} "
                      f"(ρ_Draft={d['rho_draft']}, ρ_SC={d['rho_sc']}, ρ_SR={d['rho_sr']})")

        # -------------------------------------------------------------
        # Phase 5: Arbitration & Adaptive Fusion Gating
        # -------------------------------------------------------------
        p5_start = time.perf_counter()
        top_1 = scored_drafts[0]
        top_2 = scored_drafts[1] if len(scored_drafts) > 1 else None

        final_answer = ""
        action_taken = ""

        # Check if Adaptive Fusion Gating should merge top 2 drafts
        if (top_2 and 
            (top_1["composite_score"] - top_2["composite_score"]) < ADAPTIVE_FUSION_THRESHOLD and 
            top_2["composite_score"] >= SYNTHESIS_MIN_CONFIDENCE and
            top_1["answer"].strip().lower() != top_2["answer"].strip().lower()):
            
            action_taken = "ADAPTIVE_SYNTHESIS"
            if verbose:
                print("\n[Phase 5] Adaptive Fusion Triggered! Score gap is narrow & complementary.")
            final_answer = await synthesize_complementary_drafts(query_str, top_1, top_2)
        else:
            action_taken = "DIRECT_ARGMAX"
            final_answer = await format_final_fluent_response(query_str, top_1)

        p5_time = time.perf_counter() - p5_start
        total_time = time.perf_counter() - pipeline_start

        telemetry["phase_5_arbitration_ms"] = round(p5_time * 1000, 2)
        telemetry["total_pipeline_ms"] = round(total_time * 1000, 2)
        telemetry["action"] = action_taken
        telemetry["winning_draft_id"] = top_1["draft_id"]
        telemetry["winning_confidence"] = top_1["composite_score"]

        # Approximate token savings calculation
        raw_chunks_tokens = sum(len(c["content"].split()) for c in ranked_chunks[:10])
        verifier_eval_tokens = len(query_str.split()) + len(top_1["answer"].split()) + len(top_1["rationale"].split())
        token_savings_pct = max(0, round((1.0 - (verifier_eval_tokens / max(1, raw_chunks_tokens))) * 100, 1))
        telemetry["verifier_token_savings_pct"] = token_savings_pct

        if verbose:
            print("\n" + "=" * 80)
            print("FINAL ANSWER:")
            print(final_answer)
            print("=" * 80)
            print(f"Total Pipeline Latency: {telemetry['total_pipeline_ms']} ms")
            print(f"Verifier Token Savings vs Naive RAG: {telemetry['verifier_token_savings_pct']}%")
            print(f"Decision Mode: {action_taken} (Winning Draft #{top_1['draft_id']})\n")

        return {
            "query": query_str,
            "final_answer": final_answer,
            "winning_draft": top_1,
            "all_drafts": scored_drafts,
            "telemetry": telemetry
        }
