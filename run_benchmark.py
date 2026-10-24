import asyncio
import time
from typing import List, Dict, Any
from tabulate import tabulate
import ollama

from src.config import OLLAMA_HOST, VERIFIER_MODEL
from src.retrieval.hybrid_search import HybridSearchEngine
from src.pipeline import SpeculativeRAGPipeline

ollama_client = ollama.Client(host=OLLAMA_HOST)

def run_standard_rag_sync(search_engine: HybridSearchEngine, query: str) -> Dict[str, Any]:
    """
    Standard / Naive RAG Baseline:
    1. Fetches top 10 chunks.
    2. Stuffs ALL 10 chunks into the large model's context window.
    3. Runs single serial completion call.
    """
    start_time = time.perf_counter()

    # Retrieve 10 chunks
    dense_hits = search_engine.search_dense(query, top_k=10)

    # Concatenate all into one big prompt context (causes 'lost-in-the-middle')
    context_text = "\n\n".join([f"[{i+1}] {h['content']}" for i, h in enumerate(dense_hits)])
    prompt = f"""Answer the question based strictly on the following retrieved context.

Context:
{context_text}

Question: {query}
Answer:"""

    input_tokens = len(prompt.split())

    response = ollama_client.generate(
        model=VERIFIER_MODEL,
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 120}
    )
    elapsed = time.perf_counter() - start_time

    return {
        "answer": response.get("response", "").strip(),
        "latency_sec": round(elapsed, 3),
        "input_tokens": input_tokens
    }

async def run_full_benchmark():
    """Runs a side-by-side benchmark comparing Standard RAG vs Speculative RAG."""
    print("=" * 80)
    print("   STARTING EMPIRICAL BENCHMARK: STANDARD RAG vs. SPECULATIVE RAG")
    print("=" * 80)

    # Test queries designed to probe factoids, multi-perspective, and potential distractors
    test_queries = [
        "Which actress/singer starred as Doralee Rhodes in the 1980 film Nine to Five?",
        "Who played the role of Doralee Rhodes in the 2010 national musical tour?",
        "What acting roles is Cynthia Rhodes famous for, and did she star in Nine to Five?",
        "What major music awards and records has Dolly Parton achieved over her career?",
        "Why does naive RAG suffer from position bias and latency when context is large?"
    ]

    pipeline = SpeculativeRAGPipeline()
    search_engine = pipeline.search_engine

    results_table = []
    total_std_time = 0.0
    total_spec_time = 0.0
    total_std_tokens = 0
    total_spec_tokens = 0

    print(f"\nEvaluating {len(test_queries)} queries side-by-side...\n")

    for idx, q in enumerate(test_queries, start=1):
        print(f"[{idx}/{len(test_queries)}] Running: \"{q[:55]}...\"")

        # 1. Standard RAG Run
        std_res = run_standard_rag_sync(search_engine, q)

        # 2. Speculative RAG Run
        spec_res = await pipeline.query(q, verbose=False)
        spec_telemetry = spec_res["telemetry"]
        spec_time_sec = round(spec_telemetry["total_pipeline_ms"] / 1000.0, 3)

        # Token estimate for speculative verifier
        winning_draft = spec_res["winning_draft"]
        spec_tokens = len(q.split()) + len(winning_draft["answer"].split()) + len(winning_draft["rationale"].split())

        # Accumulate stats
        total_std_time += std_res["latency_sec"]
        total_spec_time += spec_time_sec
        total_std_tokens += std_res["input_tokens"]
        total_spec_tokens += spec_tokens

        # Latency difference percentage
        lat_diff_pct = round(((std_res["latency_sec"] - spec_time_sec) / max(0.001, std_res["latency_sec"])) * 100, 1)

        results_table.append([
            f"Q{idx}",
            f"{std_res['latency_sec']}s",
            f"{spec_time_sec}s",
            f"{lat_diff_pct}%",
            f"{std_res['input_tokens']}",
            f"{spec_tokens}",
            f"{spec_telemetry['winning_confidence']:.3f}",
            spec_telemetry["action"]
        ])

    # Print summary table
    headers = [
        "Query ID",
        "Std RAG Time",
        "SpecuRAG Time",
        "Speedup %",
        "Std Verifier Tokens",
        "Specu Verifier Tokens",
        "Winning Conf.",
        "Decision Mode"
    ]

    print("\n" + "=" * 90)
    print("                           DETAILED BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))

    avg_std_time = total_std_time / len(test_queries)
    avg_spec_time = total_spec_time / len(test_queries)
    avg_speedup = round(((avg_std_time - avg_spec_time) / avg_std_time) * 100, 1)
    avg_token_reduction = round(((total_std_tokens - total_spec_tokens) / total_std_tokens) * 100, 1)

    print("\n" + "=" * 90)
    print("                             EXECUTIVE SUMMARY METRICS")
    print("=" * 90)
    print(f"• Average Standard RAG Latency   : {avg_std_time:.3f} seconds")
    print(f"• Average Speculative RAG Latency: {avg_spec_time:.3f} seconds")
    print(f"• Overall Latency Reduction       : {avg_speedup}% Faster ⚡")
    print(f"• Verifier Context Token Savings  : {avg_token_reduction}% Saved 📉")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    asyncio.run(run_full_benchmark())
