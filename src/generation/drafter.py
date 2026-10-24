import asyncio
import re
import time
from typing import List, Dict, Any
from ollama import AsyncClient

from src.config import OLLAMA_HOST, DRAFTER_MODEL

# Asynchronous client for concurrent drafting
async_client = AsyncClient(host=OLLAMA_HOST)

def parse_drafter_output(raw_text: str) -> Dict[str, str]:
    """
    Extracts Rationale (β) and Answer (α) from model response.
    Expects standard format:
    ## Rationale:
    ...
    ## Answer:
    ...
    """
    rationale = ""
    answer = ""

    # Regex search for sections
    rat_match = re.search(r"##\s*Rationale:?\s*(.*?)(?=##\s*Answer:?|$)", raw_text, re.DOTALL | re.IGNORECASE)
    ans_match = re.search(r"##\s*Answer:?\s*(.*)", raw_text, re.DOTALL | re.IGNORECASE)

    if rat_match:
        rationale = rat_match.group(1).strip()
    if ans_match:
        answer = ans_match.group(1).strip()

    # Fallback if model omitted explicit headers
    if not answer and not rationale:
        parts = raw_text.strip().split("\n\n", 1)
        if len(parts) == 2:
            rationale = parts[0].strip()
            answer = parts[1].strip()
        else:
            answer = raw_text.strip()
            rationale = "Direct evidence cited from document subset."

    return {"rationale": rationale, "answer": answer}

async def generate_single_draft(
    query: str,
    subset_chunks: List[Dict[str, Any]],
    draft_id: int
) -> Dict[str, Any]:
    """
    Executes a single specialist drafter instance on a document subset.
    Prompt enforces Chain-of-Thought (Rationale β generated before Answer α).
    """
    # Format documents context
    docs_context = ""
    for idx, chunk in enumerate(subset_chunks, start=1):
        docs_context += f"[{idx}] Source: {chunk.get('source', 'doc')}\n{chunk['content']}\n\n"

    prompt = f"""You are a Specialist RAG Drafter. Answer the query based ONLY on the provided evidence documents.
IMPORTANT: You MUST write the intermediate Rationale FIRST before stating the final answer.

## Evidence:
{docs_context.strip()}

## Instruction:
{query}

Format your response strictly as:
## Rationale:
(Summarize the specific facts from the evidence that answer the question in 1-2 concise sentences.)
## Answer:
(State the direct, concise answer.)"""

    start_time = time.perf_counter()
    try:
        response = await async_client.generate(
            model=DRAFTER_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.0,       # Deterministic greedy decoding
                "top_p": 0.9,
                "num_predict": 150        # Keep drafts short and punchy
            }
        )
        elapsed = time.perf_counter() - start_time
        raw_text = response.get("response", "").strip()

        parsed = parse_drafter_output(raw_text)

        # Baseline heuristic confidence: length penalty + token density
        tokens_eval = response.get("eval_count", len(raw_text.split()))
        rho_draft = 0.70  # Baseline Drafter confidence

        # Bonus if both rationale and answer were cleanly populated
        if len(parsed["rationale"]) > 15 and len(parsed["answer"]) > 2:
            rho_draft = 0.75

        return {
            "draft_id": draft_id,
            "answer": parsed["answer"],
            "rationale": parsed["rationale"],
            "raw_text": raw_text,
            "rho_draft": rho_draft,
            "latency": elapsed,
            "tokens_generated": tokens_eval,
            "subset_chunk_ids": [c["id"] for c in subset_chunks]
        }

    except Exception as e:
        print(f"[Drafter Error] Draft {draft_id} failed: {e}")
        return {
            "draft_id": draft_id,
            "answer": "Failed to generate draft.",
            "rationale": str(e),
            "rho_draft": 0.1,
            "latency": 0.0,
            "tokens_generated": 0,
            "subset_chunk_ids": []
        }

async def run_parallel_drafters(
    query: str,
    subsets: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Orchestrates all m drafts concurrently using AsyncIO.
    Fires all requests in parallel so total time ~ 1 single call!
    """
    start_total = time.perf_counter()
    tasks = [
        generate_single_draft(query=query, subset_chunks=subset, draft_id=i + 1)
        for i, subset in enumerate(subsets)
    ]

    drafts = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - start_total

    print(f"[Drafters] Generated {len(drafts)} drafts concurrently in {total_elapsed:.3f} seconds.")
    return drafts
