import asyncio
import time
from typing import List, Dict, Any
from ollama import AsyncClient

from src.config import OLLAMA_HOST, VERIFIER_MODEL

async_verifier_client = AsyncClient(host=OLLAMA_HOST)

async def evaluate_single_draft(
    query: str,
    draft: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates a candidate draft without re-reading raw chunks!
    Computes:
      - Self-Consistency (ρ_SC): Coherence of (Q, α, β)
      - Self-Reflection (ρ_SR): Verification probability ("Does rationale support answer?")
      - Final Composite Score: ρ = ρ_Draft × ρ_SC × ρ_SR
    """
    answer = draft["answer"]
    rationale = draft["rationale"]
    rho_draft = draft.get("rho_draft", 0.7)

    # If draft is empty or failed, immediately assign zero score
    if not answer or not rationale or "failed" in answer.lower():
        draft["rho_sc"] = 0.0
        draft["rho_sr"] = 0.0
        draft["composite_score"] = 0.0
        return draft

    # 1. Prompt for Self-Reflection verification
    verify_prompt = f"""You are an unbiased Judge. Evaluate whether the provided Rationale logically supports the Answer for the given Question.

Question: {query}
Answer: {answer}
Rationale: {rationale}

Does the rationale strictly support and prove the answer? 
Respond with ONLY ONE word: 'YES' or 'NO', followed by your confidence score from 0.0 to 1.0 (e.g. 'YES 0.95')."""

    start_time = time.perf_counter()
    try:
        response = await async_verifier_client.generate(
            model=VERIFIER_MODEL,
            prompt=verify_prompt,
            options={
                "temperature": 0.0,
                "num_predict": 10   # Only need 1-2 tokens! Ultra-fast!
            }
        )
        elapsed = time.perf_counter() - start_time
        reply = response.get("response", "").strip().upper()

        # Parse Self-Reflection Score
        is_yes = "YES" in reply
        rho_sr = 0.85 if is_yes else 0.20

        # Extract numeric confidence if model returned one
        import re
        nums = re.findall(r"0\.\d+|1\.0", reply)
        if nums:
            parsed_conf = float(nums[0])
            if is_yes:
                rho_sr = max(0.60, min(0.99, parsed_conf))
            else:
                rho_sr = min(0.30, parsed_conf)

        # 2. Self-Consistency / Semantic Alignment heuristic
        # Measures overlap of critical nouns between question, rationale, and answer
        q_words = set(re.findall(r"\w+", query.lower()))
        ans_words = set(re.findall(r"\w+", answer.lower()))
        rat_words = set(re.findall(r"\w+", rationale.lower()))

        overlap_q = len(q_words.intersection(rat_words)) / max(1, len(q_words))
        rho_sc = min(0.95, 0.40 + (0.50 * overlap_q))

        # 3. Composite Multiplicative Score: ρ = ρ_Draft × ρ_SC × ρ_SR
        composite_score = round(rho_draft * rho_sc * rho_sr, 4)

        draft["rho_sc"] = round(rho_sc, 3)
        draft["rho_sr"] = round(rho_sr, 3)
        draft["composite_score"] = composite_score
        draft["verifier_latency"] = elapsed
        draft["decision"] = "APPROVED" if is_yes else "REJECTED"

        return draft

    except Exception as e:
        print(f"[Verifier Error] Failed verifying draft {draft.get('draft_id')}: {e}")
        draft["rho_sc"] = 0.3
        draft["rho_sr"] = 0.2
        draft["composite_score"] = 0.05
        draft["decision"] = "ERROR"
        return draft

async def verify_all_drafts(
    query: str,
    drafts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Runs single-pass evaluation across all candidate drafts in parallel."""
    tasks = [evaluate_single_draft(query, d) for d in drafts]
    scored_drafts = await asyncio.gather(*tasks)

    # Sort descending by composite score (Winner at index 0)
    scored_drafts.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored_drafts

async def format_final_fluent_response(
    query: str,
    winning_draft: Dict[str, Any]
) -> str:
    """
    Paper Page 4 Line 15: Integrates the winning draft and its rationale
    into a fluent, well-crafted response via the generalist LM.
    Takes ~50 tokens (< 0.8s), giving high-quality conversational output.
    """
    prompt = f"""You are a professional assistant. Using the winning fact and rationale below, write a direct, fluent, and complete response to the user's question.

Question: {query}
Key Fact: {winning_draft['answer']}
Supporting Evidence: {winning_draft['rationale']}

Direct Response:"""

    response = await async_verifier_client.generate(
        model=VERIFIER_MODEL,
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 100}
    )
    return response.get("response", "").strip()

