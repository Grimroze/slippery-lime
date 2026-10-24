import asyncio
import time
from pathlib import Path
import streamlit as st

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    DRAFTER_MODEL,
    VERIFIER_MODEL,
    ASSETS_DIR
)
from src.pipeline import SpeculativeRAGPipeline
from run_benchmark import run_standard_rag_sync

# ==============================================================================
# Streamlit App Configuration
# ==============================================================================
st.set_page_config(
    page_title="SpecuRAG | High-Throughput Speculative RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Modern Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .winner-badge {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Cache Pipeline Initialization
@st.cache_resource(show_spinner="Initializing Speculative RAG Engine & Vector Indices...")
def get_pipeline():
    return SpeculativeRAGPipeline()

pipeline = get_pipeline()

# ==============================================================================
# Sidebar: System Controls & Document Uploader
# ==============================================================================
with st.sidebar:
    st.title("⚡ SpecuRAG Engine")
    st.caption("Based on ICLR 2025 (Google DeepMind / UCSD)")
    st.divider()

    st.subheader("🤖 Active Models")
    st.info(f"**Embedder:** `{EMBEDDING_MODEL}`\n\n"
            f"**Drafter:** `{DRAFTER_MODEL}` (Parallel)\n\n"
            f"**Verifier:** `{VERIFIER_MODEL}` (Judge)")

    if st.button("🔄 Force Reload Engine", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs / TXT files:",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Save & Re-Index Documents", use_container_width=True):
            with st.spinner("Processing documents & updating vector indices..."):
                for uploaded_file in uploaded_files:
                    target_path = DATA_DIR / uploaded_file.name
                    with open(target_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                pipeline.search_engine.index_documents(force_reload=True)
                st.success(f"Successfully indexed {len(uploaded_files)} file(s)!")

    st.divider()
    st.caption("Architecture: Hybrid RRF → Silhouette (k=2,3) → Balanced Modulo → 5 Parallel Drafts → Single-Pass Verifier.")

# ==============================================================================
# Main Page: Head-to-Head Comparison Dashboard
# ==============================================================================
st.title("⚡ Speculative RAG: Live Comparison Dashboard")
st.markdown("Compare **Standard Naive RAG** vs **Speculative RAG** side-by-side on latency, token economics, and multi-perspective accuracy.")

# Sample Query Shortcuts
st.markdown("**Quick Test Prompts:**")
col_s1, col_s2, col_s3 = st.columns(3)
sample_query = ""
if col_s1.button("🎬 Nine to Five 1980 Film Actress"):
    sample_query = "Which actress/singer starred as Doralee Rhodes in the 1980 film Nine to Five?"
if col_s2.button("🎭 2010 Musical Tour Cast"):
    sample_query = "Who played the role of Doralee Rhodes in the 2010 national musical tour?"
if col_s3.button("🏆 Dolly Parton Career Achievements"):
    sample_query = "What major music awards and records has Dolly Parton achieved over her career?"

user_query = st.text_input(
    "Enter your knowledge question:",
    value=sample_query if sample_query else "Which actress/singer starred as Doralee Rhodes in the 1980 film Nine to Five?",
    help="Type any query based on your indexed PDFs or sample data."
)

if st.button("🚀 Run Side-by-Side Comparison", type="primary", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Executing Standard RAG and Speculative RAG pipelines..."):
            # 1. Run Standard RAG Baseline
            std_result = run_standard_rag_sync(pipeline.search_engine, user_query)

            # 2. Run Speculative RAG
            spec_result = asyncio.run(pipeline.query(user_query, verbose=False))
            spec_telemetry = spec_result["telemetry"]
            winning_draft = spec_result["winning_draft"]

        st.success("Execution Complete!")

        # -------------------------------------------------------------
        # Top-Level Head-to-Head Metrics
        # -------------------------------------------------------------
        m_col1, m_col2, m_col3 = st.columns(3)
        std_time = std_result["latency_sec"]
        spec_time = round(spec_telemetry["total_pipeline_ms"] / 1000.0, 3)
        speedup = round(((std_time - spec_time) / max(0.001, std_time)) * 100, 1)

        with m_col1:
            st.metric("Latency Comparison", f"{spec_time} s", delta=f"{speedup}% faster", delta_color="normal")
        with m_col2:
            st.metric("Verifier Token Savings", f"{spec_telemetry['verifier_token_savings_pct']}%", delta="Tokens Saved")
        with m_col3:
            st.metric("Winning Confidence Score", f"{spec_telemetry['winning_confidence']:.3f}", delta=f"Draft #{spec_telemetry['winning_draft_id']}")

        st.divider()

        # -------------------------------------------------------------
        # Side-by-Side Answer Comparison
        # -------------------------------------------------------------
        ans_col1, ans_col2 = st.columns(2)

        with ans_col1:
            st.subheader("🐢 Standard / Naive RAG")
            st.caption(f"Latency: `{std_time}s` | Input Tokens: `{std_result['input_tokens']}`")
            st.warning(std_result["answer"])
            st.caption("All 10 retrieved chunks were stuffed into the context prompt.")

        with ans_col2:
            st.subheader("⚡ Speculative RAG (Ours)")
            st.caption(f"Latency: `{spec_time}s` | Decision Mode: `{spec_telemetry['action']}`")
            st.success(spec_result["final_answer"])
            st.markdown(f"<span class='winner-badge'>🏆 Grounded in Draft #{winning_draft['draft_id']}</span>", unsafe_allow_html=True)
            st.caption(f"**Supporting Rationale:** {winning_draft['rationale']}")

        st.divider()

        # -------------------------------------------------------------
        # Deep Dive: The 5 Speculative Drafts Breakdown
        # -------------------------------------------------------------
        st.subheader("🔍 Speculative Drafts & Verifier Breakdown")
        num_drafts = len(spec_result["all_drafts"])
        st.write(f"All {num_drafts} drafts were generated concurrently in parallel by the 3B model, then evaluated by the Verifier:")

        draft_cols = st.columns(num_drafts)
        for i, d in enumerate(spec_result["all_drafts"]):
            with draft_cols[i]:
                is_win = (d["draft_id"] == spec_telemetry["winning_draft_id"])
                border_color = "#22C55E" if is_win else "#E2E8F0"
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; background-color: #FFFFFF;">
                    <b style="color: {'#15803D' if is_win else '#1E293B'};">Draft #{d['draft_id']} {'🏆' if is_win else ''}</b><br>
                    <small>Score: <b>{d['composite_score']}</b></small><br>
                    <small>ρ_SR: {d['rho_sr']} | ρ_SC: {d['rho_sc']}</small>
                    <hr style="margin: 6px 0;">
                    <small style="color: #64748B;"><b>Answer:</b> {d['answer'][:70]}...</small><br><br>
                    <small style="color: #475569;"><b>Rationale:</b> {d['rationale'][:80]}...</small>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# Architecture Diagram Expander
# ==============================================================================
st.divider()
with st.expander("🖼️ View System Architecture & Execution Flowchart"):
    svg_path = ASSETS_DIR / "speculative_rag_architecture.svg"
    if svg_path.exists():
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        st.image(svg_content, use_container_width=True)
    else:
        st.info("Architecture SVG available at: assets/speculative_rag_architecture.svg")
