"""
Evaluation Dashboard UI Component.
Visualizes benchmark results, retrieval relevance, citation validity, and latency.
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any, Optional

from src.evaluation import run_benchmark_suite, load_latest_benchmark_results, BENCHMARK_DATASET


def render_evaluation_dashboard(active_provider: str = "Groq"):
    st.markdown("### System Evaluation and Benchmark Dashboard")
    st.caption("Quantitative validation against standard Operating Systems questions (BCSE303P) according to Chapter 6.14.")

    col_ctrl, col_stats = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Benchmark Controls")
        sample_opt = st.selectbox("Benchmark Scope", ["Full Benchmark (8 Questions)", "Quick Test (3 Questions)"], index=0)
        n_sample = 3 if "Quick" in sample_opt else len(BENCHMARK_DATASET)

        provider_choice = st.selectbox("Benchmark Provider", ["Groq", "Gemini"], index=0 if active_provider == "Groq" else 1)

        run_btn = st.button("Run Benchmark Evaluation", type="primary", use_container_width=True)

    # Handle run
    if run_btn:
        progress_bar = st.progress(0, text="Starting benchmark evaluation...")
        def on_progress(current, total):
            ratio = current / total
            progress_bar.progress(ratio, text=f"Evaluating question {current}/{total}...")

        with st.spinner("Executing retrieval, generation, citation verification, and grounding checks..."):
            results = run_benchmark_suite(provider=provider_choice, sample_size=n_sample, progress_callback=on_progress)
            progress_bar.empty()
            st.success(f"Benchmark completed at {results['timestamp']} across {results['total_evaluated']} questions.")
    else:
        results = load_latest_benchmark_results()

    if not results:
        st.info("No benchmark results found. Click **Run Benchmark Evaluation** above to run the automated test suite.")
        return

    st.markdown("---")
    st.markdown(f"#### Performance Summary (Evaluated with `{results.get('provider', 'Groq')}` at {results.get('timestamp', 'N/A')})")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Retrieval Relevance", f"{results.get('retrieval_relevance_rate', 0)}%", help="Target document found in top-k hybrid search")
    with m2:
        st.metric("Citation Correctness", f"{results.get('citation_correctness_rate', 0)}%", help="Citations resolve to valid document and page number")
    with m3:
        st.metric("Grounding Score", f"{results.get('grounding_quality_score', 0)}%", help="Evidence alignment and factual grounding score")
    with m4:
        st.metric("Avg Latency", f"{results.get('average_latency_ms', 0):.0f} ms", help="Round-trip generation, retrieval, and validation time")

    st.markdown("#### Detailed Test Case Breakdown")
    cases = results.get("test_cases", [])

    if cases:
        # Convert to display table
        table_rows = []
        for c in cases:
            table_rows.append({
                "ID": c["id"],
                "Category": c["category"],
                "Question": c["question"][:55] + "...",
                "Retrieval": "Match" if c["retrieval_relevant"] else "Miss",
                "Citations": f"{c['citations_count']} Verified" if c["citation_valid"] else "None",
                "Grounding": f"{int(c['grounding_score'] * 100)}%",
                "Keyword Coverage": f"{int(c['keyword_coverage'] * 100)}%",
                "Latency": f"{c['latency_ms']:.0f} ms"
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Inspect Full Question Answers and Citations"):
            for c in cases:
                st.markdown(f"**[{c['id']}] {c['category']}: {c['question']}**")
                st.markdown(f"*Answer Preview:* {c['answer_preview']}")
                if c["citations"]:
                    st.caption(f"Citations: {', '.join([cit['doc_name'] + ' (p.' + str(cit['page']) + ')' for cit in c['citations']])}")
                st.markdown("---")

        # Download report
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="Download Benchmark Report (JSON)",
            data=json_data,
            file_name="ostutor_benchmark_report.json",
            mime="application/json"
        )

