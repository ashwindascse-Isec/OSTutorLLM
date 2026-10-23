"""
Automated Evaluation and Benchmarking Suite.
Runs a fixed benchmark of representative Operating Systems questions,
evaluating Retrieval Relevance, Citation Correctness, Grounding Quality, and Latency.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import DATA_DIR
from src.library import DocumentLibrary
from src.retrieval import hybrid_search
from src.tutor import generate_tutor_response

BENCHMARK_RESULTS_FILE = DATA_DIR / "evaluation_results.json"

# Fixed benchmark dataset of standard Operating Systems exam/lab questions
BENCHMARK_DATASET = [
    {
        "id": "bench_01",
        "category": "Deadlocks",
        "question": "What are the four necessary Coffman conditions required for a deadlock to occur?",
        "expected_keywords": ["mutual exclusion", "hold and wait", "no preemption", "circular wait"],
        "expected_doc": "os_deadlocks"
    },
    {
        "id": "bench_02",
        "category": "Virtual Memory",
        "question": "What is a page fault and what sequence of steps does the OS take to service it?",
        "expected_keywords": ["page fault", "invalid", "frame", "trap", "disk", "secondary storage"],
        "expected_doc": "os_virtual_memory"
    },
    {
        "id": "bench_03",
        "category": "CPU Scheduling",
        "question": "How does the Round Robin CPU scheduling algorithm work and how does the time quantum affect its performance?",
        "expected_keywords": ["round robin", "time quantum", "preemption", "fcfs", "context switch"],
        "expected_doc": "os_cpu_scheduling"
    },
    {
        "id": "bench_04",
        "category": "Process Synchronization",
        "question": "What are the three essential requirements for a valid solution to the critical-section problem?",
        "expected_keywords": ["mutual exclusion", "progress", "bounded waiting"],
        "expected_doc": "os_process_synchronization"
    },
    {
        "id": "bench_05",
        "category": "Deadlocks",
        "question": "Explain Dijkstra's Banker's Algorithm for deadlock avoidance and how the Need matrix is computed.",
        "expected_keywords": ["banker", "safety", "need = max - allocation", "available", "work"],
        "expected_doc": "os_deadlocks"
    },
    {
        "id": "bench_06",
        "category": "Virtual Memory",
        "question": "What is Belady's Anomaly and which page replacement algorithm is known to suffer from it?",
        "expected_keywords": ["belady", "anomaly", "fifo", "frames", "page fault rate"],
        "expected_doc": "os_virtual_memory"
    },
    {
        "id": "bench_07",
        "category": "File Systems",
        "question": "Compare contiguous allocation with indexed allocation in file systems.",
        "expected_keywords": ["contiguous", "indexed", "fragmentation", "index block", "direct access"],
        "expected_doc": "os_file_systems"
    },
    {
        "id": "bench_08",
        "category": "CPU Scheduling",
        "question": "Why is Shortest Job First (SJF) scheduling considered optimal and what is its preemptive form called?",
        "expected_keywords": ["sjf", "optimal", "minimum average waiting time", "srtf", "remaining time"],
        "expected_doc": "os_cpu_scheduling"
    }
]


def run_benchmark_suite(
    provider: str = "Groq",
    sample_size: Optional[int] = None,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes benchmark questions through the complete RAG and validation pipeline.
    Calculates quantitative metrics across retrieval relevance, citation resolution,
    grounding score, and latency.
    """
    library = DocumentLibrary()
    passages = library.get_passages_for_docs()

    items = BENCHMARK_DATASET[:sample_size] if sample_size else BENCHMARK_DATASET
    total_items = len(items)

    results = []
    total_retrieval_hits = 0
    total_citation_hits = 0
    total_grounding_score = 0.0
    total_latency = 0.0

    for idx, item in enumerate(items):
        start_t = time.time()
        q = item["question"]

        # Run full tutor response pipeline
        res = generate_tutor_response(
            question=q,
            all_passages=passages,
            conversation_history=[],
            answer_length="Medium",
            provider=provider,
            top_k=4
        )
        elapsed = round((time.time() - start_t) * 1000, 1) # ms
        total_latency += elapsed

        # Evaluate Retrieval Relevance
        retrieved = res.get("retrieved_passages", [])
        expected_doc = item["expected_doc"].lower()
        has_relevant_doc = any(expected_doc in p["doc_id"].lower() or expected_doc in p["doc_name"].lower() for p in retrieved)
        if has_relevant_doc:
            total_retrieval_hits += 1

        # Evaluate Keywords and Citations
        citations = res.get("citations", [])
        citations_valid = len(citations) > 0 and all(c.get("verified", False) for c in citations)
        if citations_valid:
            total_citation_hits += 1

        # Grounding
        grounding_info = res.get("grounding", {})
        g_score = grounding_info.get("grounding_score", 0.0)
        total_grounding_score += g_score

        # Keyword match in answer
        ans_text = res.get("answer", "").lower()
        matched_kw = [kw for kw in item["expected_keywords"] if kw in ans_text]
        kw_ratio = round(len(matched_kw) / len(item["expected_keywords"]), 2)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "success": res.get("success", False),
            "retrieval_relevant": has_relevant_doc,
            "citation_valid": citations_valid,
            "citations_count": len(citations),
            "grounding_score": g_score,
            "keyword_coverage": kw_ratio,
            "matched_keywords": matched_kw,
            "latency_ms": elapsed,
            "answer_preview": res.get("answer", "")[:180] + "...",
            "citations": citations
        })

        if progress_callback:
            progress_callback(idx + 1, total_items)

    avg_retrieval = round(total_retrieval_hits / total_items * 100, 1)
    avg_citations = round(total_citation_hits / total_items * 100, 1)
    avg_grounding = round((total_grounding_score / total_items) * 100, 1)
    avg_latency = round(total_latency / total_items, 1)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "total_evaluated": total_items,
        "retrieval_relevance_rate": avg_retrieval,
        "citation_correctness_rate": avg_citations,
        "grounding_quality_score": avg_grounding,
        "average_latency_ms": avg_latency,
        "test_cases": results
    }

    # Save benchmark history
    try:
        with open(BENCHMARK_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    return summary


def load_latest_benchmark_results() -> Optional[Dict[str, Any]]:
    if BENCHMARK_RESULTS_FILE.exists():
        try:
            with open(BENCHMARK_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
